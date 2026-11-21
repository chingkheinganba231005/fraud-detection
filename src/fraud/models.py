"""Detectors: a peer-outlier rule, an isolation forest, logistic regression, LightGBM and
nearest-case retrieval. Every detector returns a score where higher means more suspicious."""

import json
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import config

# The ratios an investigator would check first when comparing a provider with peers.
RULE_METRICS = ("z_payment_per_bene", "z_services_per_bene", "z_payment_per_service",
                "z_log_b_payment", "z_d_cost_per_claim", "z_d_opioid_share")


def peer_rule_score(df: pd.DataFrame) -> np.ndarray:
    """Largest upward deviation from peers on a short list of billing ratios."""
    z = df[list(RULE_METRICS)].to_numpy(dtype=float)
    return np.nan_to_num(np.nanmax(np.where(np.isnan(z), -np.inf, z), axis=1), neginf=0.0)


def numeric_matrix(df: pd.DataFrame, medians: dict) -> np.ndarray:
    """Numeric inputs with missing values filled: z-scores with 0 (the peer median),
    raw ratios with the training median."""
    cols = list(config.RAW_FEATURES) + list(config.Z_FEATURES)
    out = df[cols].astype(float).copy()
    for c in config.Z_FEATURES:
        out[c] = out[c].fillna(0.0)
    for c in config.RAW_FEATURES:
        out[c] = out[c].fillna(medians[c])
    return out.to_numpy()


def categorical_frame(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    out = df[list(config.FEATURES)].copy()
    for c in config.CATEGORICAL_FEATURES:
        out[c] = pd.Categorical(out[c], categories=categories[c])
    for c in out.columns:
        if c not in config.CATEGORICAL_FEATURES:
            out[c] = out[c].astype(float)
    return out


@dataclass
class Prep:
    """What is learned from the training years to turn a feature table into model inputs."""
    medians: dict
    categories: dict

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "Prep":
        medians = {c: float(np.nanmedian(train[c].astype(float))) for c in config.RAW_FEATURES}
        categories = {c: sorted(train[c].dropna().unique().tolist())
                      for c in config.CATEGORICAL_FEATURES}
        return cls(medians, categories)

    def to_json(self) -> dict:
        return {"medians": self.medians, "categories": self.categories}

    @classmethod
    def from_json(cls, d: dict) -> "Prep":
        return cls(d["medians"], d["categories"])


def fit_isolation_forest(train: pd.DataFrame, prep: Prep, seed: int = config.SEED,
                         n_fit: int = 200_000) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train), size=min(n_fit, len(train)), replace=False)
    x = numeric_matrix(train.iloc[idx], prep.medians)
    scaler = StandardScaler().fit(x)
    model = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1)
    model.fit(scaler.transform(x))
    return scaler, model


def isolation_forest_score(df: pd.DataFrame, prep: Prep, fitted: tuple) -> np.ndarray:
    scaler, model = fitted
    return -model.score_samples(scaler.transform(numeric_matrix(df, prep.medians)))


def downsample(train: pd.DataFrame, neg_fraction: float, seed: int) -> pd.DataFrame:
    """All positives and a random share of negatives. Ranking is unchanged by this."""
    rng = np.random.default_rng(seed)
    keep = (train["label"].to_numpy() == 1) | (rng.random(len(train)) < neg_fraction)
    return train[keep]


def fit_logistic(train: pd.DataFrame, prep: Prep, c: float, seed: int = config.SEED) -> tuple:
    sample = downsample(train, 0.1, seed)
    x = numeric_matrix(sample, prep.medians)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
    model.fit(np.clip(scaler.transform(x), -10, 10), sample["label"].to_numpy())
    return scaler, model


def logistic_score(df: pd.DataFrame, prep: Prep, fitted: tuple) -> np.ndarray:
    scaler, model = fitted
    x = np.clip(scaler.transform(numeric_matrix(df, prep.medians)), -10, 10)
    return model.decision_function(x)


def fit_lightgbm(train: pd.DataFrame, val: pd.DataFrame, prep: Prep, params: dict,
                 seed: int = config.SEED) -> lgb.Booster:
    sample = downsample(train, 0.2, seed)
    dtrain = lgb.Dataset(categorical_frame(sample, prep.categories), sample["label"].to_numpy(),
                         categorical_feature=list(config.CATEGORICAL_FEATURES), free_raw_data=False)
    dval = lgb.Dataset(categorical_frame(val, prep.categories), val["label"].to_numpy(),
                       reference=dtrain)
    full = {"objective": "binary", "metric": "average_precision", "learning_rate": 0.03,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l2": 1.0, "cat_smooth": 50, "min_data_per_group": 200,
            "seed": seed, "verbose": -1, "num_threads": 0, **params}
    return lgb.train(full, dtrain, num_boost_round=2000, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(200, verbose=False)])


def lightgbm_score(df: pd.DataFrame, prep: Prep, booster: lgb.Booster) -> np.ndarray:
    return booster.predict(categorical_frame(df, prep.categories),
                           num_iteration=booster.best_iteration, raw_score=True)


def lightgbm_contributions(df: pd.DataFrame, prep: Prep, booster: lgb.Booster) -> np.ndarray:
    """Per-feature contributions (TreeSHAP) to the raw score, one row per provider."""
    return booster.predict(categorical_frame(df, prep.categories),
                           num_iteration=booster.best_iteration, pred_contrib=True)


@dataclass
class CaseIndex:
    """Vector index of providers that were later excluded for fraud, in peer z-score space.
    For a new provider it returns the most similar known cases as evidence for a reviewer."""
    vectors: np.ndarray
    info: list = field(default_factory=list)

    @staticmethod
    def embed(df: pd.DataFrame) -> np.ndarray:
        z = df[list(config.Z_FEATURES)].astype(float).fillna(0.0).to_numpy()
        # Signed log compresses extreme z-scores so one huge ratio does not dominate.
        z = np.sign(z) * np.log1p(np.abs(z))
        norm = np.linalg.norm(z, axis=1, keepdims=True)
        return z / np.where(norm == 0, 1.0, norm)

    @classmethod
    def build(cls, positives: pd.DataFrame) -> "CaseIndex":
        latest = positives.sort_values("year").groupby("npi").tail(1)
        info = [{"year": int(r.year), "provider_type": r.provider_type,
                 "exclusion_type": r.exclusion_type,
                 "exclusion_reason": config.FRAUD_EXCLUSION_TYPES.get(r.exclusion_type, "")}
                for r in latest.itertuples()]
        return cls(cls.embed(latest), info)

    def similarity(self, df: pd.DataFrame) -> np.ndarray:
        """Cosine similarity of every row to every indexed case."""
        return self.embed(df) @ self.vectors.T

    def score(self, df: pd.DataFrame, k: int = 5) -> np.ndarray:
        """Mean similarity to the k most similar known cases."""
        out = np.empty(len(df))
        for start in range(0, len(df), 50_000):
            s = self.similarity(df.iloc[start:start + 50_000])
            out[start:start + 50_000] = np.sort(s, axis=1)[:, -k:].mean(axis=1)
        return out

    def neighbours(self, row: pd.DataFrame, k: int = 5) -> list:
        s = self.similarity(row)[0]
        top = np.argsort(-s)[:k]
        return [{**self.info[i], "similarity": round(float(s[i]), 3)} for i in top]

    def save(self, path) -> None:
        np.savez_compressed(path, vectors=self.vectors, info=json.dumps(self.info))

    @classmethod
    def load(cls, path) -> "CaseIndex":
        d = np.load(path)
        return cls(d["vectors"], json.loads(str(d["info"])))


def rank_average(*scores: np.ndarray) -> np.ndarray:
    """Average of percentile ranks, so detectors on different scales can be combined."""
    ranks = [pd.Series(s).rank(pct=True).to_numpy() for s in scores]
    return np.mean(ranks, axis=0)
