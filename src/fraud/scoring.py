"""One scorer used by the evaluation, the stress test and the service, so all three rank
providers the same way."""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config, models

LABELS = {
    "log_b_benes": "number of Medicare patients",
    "log_b_services": "number of services billed",
    "log_b_payment": "total Medicare payment",
    "log_b_codes": "number of distinct billing codes",
    "services_per_bene": "services per patient",
    "payment_per_bene": "payment per patient",
    "payment_per_service": "payment per service",
    "charge_to_allowed": "charges relative to the allowed amount",
    "codes_per_bene": "billing codes per patient",
    "drug_payment_share": "share of payment for drugs given in the office",
    "share_under_65": "share of patients under 65 (disability)",
    "share_dual": "share of patients also on Medicaid",
    "risk_score": "average patient risk score",
    "log_d_claims": "number of prescriptions",
    "log_d_cost": "total prescription cost",
    "d_cost_per_claim": "cost per prescription",
    "d_claims_per_bene": "prescriptions per patient",
    "d_brand_share": "share of brand-name prescriptions",
    "d_opioid_share": "share of opioid prescriptions",
    "d_opioid_la_share": "share of long-acting opioids",
    "d_antipsychotic_share": "share of antipsychotic prescriptions",
}


def describe(feature: str, value: float) -> str:
    if feature.startswith("z_"):
        name = LABELS.get(feature[2:], feature[2:])
        if np.isnan(value):
            return f"{name}: not reported"
        side = "above" if value >= 0 else "below"
        return f"{name}: {abs(value):.1f} IQRs {side} the median of the same specialty"
    if feature == "provider_type":
        return "specialty"
    name = LABELS.get(feature, feature.replace("_", " "))
    return f"{name}: {value:.3g}" if isinstance(value, (int, float)) and not np.isnan(value) else name


class Scorer:
    def __init__(self, models_dir: Path = config.MODELS):
        self.dir = Path(models_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.prep = models.Prep.from_json(self.meta["prep"])
        self.booster = lgb.Booster(model_file=str(self.dir / "lightgbm.txt"))
        self.booster.best_iteration = 0
        self.index = models.CaseIndex.load(self.dir / "case_index.npz")
        base = joblib.load(self.dir / "baselines.joblib")
        self.iforest, self.logistic = base["iforest"], base["logistic"]
        ref = self.dir / "reference.npz"
        self.reference = dict(np.load(ref)) if ref.exists() else None
        self.chosen = self.meta["chosen"]

    def detector_scores(self, df: pd.DataFrame) -> dict:
        return {
            "peer rule": models.peer_rule_score(df),
            "isolation forest": models.isolation_forest_score(df, self.prep, self.iforest),
            "logistic regression": models.logistic_score(df, self.prep, self.logistic),
            "lightgbm": models.lightgbm_score(df, self.prep, self.booster),
            "nearest known cases": self.index.score(df),
        }

    def percentile(self, name: str, s: np.ndarray) -> np.ndarray:
        """Position of each score in the reference year's distribution, from 0 to 1."""
        ref = self.reference[name.replace(" ", "_")]
        return np.searchsorted(ref, s, side="right") / len(ref)

    def all_scores(self, df: pd.DataFrame) -> dict:
        s = self.detector_scores(df)
        if self.reference is not None:
            s["lightgbm + peer rule"] = (self.percentile("lightgbm", s["lightgbm"])
                                         + self.percentile("peer rule", s["peer rule"])) / 2
        else:
            s["lightgbm + peer rule"] = models.rank_average(s["lightgbm"], s["peer rule"])
        return s

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.all_scores(df)[self.chosen]

    def reasons(self, df: pd.DataFrame, top: int = 3) -> list:
        """Plain-language reasons from the LightGBM contributions, largest first."""
        contrib = models.lightgbm_contributions(df, self.prep, self.booster)[:, :-1]
        names = list(config.FEATURES)
        out = []
        for i in range(len(df)):
            order = np.argsort(-contrib[i])[:top]
            row = df.iloc[i]
            out.append([{"feature": names[j], "contribution": round(float(contrib[i, j]), 3),
                         "text": describe(names[j], row[names[j]]
                                          if names[j] not in config.CATEGORICAL_FEATURES else 0)}
                        for j in order if contrib[i, j] > 0])
        return out
