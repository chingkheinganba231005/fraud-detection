"""Metrics for a review list: what an investigation team finds if it checks the top K."""

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def top_k(score: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Indices of the k highest scores. Ties are broken at random, not by row order."""
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(score))
    order = np.lexsort((jitter, -score))
    return order[:k]


def at_budget(y: np.ndarray, score: np.ndarray, k: int, value: np.ndarray | None = None) -> dict:
    idx = top_k(score, k)
    hits = int(y[idx].sum())
    pos = int(y.sum())
    base = pos / len(y)
    out = {"k": k, "hits": hits, "precision": hits / k, "recall": hits / pos if pos else 0.0,
           "lift": (hits / k) / base if base else 0.0,
           "precision_ci": list(wilson(hits, k)),
           "expected_hits_random": k * base}
    if value is not None:
        out["value_found"] = float(value[idx][y[idx] == 1].sum())
        out["value_found_random"] = float(value[y == 1].sum() * k / len(y))
    return out


def wilson(hits: int, n: int, level: float = 0.95) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(0.5 + level / 2)
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summary(y: np.ndarray, score: np.ndarray, budgets, value=None) -> dict:
    return {"roc_auc": float(roc_auc_score(y, score)),
            "average_precision": float(average_precision_score(y, score)),
            "base_rate": float(y.mean()), "positives": int(y.sum()), "n": int(len(y)),
            "budgets": {str(k): at_budget(y, score, k, value) for k in budgets}}


def bootstrap(y: np.ndarray, score: np.ndarray, budgets, n_boot: int, seed: int) -> dict:
    """Confidence intervals from resampling the positives.

    With about a million negatives and under a hundred positives, nearly all of the
    sampling uncertainty is in which positives happened to occur, so the negatives are held
    fixed and only the positives are resampled. Each positive's rank among the negatives is
    computed once, which keeps 2,000 resamples cheap.
    """
    rng = np.random.default_rng(seed)
    pos = np.sort(score[y == 1])
    neg = np.sort(score[y == 0])
    # Negatives scoring above each positive (ties count half) and the review thresholds.
    above = len(neg) - (np.searchsorted(neg, pos, "left") + np.searchsorted(neg, pos, "right")) / 2
    order_all = np.sort(score)[::-1]
    thresholds = {k: order_all[k - 1] for k in budgets}
    auc, ap, rec = [], [], {k: [] for k in budgets}
    n_pos = len(pos)
    for _ in range(n_boot):
        i = rng.integers(0, n_pos, n_pos)
        p, a = pos[i], above[i]
        auc.append(1 - a.mean() / len(neg))
        ps = np.sort(p)[::-1]
        a_sorted = a[np.argsort(-p, kind="stable")]
        ap.append(np.mean(np.arange(1, n_pos + 1) / (np.arange(1, n_pos + 1) + a_sorted)))
        for k in budgets:
            rec[k].append(np.mean(ps >= thresholds[k]))

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    return {"roc_auc_ci": ci(auc), "average_precision_ci": ci(ap),
            "recall_ci": {str(k): ci(rec[k]) for k in budgets}}
