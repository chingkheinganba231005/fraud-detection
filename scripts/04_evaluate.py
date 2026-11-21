"""Score the held-out 2022 providers once and measure what a review team would find.

Writes results/test.json, figures, and the reference score distributions the service uses
to turn a score into a percentile.

Usage: python scripts/04_evaluate.py
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud import config, db, evaluation, plots  # noqa: E402
from fraud.scoring import Scorer  # noqa: E402


def main() -> None:
    t0 = time.time()
    con = db.connect()
    test = db.load_split(con, (config.TEST_YEAR,))
    y = test["label"].to_numpy()
    value = (test["b_payment"].fillna(0) + test["d_cost"].fillna(0)).to_numpy()
    scorer = Scorer()
    raw = scorer.detector_scores(test)

    # Reference distributions from this year, used later to turn a score into a percentile.
    ref = {k.replace(" ", "_"): np.sort(v) for k, v in raw.items()}
    np.savez_compressed(config.MODELS / "reference.npz", **ref)
    scorer = Scorer()
    scores = scorer.all_scores(test)
    ref["chosen"] = np.sort(scores[scorer.chosen])
    np.savez_compressed(config.MODELS / "reference.npz", **ref)
    budgets = config.REVIEW_BUDGETS
    k = config.HEADLINE_BUDGET

    results = {"year": config.TEST_YEAR, "chosen": scorer.chosen, "detectors": {}}
    for name, s in scores.items():
        r = evaluation.summary(y, s, budgets, value)
        r.update(evaluation.bootstrap(y, s, budgets, config.N_BOOTSTRAP, config.SEED))
        results["detectors"][name] = r
        print(f"{name:24s} AUC {r['roc_auc']:.3f}  AP {r['average_precision']:.4f}  "
              f"hits@{k} {r['budgets'][str(k)]['hits']}/{r['positives']}")

    chosen = scores[scorer.chosen]
    idx = evaluation.top_k(chosen, k)
    flagged = test.iloc[idx]
    hits = flagged[flagged["label"] == 1]
    threshold = float(np.sort(chosen)[::-1][k - 1])

    # How long before the exclusion the provider would have been on the list.
    end_of_year = np.datetime64(f"{config.TEST_YEAR}-12-31")
    months = ((hits["exclusion_date"].to_numpy().astype("datetime64[D]") - end_of_year)
              .astype(int) / 30.44)
    results["headline"] = {
        "budget": k, "threshold": threshold,
        "hits": len(hits), "positives": int(y.sum()), "providers": len(test),
        "share_reviewed": k / len(test),
        "months_from_end_of_year_to_exclusion": {
            "median": float(np.median(months)) if len(months) else None,
            "min": float(months.min()) if len(months) else None,
            "max": float(months.max()) if len(months) else None},
        "hits_by_specialty": Counter(hits["provider_type"]).most_common(),
        "hits_by_exclusion_type": Counter(hits["exclusion_type"]).most_common(),
        "payment_of_hits": float(value[idx][y[idx] == 1].sum()),
        "payment_of_all_positives": float(value[y == 1].sum()),
    }

    # Which reasons the model gives for the providers it puts on the list.
    reasons = scorer.reasons(flagged)
    top_reason = Counter(r[0]["feature"] for r in reasons if r)
    results["top_reason_on_list"] = top_reason.most_common(10)
    gain = scorer.booster.feature_importance("gain")
    order = np.argsort(-gain)[:12]
    results["lightgbm_gain"] = [(config.FEATURES[i], float(gain[i] / gain.sum())) for i in order]

    # Is the list concentrated in rural areas or by entity type more than the population?
    rural = test["ruca"].fillna(1) >= 4
    results["mix"] = {
        "rural_share_population": float(rural.mean()),
        "rural_share_list": float(rural.iloc[idx].mean()),
        "organisation_share_population": float((test["entity"] == "O").mean()),
        "organisation_share_list": float((flagged["entity"] == "O").mean()),
    }

    (config.MODELS / "threshold.json").write_text(json.dumps(
        {"year": config.TEST_YEAR, "budget": k, "threshold": threshold,
         "providers": len(test)}, indent=2) + "\n")
    stats = con.execute(f"SELECT * FROM peer_stats WHERE year = {config.TEST_YEAR}").df()
    stats.to_parquet(config.MODELS / "peer_stats.parquet", index=False)

    results["seconds"] = round(time.time() - t0)
    (config.RESULTS / "test.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    plots.precision_recall(y, scores, config.FIGURES / "precision_recall.png")
    plots.recall_by_budget(results, config.FIGURES / "recall_by_budget.png")
    plots.importance(results["lightgbm_gain"], config.FIGURES / "importance.png")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
