"""Fit every detector on the training years and choose one on the validation year.

The choice rule is fixed before looking at the test year: the detector that finds the most
later-excluded providers in the top 1,000 of the validation year, with average precision as
the tie-break. The test year is not loaded here.

Usage: python scripts/03_train.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud import config, db, evaluation, models  # noqa: E402


def main() -> None:
    t0 = time.time()
    con = db.connect()
    train = db.load_split(con, config.TRAIN_YEARS)
    val = db.load_split(con, (config.VAL_YEAR,))
    print(f"train {len(train):,} rows ({int(train.label.sum())} positive), "
          f"val {len(val):,} rows ({int(val.label.sum())} positive)")
    prep = models.Prep.fit(train)
    y_val = val["label"].to_numpy()
    budgets = config.REVIEW_BUDGETS
    scores, report = {}, {}

    scores["peer rule"] = models.peer_rule_score(val)
    iforest = models.fit_isolation_forest(train, prep)
    scores["isolation forest"] = models.isolation_forest_score(val, prep, iforest)
    print(f"unsupervised done ({time.time() - t0:.0f} s)")

    best_lr = None
    for c in (0.01, 0.1, 1.0):
        fitted = models.fit_logistic(train, prep, c)
        s = models.logistic_score(val, prep, fitted)
        ap = evaluation.summary(y_val, s, budgets)["average_precision"]
        report[f"logistic C={c}"] = ap
        if best_lr is None or ap > best_lr[0]:
            best_lr = (ap, c, fitted, s)
    scores["logistic regression"] = best_lr[3]
    print(f"logistic done ({time.time() - t0:.0f} s)")

    best_gb = None
    for leaves in (7, 15, 31):
        for min_child in (50, 200):
            booster = models.fit_lightgbm(train, val, prep,
                                          {"num_leaves": leaves, "min_child_samples": min_child})
            s = models.lightgbm_score(val, prep, booster)
            ap = evaluation.summary(y_val, s, budgets)["average_precision"]
            report[f"lightgbm leaves={leaves} min_child={min_child}"] = {
                "average_precision": ap, "rounds": booster.best_iteration}
            print(f"  lightgbm {leaves}/{min_child}: AP {ap:.4f}, {booster.best_iteration} rounds")
            if best_gb is None or ap > best_gb[0]:
                best_gb = (ap, (leaves, min_child), booster, s)
    scores["lightgbm"] = best_gb[3]
    print(f"lightgbm done ({time.time() - t0:.0f} s)")

    positives = train[train["label"] == 1]
    index = models.CaseIndex.build(positives)
    scores["nearest known cases"] = index.score(val)
    scores["lightgbm + peer rule"] = models.rank_average(scores["lightgbm"], scores["peer rule"])

    val_results = {name: evaluation.summary(y_val, s, budgets) for name, s in scores.items()}
    headline = str(config.HEADLINE_BUDGET)
    chosen = max(val_results, key=lambda n: (val_results[n]["budgets"][headline]["hits"],
                                             val_results[n]["average_precision"]))
    for name, r in val_results.items():
        print(f"{name:24s} AUC {r['roc_auc']:.3f}  AP {r['average_precision']:.4f}  "
              f"hits@1000 {r['budgets'][headline]['hits']}/{r['positives']}")
    print(f"chosen on validation: {chosen}")

    config.MODELS.mkdir(exist_ok=True)
    best_gb[2].save_model(str(config.MODELS / "lightgbm.txt"), num_iteration=best_gb[2].best_iteration)
    index.save(config.MODELS / "case_index.npz")
    import joblib
    joblib.dump({"iforest": iforest, "logistic": best_lr[2]}, config.MODELS / "baselines.joblib")
    meta = {"prep": prep.to_json(), "chosen": chosen,
            "lightgbm_params": {"num_leaves": best_gb[1][0], "min_child_samples": best_gb[1][1],
                                "rounds": best_gb[2].best_iteration},
            "logistic_c": best_lr[1], "features": list(config.FEATURES)}
    (config.MODELS / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    out = {"train_rows": len(train), "train_positives": int(train.label.sum()),
           "train_positive_providers": int(positives.npi.nunique()),
           "val_rows": len(val), "val_positives": int(y_val.sum()),
           "selection_rule": f"most later-excluded providers in the top {headline} of "
                             f"{config.VAL_YEAR}, then average precision",
           "tuning": report, "validation": val_results, "chosen": chosen,
           "indexed_cases": len(index.info), "seconds": round(time.time() - t0)}
    (config.RESULTS / "validation.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
