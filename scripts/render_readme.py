"""Fill the README from results/*.json, so every number in it comes from a script.

  python scripts/render_readme.py          # write README.md from docs/README.template.md
  python scripts/render_readme.py --check  # fail if README.md is out of date
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ORDER = ("lightgbm", "lightgbm + peer rule", "logistic regression", "nearest known cases",
         "peer rule", "isolation forest")


def load(name):
    return json.loads((RESULTS / name).read_text())


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def num(x):
    return f"{x:,.0f}"


def money_m(x):
    return f"${x / 1e6:,.1f} million"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def context() -> dict:
    ds, val, test, st = load("dataset.json"), load("validation.json"), load("test.json"), load("stress_test.json")
    k = str(test["headline"]["budget"])
    chosen = test["chosen"]
    det = test["detectors"][chosen]
    b = det["budgets"][k]
    h = test["headline"]
    years = ds["years"]
    train_years = [y for y in years if int(y) < 2019]
    c = {
        "budget": num(int(k)),
        "chosen": chosen,
        "test_providers": num(h["providers"]),
        "test_positives": str(h["positives"]),
        "hits": str(b["hits"]),
        "precision": pct(b["precision"], 1),
        "precision_ci": f"{pct(b['precision_ci'][0])}–{pct(b['precision_ci'][1])}",
        "recall": pct(b["recall"], 0),
        "recall_ci": f"{pct(det['recall_ci'][k][0], 0)}–{pct(det['recall_ci'][k][1], 0)}",
        "lift": f"{b['lift']:,.0f}",
        "random_hits": f"{b['expected_hits_random']:.2f}",
        "share_reviewed": pct(h["share_reviewed"], 2),
        "auc": f"{det['roc_auc']:.3f}",
        "auc_ci": f"{det['roc_auc_ci'][0]:.3f}–{det['roc_auc_ci'][1]:.3f}",
        "ap": f"{det['average_precision']:.3f}",
        "ap_ci": f"{det['average_precision_ci'][0]:.3f}–{det['average_precision_ci'][1]:.3f}",
        "value_found": money_m(b["value_found"]),
        "value_random": money_m(b["value_found_random"]),
        "value_all_pos": money_m(h["payment_of_all_positives"]),
        "lead_median": f"{h['months_from_end_of_year_to_exclusion']['median']:.0f}",
        "lead_min": f"{h['months_from_end_of_year_to_exclusion']['min']:.0f}",
        "lead_max": f"{h['months_from_end_of_year_to_exclusion']['max']:.0f}",
        "train_rows": num(val["train_rows"]),
        "train_positives": str(val["train_positives"]),
        "train_positive_providers": str(val["train_positive_providers"]),
        "val_rows": num(val["val_rows"]),
        "val_positives": str(val["val_positives"]),
        "indexed_cases": str(val["indexed_cases"]),
        "all_rows": num(sum(v["providers"] for v in years.values())),
        "dropped_prior": num(sum(v["dropped_already_excluded"] for v in years.values())),
        "dropped_other": num(sum(v["dropped_non_fraud_exclusion"] for v in years.values())),
        "train_years": f"{train_years[0]}–{train_years[-1]}",
        "rural_pop": pct(test["mix"]["rural_share_population"], 0),
        "rural_list": pct(test["mix"]["rural_share_list"], 0),
        "org_pop": pct(test["mix"]["organisation_share_population"], 1),
        "org_list": pct(test["mix"]["organisation_share_list"], 1),
        "stress_n": num(st["providers"]),
    }
    rows = []
    for name in ORDER:
        r = test["detectors"][name]
        bb = r["budgets"][k]
        rows.append([("**" + name + "**") if name == chosen else name,
                     f"{r['roc_auc']:.3f}", f"{r['average_precision']:.3f}",
                     f"{bb['hits']} of {r['positives']}", pct(bb["precision"]),
                     f"{bb['lift']:,.0f}×", money_m(bb["value_found"])])
    c["detector_table"] = table(["detector", "ROC AUC", "average precision",
                                 f"found in top {num(int(k))}", "precision", "lift over random",
                                 "2022 Medicare payments to the providers found"], rows)
    vrows = []
    for name in ORDER:
        r = val["validation"][name]
        vrows.append([name, f"{r['roc_auc']:.3f}", f"{r['average_precision']:.3f}",
                      f"{r['budgets'][k]['hits']} of {r['positives']}"])
    c["validation_table"] = table(["detector", "ROC AUC", "average precision",
                                   f"found in top {num(int(k))}"], vrows)
    brows = []
    for kk, bb in det["budgets"].items():
        ci = det["recall_ci"][kk]
        brows.append([num(int(kk)), str(bb["hits"]), pct(bb["recall"], 0),
                      f"{pct(ci[0], 0)}–{pct(ci[1], 0)}", pct(bb["precision"]), f"{bb['lift']:,.0f}×"])
    c["budget_table"] = table(["providers reviewed", "later excluded", "recall", "95% CI",
                               "precision", "lift"], brows)
    spec = ", ".join(f"{name.lower()} ({n})" for name, n in h["hits_by_specialty"][:5])
    c["hit_specialties"] = spec
    srows = []
    dets = ("lightgbm", "peer rule", "isolation forest", "lightgbm + peer rule")
    for scheme, by_f in st["results"].items():
        for f in ("1.0", "2.0", "5.0"):
            srows.append([scheme, f"×{float(f):g}"] + [pct(by_f[f][d]["detected"], 0) for d in dets])
    c["stress_table"] = table(["scheme", "strength"] + list(dets), srows)
    return c


def render() -> str:
    text = (ROOT / "docs" / "README.template.md").read_text()
    c = context()
    missing = set(re.findall(r"{{(\w+)}}", text)) - set(c)
    if missing:
        sys.exit(f"template uses unknown keys: {sorted(missing)}")
    return re.sub(r"{{(\w+)}}", lambda m: c[m.group(1)], text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out = render()
    readme = ROOT / "README.md"
    if args.check:
        if not readme.exists() or readme.read_text() != out:
            sys.exit("README.md is out of date: run python scripts/render_readme.py")
        print("README.md is up to date")
    else:
        readme.write_text(out)
        print("wrote README.md")


if __name__ == "__main__":
    main()
