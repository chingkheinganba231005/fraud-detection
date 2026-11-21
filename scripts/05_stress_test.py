"""Inject synthetic fraud schemes into real 2022 providers and see which detectors notice.

A copy counts as detected when its score would put it on the top-1,000 review list of the
clean 2022 population. Writes results/stress_test.json and figures/stress_test.png.

Usage: python scripts/05_stress_test.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud import config, db, evaluation, plots, stress  # noqa: E402
from fraud.scoring import Scorer  # noqa: E402

N_PROVIDERS = 2000
DETECTORS = ("lightgbm", "peer rule", "isolation forest", "lightgbm + peer rule")


def main() -> None:
    t0 = time.time()
    con = db.connect()
    year = config.TEST_YEAR
    scorer = Scorer()
    test = db.load_split(con, (year,))
    clean = scorer.all_scores(test)
    k = config.HEADLINE_BUDGET
    thresholds = {d: float(np.sort(clean[d])[::-1][k - 1]) for d in DETECTORS}

    npis = stress.sample_providers(con, year, N_PROVIDERS, config.SEED)
    ids = ", ".join(str(n) for n in npis)
    b = con.execute(f"SELECT * FROM partb WHERE year = {year} AND npi IN ({ids})").df()
    d = con.execute(f"SELECT * FROM partd WHERE year = {year} AND npi IN ({ids})").df()
    peer = con.execute(f"SELECT * FROM peer_stats WHERE year = {year}").df()

    table = {}
    for scheme, inject in stress.SCHEMES.items():
        table[scheme] = {}
        for f in stress.STRENGTHS:
            b2, d2 = inject(b, d, f)
            feats = db.features_from_raw(b2, d2, peer)
            scores = scorer.all_scores(feats)
            row = {}
            for det in DETECTORS:
                hit = int((scores[det] >= thresholds[det]).sum())
                row[det] = {"detected": hit / len(feats), "ci": list(evaluation.wilson(hit, len(feats)))}
            table[scheme][str(f)] = row
            print(f"{scheme:20s} x{f:<4g} " + "  ".join(f"{det} {row[det]['detected']:.1%}"
                                                    for det in DETECTORS))
    out = {"year": year, "providers": len(npis), "budget": k,
           "sample": "never-excluded 2022 providers with at least 100 prescriptions, from "
                     "specialties with at least 1,000 providers",
           "schemes": {name: fn.__doc__.strip() for name, fn in stress.SCHEMES.items()},
           "results": table, "seconds": round(time.time() - t0)}
    (config.RESULTS / "stress_test.json").write_text(json.dumps(out, indent=2) + "\n")
    plots.stress_heatmap(table, DETECTORS, config.FIGURES / "stress_test.png")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
