"""Inject synthetic fraud schemes into real 2022 providers and see which detectors notice.

A copy counts as detected when its score would put it on a review list of the clean 2022
population: the top 1,000, and the top 1% of providers. Writes results/stress_test.json and figures/stress_test.png.

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
    lists = {"top 1,000": config.HEADLINE_BUDGET, "top 1%": round(0.01 * len(test))}
    # Each list is the top k with ties broken at random. A copy scoring exactly at the cut-off
    # is on the list with the chance a tied clean provider has of making it.
    cutoff = {}
    for name, k in lists.items():
        for det in DETECTORS:
            s = clean[det]
            t = float(np.sort(s)[::-1][k - 1])
            above, tied = int((s > t).sum()), int((s == t).sum())
            cutoff[name, det] = (t, (k - above) / tied)

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
            for name in lists:
                row[name] = {}
                for det in DETECTORS:
                    t, p_tie = cutoff[name, det]
                    hit = float((scores[det] > t).sum() + p_tie * (scores[det] == t).sum())
                    row[name][det] = {"detected": hit / len(feats),
                                      "ci": list(evaluation.wilson(round(hit), len(feats)))}
            table[scheme][str(f)] = row
            print(f"{scheme:20s} x{f:<4g} " + "  ".join(
                f"{det} {row['top 1%'][det]['detected']:.1%}" for det in DETECTORS))
    out = {"year": year, "providers": len(npis), "lists": lists,
           "cutoff": {f"{n} / {d}": {"score": t, "share_of_ties_on_list": p}
                      for (n, d), (t, p) in cutoff.items()},
           "sample": "never-excluded 2022 providers with at least 100 prescriptions, from "
                     "specialties with at least 1,000 providers",
           "schemes": {name: fn.__doc__.strip() for name, fn in stress.SCHEMES.items()},
           "results": table, "seconds": round(time.time() - t0)}
    (config.RESULTS / "stress_test.json").write_text(json.dumps(out, indent=2) + "\n")
    plots.stress_heatmap({sc: {f: v["top 1%"] for f, v in by_f.items()} for sc, by_f in table.items()},
                         DETECTORS, config.FIGURES / "stress_test.png")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
