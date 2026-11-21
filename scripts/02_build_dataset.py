"""Load the raw files into DuckDB, label every provider-year and build the features.

Writes data/fraud.duckdb and results/dataset.json (row counts, positives, dropped rows).

Usage: python scripts/02_build_dataset.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud import config, db  # noqa: E402


def main() -> None:
    t0 = time.time()
    con = db.connect()
    for name in ("01_load.sql", "02_labels.sql", "03_provider_year.sql", "04_peer_stats.sql",
                 "05_peer_z.sql"):
        db.run(con, name)
        print(f"{name} done ({time.time() - t0:.0f} s)")

    summary = {"label_window_years": config.LABEL_WINDOW_YEARS,
               "fraud_exclusion_types": config.FRAUD_EXCLUSION_TYPES, "years": {}}
    rows = con.execute("""
        SELECT l.year,
               count(*)                                          AS providers,
               count(*) FILTER (WHERE l.drop_reason IS NULL)     AS kept,
               count(*) FILTER (WHERE l.label = 1 AND l.drop_reason IS NULL) AS positives,
               count(*) FILTER (WHERE l.drop_reason LIKE 'excluded before%') AS dropped_prior,
               count(*) FILTER (WHERE l.drop_reason LIKE '%non-fraud%')      AS dropped_other,
               sum(p.b_payment)                                  AS part_b_payment,
               count(*) FILTER (WHERE p.has_part_d = 1)          AS with_part_d
        FROM labels l JOIN provider_year p USING (year, npi)
        GROUP BY l.year ORDER BY l.year
    """).fetchall()
    for y, n, kept, pos, prior, other, pay, with_d in rows:
        summary["years"][str(y)] = {
            "providers": n, "kept": kept, "positives": pos,
            "dropped_already_excluded": prior, "dropped_non_fraud_exclusion": other,
            "part_b_payment": pay, "with_part_d": with_d,
        }
        print(y, summary["years"][str(y)])
    types = con.execute("""
        SELECT exclusion_type, count(*) FROM labels
        WHERE label = 1 AND drop_reason IS NULL GROUP BY 1 ORDER BY 2 DESC
    """).fetchall()
    summary["positives_by_exclusion_type"] = {t: n for t, n in types}
    summary["distinct_positive_providers_train"] = con.execute(f"""
        SELECT count(DISTINCT npi) FROM labels
        WHERE label = 1 AND drop_reason IS NULL AND year IN {config.TRAIN_YEARS}
    """).fetchone()[0]
    config.RESULTS.mkdir(exist_ok=True)
    (config.RESULTS / "dataset.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
