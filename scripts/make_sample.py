"""Cut a small real sample of the raw files for the tests and CI.

Keeps every provider later excluded for fraud plus a random 1,500 others in two years
(one training year and the test year), the matching OIG rows, and only the columns the
pipeline reads. Names and street addresses are dropped.

Usage: python scripts/make_sample.py
"""

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud import config, db  # noqa: E402

OUT = config.ROOT / "tests" / "data"
YEARS = (2017, 2022)
B_COLS = ["Rndrng_NPI", "Rndrng_Prvdr_Type", "Rndrng_Prvdr_Ent_Cd", "Rndrng_Prvdr_State_Abrvtn",
          "Rndrng_Prvdr_RUCA", "Rndrng_Prvdr_Mdcr_Prtcptg_Ind", "Tot_HCPCS_Cds", "Tot_Benes",
          "Tot_Srvcs", "Tot_Sbmtd_Chrg", "Tot_Mdcr_Alowd_Amt", "Tot_Mdcr_Pymt_Amt",
          "Drug_Mdcr_Pymt_Amt", "Bene_Avg_Age", "Bene_Age_LT_65_Cnt", "Bene_Age_GT_84_Cnt",
          "Bene_Feml_Cnt", "Bene_Dual_Cnt", "Bene_Avg_Risk_Scre"]
D_COLS = ["Prscrbr_NPI", "Tot_Clms", "Tot_Drug_Cst", "Tot_Day_Suply", "Tot_Benes", "Brnd_Tot_Clms",
          "Opioid_Tot_Clms", "Opioid_LA_Tot_Clms", "Antbtc_Tot_Clms", "Antpsyct_GE65_Tot_Clms",
          "LIS_Tot_Clms"]
L_COLS = ["NPI", "EXCLTYPE", "EXCLDATE", "GENERAL", "SPECIALTY"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = db.connect()
    for y in YEARS:
        npis = [r[0] for r in con.execute(f"""
            (SELECT npi FROM labels WHERE year = {y} AND (label = 1 OR drop_reason IS NOT NULL))
            UNION
            (SELECT npi FROM labels WHERE year = {y} USING SAMPLE reservoir(1500 ROWS) REPEATABLE ({config.SEED}))
        """).fetchall()]
        ids = ", ".join(f"'{n}'" for n in npis)
        raw = duckdb.connect()
        raw.execute(f"""COPY (SELECT {", ".join(B_COLS)} FROM read_csv('{config.part_b_path(y)}', all_varchar=true)
                     WHERE Rndrng_NPI IN ({ids}) ORDER BY Rndrng_NPI)
                     TO '{OUT / f"partb_{y}.csv"}' (HEADER)""")
        raw.execute(f"""COPY (SELECT {", ".join(D_COLS)} FROM read_csv('{config.part_d_path(y)}', all_varchar=true)
                     WHERE Prscrbr_NPI IN ({ids}) ORDER BY Prscrbr_NPI)
                     TO '{OUT / f"partd_{y}.csv"}' (HEADER)""")
        print(y, len(npis), "providers")
    all_ids = ", ".join(f"'{r[0]}'" for r in con.execute(
        f"SELECT DISTINCT npi FROM labels WHERE year IN {YEARS}").fetchall())
    duckdb.connect().execute(f"""COPY (SELECT {", ".join(L_COLS)} FROM read_csv('{config.LEIE_PATH}', all_varchar=true)
                 WHERE NPI IN (SELECT CAST(npi AS VARCHAR) FROM (SELECT unnest([{all_ids}]) AS npi)))
                 TO '{OUT / "leie_updated.csv"}' (HEADER)""")


if __name__ == "__main__":
    main()
