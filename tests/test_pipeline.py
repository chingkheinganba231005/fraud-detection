import csv

import pandas as pd
import pytest

from fraud import config, db
from conftest import DATA


def test_label_windows_do_not_overlap():
    w = config.LABEL_WINDOW_YEARS
    assert max(config.TRAIN_YEARS) + w < config.VAL_YEAR + 1
    assert config.VAL_YEAR + w < config.TEST_YEAR + 1


def test_every_sample_row_is_loaded(sample_db):
    for y in (2017, 2022):
        with open(DATA / f"partb_{y}.csv") as f:
            n = sum(1 for _ in csv.reader(f)) - 1
        got = sample_db.execute(f"SELECT count(*) FROM partb WHERE year = {y}").fetchone()[0]
        assert got == n


def test_positive_labels_are_fraud_exclusions_inside_the_window(sample_db):
    rows = sample_db.execute("""
        SELECT year, exclusion_type, exclusion_date FROM labels
        WHERE label = 1 AND drop_reason IS NULL
    """).fetchall()
    assert rows, "the sample should contain later-excluded providers"
    for year, etype, date in rows:
        assert etype in config.FRAUD_EXCLUSION_TYPES
        assert year < date.year <= year + config.LABEL_WINDOW_YEARS


def test_providers_excluded_before_the_year_are_dropped(sample_db):
    bad = sample_db.execute("""
        SELECT count(*) FROM labels l JOIN leie e USING (npi)
        WHERE e.excldate <= make_date(l.year, 12, 31) AND l.drop_reason IS NULL
    """).fetchone()[0]
    assert bad == 0


def test_non_fraud_exclusions_are_not_negatives(sample_db):
    bad = sample_db.execute(f"""
        SELECT count(*) FROM labels l JOIN leie e USING (npi)
        WHERE l.label = 0 AND l.drop_reason IS NULL
          AND e.excltype NOT IN ({", ".join(f"'{t}'" for t in config.FRAUD_EXCLUSION_TYPES)})
          AND e.excldate BETWEEN make_date(l.year + 1, 1, 1)
                             AND make_date(l.year + {config.LABEL_WINDOW_YEARS}, 12, 31)
    """).fetchone()[0]
    assert bad == 0


def test_features_have_one_row_per_provider_year(sample_db):
    n_py = sample_db.execute("SELECT count(*) FROM provider_year").fetchone()[0]
    n_z = sample_db.execute("SELECT count(*) FROM peer_z").fetchone()[0]
    dup = sample_db.execute("SELECT count(*) - count(DISTINCT (year, npi)) FROM peer_z").fetchone()[0]
    assert n_py == n_z and dup == 0


def _rows(n, provider_type="Internal Medicine", payment=1000.0):
    b = pd.DataFrame({
        "year": 2022, "npi": range(1, n + 1), "provider_type": provider_type, "entity": "I",
        "state": "NY", "ruca": 1.0, "participating": "Y", "b_codes": 10.0, "b_benes": 100.0,
        "b_services": 300.0, "b_charges": 3000.0, "b_allowed": 1500.0, "b_payment": payment,
        "b_drug_payment": 0.0, "bene_age": 72.0, "bene_lt65": 10.0, "bene_gt84": 10.0,
        "bene_female": 50.0, "bene_dual": 20.0, "risk_score": 1.2})
    d = pd.DataFrame({"year": pd.Series(dtype="int64"), "npi": pd.Series(dtype="int64")})
    for c in ["d_claims", "d_cost", "d_days", "d_benes", "d_brand_claims", "d_opioid_claims",
              "d_opioid_la_claims", "d_antibiotic_claims", "d_antipsychotic_claims", "d_lis_claims"]:
        d[c] = pd.Series(dtype="float64")
    return b, d


def _peer_stats(b, d):
    import duckdb
    con = duckdb.connect()
    con.register("b_in", b)
    con.register("d_in", d)
    con.execute("CREATE TABLE partb AS SELECT * FROM b_in")
    con.execute("CREATE TABLE partd AS SELECT * FROM d_in")
    db.run(con, "03_provider_year.sql")
    db.run(con, "04_peer_stats.sql")
    return con.execute("SELECT * FROM peer_stats").df()


def test_ratios_and_peer_z_scores():
    b, d = _rows(60)
    b.loc[b.npi % 2 == 0, "b_payment"] = 3000.0  # peers pay 1000 or 3000 per 100 patients
    stats = _peer_stats(b, d)
    probe = b.iloc[[0]].copy()
    probe["b_payment"] = 6000.0
    f = db.features_from_raw(probe, d, stats).iloc[0]
    assert f["payment_per_bene"] == pytest.approx(60.0)
    # Peer median payment per patient is 20 and the IQR is 20, so 60 is 2 IQRs above.
    assert f["z_payment_per_bene"] == pytest.approx(2.0)
    assert f["has_part_d"] == 0 and pd.isna(f["z_d_opioid_share"])


def test_small_peer_groups_fall_back_to_all_providers():
    b, d = _rows(60)
    b.loc[b.npi % 2 == 0, "b_payment"] = 3000.0
    small, _ = _rows(5, provider_type="Rare Specialty", payment=2000.0)
    small["npi"] = small["npi"] + 1000
    allb = pd.concat([b, small], ignore_index=True)
    stats = _peer_stats(allb, d)
    probe = small.iloc[[0]].copy()
    probe["b_payment"] = 6000.0
    f = db.features_from_raw(probe, d, stats).iloc[0]
    overall = stats[(stats.provider_type == "__all__")].iloc[0]
    expected = (60.0 - overall["med__payment_per_bene"]) / overall["iqr__payment_per_bene"]
    assert f["z_payment_per_bene"] == pytest.approx(expected)


def test_missing_metrics_stay_missing_after_clipping():
    """DuckDB's least() and greatest() ignore NULLs; a missing ratio must not become 50."""
    b, d = _rows(60)
    stats = _peer_stats(b, d)
    f = db.features_from_raw(b.iloc[[0]], d, stats).iloc[0]
    for c in ("z_d_opioid_share", "z_d_cost_per_claim", "z_log_d_claims"):
        assert pd.isna(f[c])


def test_zero_iqr_falls_back_to_standard_deviation():
    b, d = _rows(60)
    b.loc[b.npi <= 10, "b_drug_payment"] = 100.0  # most peers bill no drugs, so the IQR is 0
    stats = _peer_stats(b, d)
    probe = b.iloc[[20]].copy()
    probe["b_drug_payment"] = 500.0
    f = db.features_from_raw(probe, d, stats).iloc[0]
    row = stats[stats.provider_type == "Internal Medicine"].iloc[0]
    assert row["iqr__drug_payment_share"] == 0
    expected = (0.5 - row["med__drug_payment_share"]) / row["sd__drug_payment_share"]
    assert f["z_drug_payment_share"] == pytest.approx(expected)
    assert -50 < f["z_drug_payment_share"] < 50 or expected >= 50
