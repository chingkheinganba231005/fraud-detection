"""The service on the committed model files, checked against the batch scorer."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud import db
from fraud.scoring import Scorer
from conftest import DATA

PROVIDER = {
    "part_b": {"provider_type": "Internal Medicine", "entity": "I", "state": "NY", "ruca": 1,
               "participating": "Y", "b_codes": 25, "b_benes": 400, "b_services": 1500,
               "b_charges": 450000, "b_allowed": 150000, "b_payment": 115000,
               "b_drug_payment": 0, "bene_age": 74, "bene_lt65": 40, "bene_gt84": 60,
               "bene_female": 230, "bene_dual": 80, "risk_score": 1.4},
    "part_d": {"d_claims": 3000, "d_cost": 150000, "d_days": 90000, "d_benes": 350,
               "d_brand_claims": 600, "d_opioid_claims": 150, "d_opioid_la_claims": 10,
               "d_antibiotic_claims": 80, "d_antipsychotic_claims": 0, "d_lis_claims": 700},
}


@pytest.fixture(scope="module")
def client():
    from fraud.service import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_score_returns_a_decision_reasons_and_similar_cases(client):
    body = client.post("/score", json=PROVIDER).json()
    assert 0 <= body["percentile"] <= 1
    assert isinstance(body["on_review_list"], bool)
    assert len(body["similar_known_cases"]) == 5
    assert all(c["exclusion_type"] for c in body["similar_known_cases"])


def test_extreme_billing_ranks_higher(client):
    normal = client.post("/score", json=PROVIDER).json()["percentile"]
    heavy = {**PROVIDER, "part_b": {**PROVIDER["part_b"], "b_services": 15000,
                                    "b_payment": 1150000, "b_allowed": 1500000,
                                    "b_charges": 4500000}}
    assert client.post("/score", json=heavy).json()["percentile"] >= normal


def test_service_matches_batch_scoring_on_a_real_provider(client):
    """A real 2022 provider from the sample scores the same through the API as in batch."""
    import duckdb
    from fraud import config
    con = duckdb.connect()
    old = config.RAW
    config.RAW = DATA
    db.run(con, "01_load.sql")
    config.RAW = old
    b = con.execute("SELECT * FROM partb WHERE year = 2022 ORDER BY npi LIMIT 1").df()
    d = con.execute(f"SELECT * FROM partd WHERE year = 2022 AND npi = {int(b.npi[0])}").df()
    stats = pd.read_parquet(config.MODELS / "peer_stats.parquet")
    batch = Scorer().score(db.features_from_raw(b, d, stats))[0]
    payload = {"part_b": {k: (None if pd.isna(v) else v) for k, v in
                          b.drop(columns=["year", "npi"]).iloc[0].to_dict().items()}}
    if len(d):
        payload["part_d"] = {k: (None if pd.isna(v) else v) for k, v in
                             d.drop(columns=["year", "npi"]).iloc[0].to_dict().items()}
    api = client.post("/score", json=payload).json()["score"]
    assert np.isclose(api, batch)
