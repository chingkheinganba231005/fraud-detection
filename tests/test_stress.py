import pandas as pd
import pytest

from fraud import db, stress
from test_pipeline import _peer_stats, _rows


def _features(scheme, f):
    b, d = _rows(60)
    d = pd.DataFrame({"year": 2022, "npi": b.npi, "d_claims": 1000.0, "d_cost": 50000.0,
                      "d_days": 30000.0, "d_benes": 90.0, "d_brand_claims": 200.0,
                      "d_opioid_claims": 50.0, "d_opioid_la_claims": 5.0,
                      "d_antibiotic_claims": 30.0, "d_antipsychotic_claims": 0.0,
                      "d_lis_claims": 100.0})
    stats = _peer_stats(b, d)
    b2, d2 = stress.SCHEMES[scheme](b.iloc[[0]], d.iloc[[0]], f)
    return db.features_from_raw(b2, d2, stats).iloc[0]


def test_phantom_services_raise_services_per_patient_only():
    f = _features("phantom services", 2.0)
    assert f["services_per_bene"] == pytest.approx(6.0)
    assert f["payment_per_service"] == pytest.approx(1000 / 300)


def test_upcoding_raises_payment_per_service_only():
    f = _features("upcoding", 2.0)
    assert f["payment_per_service"] == pytest.approx(2000 / 300)
    assert f["services_per_bene"] == pytest.approx(3.0)


def test_patient_mill_keeps_every_ratio():
    f = _features("patient mill", 3.0)
    assert f["services_per_bene"] == pytest.approx(3.0)
    assert f["payment_per_bene"] == pytest.approx(10.0)
    assert f["z_log_b_payment"] > 0


def test_opioid_scheme_raises_opioid_share():
    f = _features("opioid prescribing", 3.0)
    assert f["d_opioid_share"] == pytest.approx((50 + 200) / 1200)


def test_no_change_at_strength_one():
    f = _features("phantom services", 1.0)
    assert f["z_services_per_bene"] == pytest.approx(0.0)
