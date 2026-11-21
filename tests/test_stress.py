import pandas as pd
import pytest

from fraud import db, stress
from test_pipeline import _peer_stats, _rows


def _setup():
    b, d = _rows(60)
    # Spread the peers out so every peer statistic has a non-zero scale.
    b["b_services"] = 300.0 + 5 * b["npi"]
    b["b_payment"] = 1000.0 + 20 * b["npi"]
    b["b_benes"] = 100.0 + b["npi"]
    d = pd.DataFrame({"year": 2022, "npi": b.npi, "d_claims": 1000.0 + 10 * b.npi,
                      "d_cost": 50000.0, "d_days": 30000.0, "d_benes": 90.0,
                      "d_brand_claims": 200.0, "d_opioid_claims": 50.0 + b.npi,
                      "d_opioid_la_claims": 5.0, "d_antibiotic_claims": 30.0,
                      "d_antipsychotic_claims": 0.0, "d_lis_claims": 100.0})
    return b, d, _peer_stats(b, d)


def _features(scheme, f):
    b, d, stats = _setup()
    b2, d2 = stress.SCHEMES[scheme](b.iloc[[30]], d.iloc[[30]], f)
    return db.features_from_raw(b2, d2, stats).iloc[0]


def test_no_change_at_strength_one():
    for scheme in stress.SCHEMES:
        a, b = _features(scheme, 1.0), _features("phantom services", 1.0)
        assert a["payment_per_bene"] == pytest.approx(b["payment_per_bene"])
        assert a["z_services_per_bene"] == pytest.approx(b["z_services_per_bene"])


def test_phantom_services_raise_services_per_patient_only():
    base, f = _features("phantom services", 1.0), _features("phantom services", 2.0)
    assert f["services_per_bene"] == pytest.approx(2 * base["services_per_bene"])
    assert f["payment_per_service"] == pytest.approx(base["payment_per_service"])
    assert f["z_services_per_bene"] > base["z_services_per_bene"]


def test_upcoding_raises_payment_per_service_only():
    base, f = _features("upcoding", 1.0), _features("upcoding", 2.0)
    assert f["payment_per_service"] == pytest.approx(2 * base["payment_per_service"])
    assert f["services_per_bene"] == pytest.approx(base["services_per_bene"])


def test_patient_mill_keeps_every_ratio():
    base, f = _features("patient mill", 1.0), _features("patient mill", 3.0)
    for c in ("services_per_bene", "payment_per_bene", "payment_per_service"):
        assert f[c] == pytest.approx(base[c])
    assert f["z_log_b_payment"] > base["z_log_b_payment"]


def test_opioid_scheme_raises_opioid_share():
    base, f = _features("opioid prescribing", 1.0), _features("opioid prescribing", 3.0)
    assert f["d_opioid_share"] > base["d_opioid_share"]
    assert f["z_d_opioid_share"] > base["z_d_opioid_share"]
