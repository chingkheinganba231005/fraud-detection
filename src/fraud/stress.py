"""Synthetic fraud schemes injected into real provider records.

The real labels only cover providers that were caught and excluded, so they cannot say
whether a detector would notice a scheme that was rarely prosecuted. Here real 2022
providers who were never excluded are copied, one known scheme is applied to their billing,
and the copies are scored against the clean 2022 population. The copies are labelled
synthetic and are never used for training or for the headline results.
"""

import numpy as np
import pandas as pd

B_MONEY = ["b_charges", "b_allowed", "b_payment"]
B_PATIENTS = ["b_benes", "bene_lt65", "bene_gt84", "bene_female", "bene_dual"]


def phantom_services(b: pd.DataFrame, d: pd.DataFrame, f: float):
    """Billing for services that were never given: more services per patient."""
    b = b.copy()
    for c in ["b_services"] + B_MONEY:
        b[c] = b[c] * f
    return b, d


def upcoding(b: pd.DataFrame, d: pd.DataFrame, f: float):
    """Billing each service as a more expensive one: higher payment per service."""
    b = b.copy()
    for c in B_MONEY:
        b[c] = b[c] * f
    return b, d


def patient_mill(b: pd.DataFrame, d: pd.DataFrame, f: float):
    """More patients with the same pattern per patient: only the volume changes."""
    b = b.copy()
    for c in B_PATIENTS + ["b_services"] + B_MONEY + ["b_drug_payment"]:
        b[c] = b[c] * f
    return b, d


def opioid_mill(b: pd.DataFrame, d: pd.DataFrame, f: float):
    """Extra opioid prescriptions. The added count is (f - 1) times 10% of the provider's
    prescriptions, at the provider's own cost per prescription."""
    d = d.copy()
    extra = (f - 1) * 0.1 * d["d_claims"].fillna(0)
    cost_per = (d["d_cost"] / d["d_claims"]).fillna(0)
    d["d_opioid_claims"] = d["d_opioid_claims"].fillna(0) + extra
    d["d_claims"] = d["d_claims"] + extra
    d["d_cost"] = d["d_cost"] + extra * cost_per
    return b, d


SCHEMES = {
    "phantom services": phantom_services,
    "upcoding": upcoding,
    "patient mill": patient_mill,
    "opioid prescribing": opioid_mill,
}
STRENGTHS = (1.0, 1.5, 2.0, 3.0, 5.0)


def sample_providers(con, year: int, n: int, seed: int, min_peers: int = 1000) -> list:
    """Random never-excluded providers of the given year from specialties with at least
    min_peers providers, so the peer statistics they are compared with are stable."""
    rows = con.execute(f"""
        WITH big AS (SELECT provider_type FROM provider_year WHERE year = {year}
                     GROUP BY 1 HAVING count(*) >= {min_peers})
        SELECT p.npi FROM provider_year p
        JOIN labels l USING (year, npi)
        JOIN partd d USING (year, npi)
        WHERE p.year = {year} AND l.label = 0 AND l.drop_reason IS NULL
          AND p.provider_type IN (SELECT provider_type FROM big)
          AND d.d_claims >= 100
    """).fetchall()
    rng = np.random.default_rng(seed)
    npis = np.array([r[0] for r in rows])
    return sorted(rng.choice(npis, size=min(n, len(npis)), replace=False).tolist())
