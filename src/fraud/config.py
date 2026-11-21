"""Paths, data sources, years and constants shared by every script."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("FRAUD_DATA", ROOT / "data"))
RAW = DATA / "raw"
DB_PATH = DATA / "fraud.duckdb"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"
SQL = ROOT / "sql"

SEED = 7

# Each provider-year is labelled with what happened in the two calendar years after it.
# Training years end in 2017, so their label windows end in 2019, before the validation
# window (2020-21) starts. The validation window ends before the test window (2023-24).
TRAIN_YEARS = (2014, 2015, 2016, 2017)
VAL_YEAR = 2019
TEST_YEAR = 2022
YEARS = TRAIN_YEARS + (VAL_YEAR, TEST_YEAR)
LABEL_WINDOW_YEARS = 2

# OIG exclusion authorities that are about fraud. The others (licence loss, patient abuse,
# controlled substances, student loans and so on) are left out of the positive class, and
# providers excluded for them in the window are dropped as ambiguous.
FRAUD_EXCLUSION_TYPES = {
    "1128a1": "conviction of a program-related crime",
    "1128a3": "felony conviction relating to health care fraud",
    "1128b1": "misdemeanor conviction relating to health care fraud",
    "1128b2": "conviction relating to obstruction of an investigation",
    "1128b6": "claims for excessive charges or unnecessary services",
    "1128b7": "fraud, kickbacks and other prohibited activities",
    "1128Aa": "civil monetary penalty",
}

LEIE_URL = "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"

_B = "https://data.cms.gov/sites/default/files/2025-11/"
PART_B_URLS = {
    2014: _B + "da4b3b2b-1ee2-4e25-b2fe-012b880afd37/MUP_PHY_R25_P05_V20_D14_Prov.csv",
    2015: _B + "efb7e2f9-b903-4f54-b57f-742f436c316f/MUP_PHY_R25_P05_V20_D15_Prov.csv",
    2016: _B + "e4cd28e1-2d05-4f33-b232-f32b5bebc153/MUP_PHY_R25_P05_V20_D16_Prov.csv",
    2017: _B + "b7e195ce-710d-4584-8293-787ffc38b017/MUP_PHY_R25_P07_V20_D17_Prov.csv",
    2019: _B + "ac110c46-3429-4f3c-9348-56f0a5312cb8/MUP_PHY_R25_P07_V20_D19_Prov.csv",
    2022: _B + "adcd20c5-4534-43cd-8dfa-881ebe7bacfd/MUP_PHY_R25_P07_V20_D22_Prov.csv",
}
_D = "https://data.cms.gov/sites/default/files/2026-08/"
PART_D_URLS = {
    2014: _D + "e11a5415-1967-4207-8ae3-329cfe3fedef/mup_dpr_ry21_p04_v20_dy14_npi.csv",
    2015: _D + "a846e593-06fb-4ed5-9b2f-b5a31acfef82/mup_dpr_ry21_p04_v20_dy15_npi.csv",
    2016: _D + "e8255bdf-5e85-425c-9e6c-1cc460ccce51/mup_dpr_ry21_p04_v20_dy16_npi.csv",
    2017: _D + "8a65e4c8-3c6a-48f2-99ba-422987d953af/mup_dpr_ry21_p04_v20_dy17_npi.csv",
    2019: _D + "d4a0a871-5e6a-4180-8d89-82b56ff08175/mup_dpr_ry21_p04_v20_dy19_npi.csv",
    2022: _D + "0cdbfc0d-257d-4eb0-bcc8-71e5462ec320/mup_dpr_ry24_p04_v20_dy22_npi.csv",
}


def part_b_path(year: int) -> Path:
    return RAW / f"partb_{year}.csv"


def part_d_path(year: int) -> Path:
    return RAW / f"partd_{year}.csv"


LEIE_PATH = RAW / "leie_updated.csv"

# Review budgets: how many providers an investigation team could look at in a year.
REVIEW_BUDGETS = (100, 500, 1000, 5000)
HEADLINE_BUDGET = 1000
# Review lists as a share of all providers. Added after the first test run showed that the
# fixed lists above are too short to say much at a 1-in-15,000 positive rate; the model and
# its selection were not changed.
REVIEW_SHARES = (0.01, 0.05, 0.10, 0.20)
N_BOOTSTRAP = 2000

# Metrics compared against the provider's peers (same provider type, same year).
PEER_METRICS = (
    "log_b_benes", "log_b_services", "log_b_payment", "log_b_codes",
    "services_per_bene", "payment_per_bene", "payment_per_service", "charge_to_allowed",
    "codes_per_bene", "drug_payment_share", "share_under_65", "share_dual", "risk_score",
    "log_d_claims", "log_d_cost", "d_cost_per_claim", "d_claims_per_bene", "d_brand_share",
    "d_opioid_share", "d_opioid_la_share", "d_antipsychotic_share",
)
MIN_PEERS = 30

# Columns of provider_year used as model inputs next to the peer z-scores. Names, addresses
# and NPIs are never inputs. State is left out so the model cannot learn where past
# enforcement happened to be concentrated.
RAW_FEATURES = (
    "participating", "ruca", "log_b_benes", "log_b_services", "log_b_payment", "log_b_codes",
    "services_per_bene", "payment_per_bene", "payment_per_service", "charge_to_allowed",
    "codes_per_bene", "drug_payment_share", "bene_age", "share_under_65", "share_over_84",
    "share_female", "share_dual", "risk_score", "has_part_d", "log_d_claims", "log_d_cost",
    "d_cost_per_claim", "d_days_per_claim", "d_claims_per_bene", "d_brand_share",
    "d_opioid_share", "d_opioid_la_share", "d_antibiotic_share", "d_antipsychotic_share",
    "d_lis_share",
)
CATEGORICAL_FEATURES = ("provider_type", "entity")
Z_FEATURES = tuple(f"z_{m}" for m in PEER_METRICS)
FEATURES = RAW_FEATURES + Z_FEATURES + CATEGORICAL_FEATURES
