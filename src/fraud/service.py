"""Scoring service: one provider's yearly billing in, a review decision out.

Run with: uvicorn fraud.service:app --app-dir src
"""

import json
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import config, db
from .scoring import Scorer


class PartB(BaseModel):
    provider_type: str = Field(description="CMS provider type, e.g. 'Internal Medicine'")
    entity: str = Field("I", description="'I' for an individual, 'O' for an organisation")
    state: Optional[str] = None
    ruca: Optional[float] = None
    participating: str = "Y"
    b_codes: float
    b_benes: float
    b_services: float
    b_charges: float
    b_allowed: float
    b_payment: float
    b_drug_payment: Optional[float] = 0.0
    bene_age: Optional[float] = None
    bene_lt65: Optional[float] = None
    bene_gt84: Optional[float] = None
    bene_female: Optional[float] = None
    bene_dual: Optional[float] = None
    risk_score: Optional[float] = None


class PartD(BaseModel):
    d_claims: float
    d_cost: float
    d_days: Optional[float] = None
    d_benes: Optional[float] = None
    d_brand_claims: Optional[float] = None
    d_opioid_claims: Optional[float] = None
    d_opioid_la_claims: Optional[float] = None
    d_antibiotic_claims: Optional[float] = None
    d_antipsychotic_claims: Optional[float] = None
    d_lis_claims: Optional[float] = None


class Provider(BaseModel):
    part_b: PartB
    part_d: Optional[PartD] = None


scorer = Scorer()
peer_stats = pd.read_parquet(config.MODELS / "peer_stats.parquet")
review = json.loads((config.MODELS / "threshold.json").read_text())
YEAR = review["year"]

app = FastAPI(title="Medicare provider review service")


def to_tables(p: Provider):
    b = pd.DataFrame([{"year": YEAR, "npi": 0, **p.part_b.model_dump()}])
    d_cols = list(PartD.model_fields)
    d = (pd.DataFrame([{"year": YEAR, "npi": 0, **p.part_d.model_dump()}]) if p.part_d
         else pd.DataFrame({"year": pd.Series(dtype="int32"), "npi": pd.Series(dtype="int64"),
                            **{c: pd.Series(dtype="float64") for c in d_cols}}))
    for c in d_cols:
        if c not in d:
            d[c] = np.nan
    return b, d


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "detector": scorer.chosen, "reference_year": YEAR}


@app.post("/score")
def score(p: Provider) -> dict:
    b, d = to_tables(p)
    feats = db.features_from_raw(b, d, peer_stats)
    s = float(scorer.score(feats)[0])
    ref = scorer.reference["chosen"]
    return {
        "score": s,
        "percentile": float(np.searchsorted(ref, s, side="right") / len(ref)),
        "on_review_list": s >= review["threshold"],
        "review_list": f"top {review['budget']:,} of {review['providers']:,} providers in {YEAR}",
        "reasons": scorer.reasons(feats)[0],
        "similar_known_cases": scorer.index.neighbours(feats),
    }
