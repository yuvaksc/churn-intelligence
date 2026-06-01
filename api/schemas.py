"""
api/schemas.py — Pydantic models for all request/response shapes.

These become the OpenAPI schema automatically (visible at /docs).
Everything typed explicitly so Next.js can generate matching TypeScript types.
"""

from typing import Optional
from pydantic import BaseModel


# ── Shared primitives ─────────────────────────────────────────────────────────

class ShapDriver(BaseModel):
    feature:    str
    shap_value: float
    raw_value:  float
    direction:  str   # "increases churn risk" | "decreases churn risk"


# ── Customer endpoints ────────────────────────────────────────────────────────

class CustomerSummary(BaseModel):
    """Lightweight row for the customer list table."""
    customer_id:      int
    risk_score:       float
    risk_label:       str          # "HIGH" | "LOW"
    contract:         str
    monthly_charges:  float
    tenure_months:    int
    internet_service: str
    services_count:   int
    high_risk_flag:   int
    true_label:       Optional[int] = None   # ground truth from test set


class CustomerDetail(CustomerSummary):
    """Full feature set + top SHAP drivers for the customer detail page."""
    all_features:     dict
    top_shap_drivers: list[ShapDriver]


# ── War room / analysis endpoints ─────────────────────────────────────────────

class WarRoomRequest(BaseModel):
    customer_state: str = "DEFAULT"   # US state for competitor intel


class WarRoomResult(BaseModel):
    """Full result returned by POST /analyze/{customer_id}."""
    customer_id:      str
    risk_score:       float
    risk_label:       str
    risk_summary:     str
    shap_drivers:     list[ShapDriver]
    evidence_report:  str
    similar_profiles: list[dict]
    churn_reasons:    list[dict]
    retention_offer:  str
    policy:           dict
    competitor_intel: dict
    crm_log_id:       str
    crm_logged:       bool


# SSE event payloads — one per agent node
class Agent1Event(BaseModel):
    risk_score:   float
    risk_label:   str
    shap_drivers: list[ShapDriver]
    risk_summary: str


class Agent2Event(BaseModel):
    evidence_report:  str
    similar_profiles: list[dict]
    churn_reasons:    list[dict]


class Agent3Event(BaseModel):
    retention_offer: str
    policy:          dict
    competitor_intel: dict
    crm_log_id:      str
    crm_logged:      bool


# ── Logs endpoint ─────────────────────────────────────────────────────────────

class RetentionLogEntry(BaseModel):
    log_id:         str
    customer_id:    str
    risk_score:     float
    offer_text:     str
    contract_type:  str
    monthly_charge: float
    timestamp:      str
    status:         str
    assigned_to:    str


# ── Metrics endpoint ──────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    threshold:                        float
    f1:                               float
    precision:                        float
    recall:                           float
    roc_auc:                          float
    avg_precision:                    float
    true_positives:                   int
    false_positives:                  int
    false_negatives:                  int
    true_negatives:                   int
    interventions_fired:              int
    actual_churners:                  int
    revenue_at_risk_caught_monthly:   float
    revenue_missed_monthly:           float
    false_positive_spend_monthly:     float


# ── Health endpoint ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:              str
    model:               str
    threshold:           float
    test_set_size:       int
    high_risk_count:     int
    chroma_collections:  list[str]