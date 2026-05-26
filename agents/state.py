"""
agents/state.py — Shared LangGraph state for the War Room graph.

Every node reads from and writes to this TypedDict.
Fields are grouped by which agent populates them.
No field is mutated by more than one agent.
"""

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class WarRoomState(TypedDict):
    # ── Input (set by run_agents.py before graph.invoke) ──────────────────────
    customer_id:     str       # e.g. "TEST-4521" or real CRM ID
    customer_raw:    dict      # pre-encoding feature dict — human-readable values
                               # e.g. {"Contract": "Month-to-month", "Tenure Months": 8, ...}
    customer_state:  str       # US state for competitor lookup (DEFAULT if unknown)

    # ── Agent 1 — Diagnostic Lead ─────────────────────────────────────────────
    risk_score:      float     # predict_proba output from xgb_pipeline.pkl (0.0–1.0)
    risk_label:      str       # "HIGH" (≥ threshold) | "LOW" | "PENDING"
    shap_drivers:    list      # top-5 SHAP factors, each a dict:
                               # {feature, shap_value, raw_value, direction}
    risk_summary:    str       # LLM-written 2-3 sentence paragraph

    # ── Agent 2 — Evidence Researcher ─────────────────────────────────────────
    similar_profiles: list     # top-5 from churn_profiles ChromaDB collection
    churn_reasons:    list     # top-7 from churn_reasons ChromaDB collection
    evidence_report:  str      # LLM-synthesized 3-sentence paragraph

    # ── Agent 3 — Mitigation Architect ────────────────────────────────────────
    policy:           dict     # from retention_policy_check MCP tool
    competitor_intel: dict     # from get_competitor_insights MCP tool
    retention_offer:  str      # LLM-drafted personalized offer (3 parts: hook/offer/urgency)
    crm_log_id:       str      # e.g. "RET-20240523-143201-A3F8B2"
    crm_logged:       bool     # True once log_retention_action has been called

    # ── LangGraph internals ───────────────────────────────────────────────────
    messages: Annotated[list, add_messages]