"""
api/routes/analyze.py

POST /api/analyze/{customer_id}         — run full war room, return JSON result
GET  /api/analyze/{customer_id}/stream  — SSE stream, one event per agent node

Each run is wrapped with backend guardrails (input scan + PII masking + policy /
grounding checks on the offer) and recorded in the append-only audit_log
(reconstructable via GET /api/audit/{trace_id}). LangSmith tracing is automatic
when LANGSMITH_TRACING=true.
"""

import json
import time
import uuid
import asyncio
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.schemas import WarRoomResult, ShapDriver
from db import customers as db_customers
from db import recommendations as db_recs
from db import audit as db_audit
from guardrails import scan_input, apply_output_guardrails
from agents.graph import war_room_graph

router = APIRouter(tags=["analysis"])


def _build_rec(log_id: str, customer_id_str: str, risk_score: float,
               offer_text: str, customer_raw: dict) -> dict:
    """Assemble a recommendations row from the war-room result (RetentionLogEntry shape)."""
    return {
        "log_id":         log_id,
        "customer_id":    customer_id_str,
        "risk_score":     round(float(risk_score), 4),
        "offer_text":     offer_text,
        "contract_type":  str(customer_raw.get("Contract", "")),
        "monthly_charge": round(float(customer_raw.get("Monthly Charges", 0.0)), 2),
        "timestamp":      datetime.now().isoformat(),
        "status":         db_recs.DEFAULT_STATUS,
        "assigned_to":    db_recs.DEFAULT_ASSIGNED_TO,
    }


def _build_initial_state(customer_id: int, customer_raw: dict, customer_state: str) -> dict:
    return {
        "customer_id":      f"TEST-{customer_id}",
        "customer_raw":     customer_raw,
        "customer_state":   customer_state,
        "risk_score":       0.0,
        "risk_label":       "PENDING",
        "shap_drivers":     [],
        "risk_summary":     "",
        "similar_profiles": [],
        "churn_reasons":    [],
        "evidence_report":  "",
        "policy":           {},
        "competitor_intel": {},
        "retention_offer":  "",
        "crm_log_id":       "",
        "crm_logged":       False,
        "tools_used":       [],
        "messages":         [],
    }


def _profile_digest(similar_profiles: list) -> list:
    """Compact, audit-friendly view of the retrieved profiles."""
    return [
        {
            "similarity":   p.get("similarity"),
            "rerank_score": p.get("rerank_score"),
            "doc":          (p.get("document") or "")[:120],
        }
        for p in (similar_profiles or [])
    ]


def _build_audit(trace_id: str, fallback_cid: str, customer_state: str, state: dict,
                 guard: dict, input_scan: dict, latency_ms: int) -> dict:
    return {
        "trace_id":            trace_id,
        "customer_id":         state.get("customer_id", fallback_cid),
        "customer_state":      customer_state,
        "risk_label":          state.get("risk_label"),
        "risk_score":          round(float(state.get("risk_score", 0.0) or 0.0), 4),
        "agent_path":          state.get("agent_path", []),
        "tools_used":          state.get("tools_used", []),
        "retrieved_profiles":  _profile_digest(state.get("similar_profiles", [])),
        "churn_reasons_count": len(state.get("churn_reasons", []) or []),
        "policy":              state.get("policy", {}),
        "retention_offer":     state.get("retention_offer", ""),
        "crm_log_id":          state.get("crm_log_id", ""),
        "guardrails":          guard,
        "input_scan":          input_scan,
        "latency_ms":          latency_ms,
        "created_at":          datetime.now().isoformat(),
    }


# ── POST: full synchronous result ─────────────────────────────────────────────

@router.post("/analyze/{customer_id}", response_model=WarRoomResult)
async def analyze_customer(
    customer_id:    int,
    customer_state: str = Query("DEFAULT", description="US state for competitor intel"),
):
    """
    Runs the full 3-agent war room and returns the complete result.
    Takes 20–40 seconds depending on LLM response time.
    For real-time progress, use the /stream endpoint instead.
    """
    customer_raw = await asyncio.to_thread(db_customers.get_customer_features, customer_id)
    if customer_raw is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not in test set")

    trace_id   = uuid.uuid4().hex
    input_scan = scan_input(customer_state, customer_raw)
    t0 = time.perf_counter()

    initial = _build_initial_state(customer_id, customer_raw, customer_state)
    try:
        result = await war_room_graph.ainvoke(initial)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"War room failed: {str(e)}")
    latency_ms = int((time.perf_counter() - t0) * 1000)

    # Output guardrails: PII-mask the offer + verify policy ceiling / grounding
    safe_offer, guard = apply_output_guardrails(result.get("retention_offer", ""), result.get("policy", {}))
    result["retention_offer"] = safe_offer

    if result.get("crm_log_id"):
        rec = _build_rec(result["crm_log_id"], result.get("customer_id", ""),
                         result.get("risk_score", 0.0), safe_offer, customer_raw)
        await asyncio.to_thread(db_recs.insert_recommendation, rec)

    await asyncio.to_thread(
        db_audit.insert_audit,
        _build_audit(trace_id, f"TEST-{customer_id}", customer_state, result, guard, input_scan, latency_ms),
    )

    return WarRoomResult(
        customer_id=result["customer_id"],
        risk_score=result["risk_score"],
        risk_label=result["risk_label"],
        risk_summary=result["risk_summary"],
        shap_drivers=[ShapDriver(**d) for d in result["shap_drivers"]],
        evidence_report=result.get("evidence_report", ""),
        similar_profiles=result.get("similar_profiles", []),
        churn_reasons=result.get("churn_reasons", []),
        retention_offer=safe_offer,
        policy=result.get("policy", {}),
        competitor_intel=result.get("competitor_intel", {}),
        crm_log_id=result.get("crm_log_id", ""),
        crm_logged=result.get("crm_logged", False),
        tools_used=result.get("tools_used", []),
        trace_id=trace_id,
        guardrails=guard,
    )


# ── GET /stream: SSE real-time events ────────────────────────────────────────

_NODE_TO_EVENT = {
    "agent1": "agent1_complete",
    "agent2": "agent2_complete",
    "agent3": "agent3_complete",
}

_NODE_FIELDS = {
    "agent1": ["risk_score", "risk_label", "shap_drivers", "risk_summary"],
    "agent2": ["evidence_report", "similar_profiles", "churn_reasons"],
    "agent3": ["retention_offer", "policy", "competitor_intel", "crm_log_id", "crm_logged", "tools_used"],
}


def _sse(event: str, data: dict) -> str:
    """Format one SSE message."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _stream_war_room(
    customer_id:    int,
    customer_raw:   dict,
    customer_state: str,
    trace_id:       str,
) -> AsyncGenerator[str, None]:
    """Yields SSE strings as each LangGraph node completes; masks the offer,
    persists the recommendation, and writes the audit row at the end."""
    initial      = _build_initial_state(customer_id, customer_raw, customer_state)
    final_label  = "PENDING"
    risk_score   = 0.0
    agent2_data  = None
    agent3_data  = None
    agent3_guard: dict = {}
    agent_path: list = []
    t0 = time.perf_counter()

    try:
        yield ": ping\n\n"   # tell the client the connection is live

        async for chunk in war_room_graph.astream(initial):
            node_name = next(iter(chunk))
            node_data = chunk[node_name]

            if isinstance(node_data, dict) and node_data.get("agent_path"):
                agent_path = node_data["agent_path"]    # captured from supervisor chunks

            if node_name not in _NODE_TO_EVENT:
                continue                                 # skip the supervisor node

            payload = {k: node_data[k] for k in _NODE_FIELDS[node_name] if k in node_data}

            if node_name == "agent1":
                final_label = node_data.get("risk_label", "PENDING")
                risk_score  = node_data.get("risk_score", 0.0)
            elif node_name == "agent2":
                agent2_data = node_data
            elif node_name == "agent3":
                safe_offer, agent3_guard = apply_output_guardrails(
                    node_data.get("retention_offer", ""), node_data.get("policy", {}))
                node_data["retention_offer"] = safe_offer   # mask in-place for persist/audit
                agent3_data = node_data
                payload["retention_offer"] = safe_offer
                payload["guardrails"]      = agent3_guard

            yield _sse(_NODE_TO_EVENT[node_name], payload)
            await asyncio.sleep(0)

        latency_ms = int((time.perf_counter() - t0) * 1000)

        if agent3_data and agent3_data.get("crm_log_id"):
            rec = _build_rec(agent3_data["crm_log_id"], f"TEST-{customer_id}",
                             risk_score, agent3_data.get("retention_offer", ""), customer_raw)
            await asyncio.to_thread(db_recs.insert_recommendation, rec)

        audit_state = {
            "customer_id":      f"TEST-{customer_id}",
            "risk_label":       final_label,
            "risk_score":       risk_score,
            "agent_path":       agent_path,
            "tools_used":       (agent3_data or {}).get("tools_used", []),
            "similar_profiles": (agent2_data or {}).get("similar_profiles", []),
            "churn_reasons":    (agent2_data or {}).get("churn_reasons", []),
            "policy":           (agent3_data or {}).get("policy", {}),
            "retention_offer":  (agent3_data or {}).get("retention_offer", ""),
            "crm_log_id":       (agent3_data or {}).get("crm_log_id", ""),
        }
        await asyncio.to_thread(
            db_audit.insert_audit,
            _build_audit(trace_id, f"TEST-{customer_id}", customer_state, audit_state,
                         agent3_guard, scan_input(customer_state, customer_raw), latency_ms),
        )

        yield _sse("done", {"status": "complete", "risk_label": final_label, "trace_id": trace_id})

    except Exception as e:
        yield _sse("error", {"message": str(e)})


@router.get("/analyze/{customer_id}/stream")
async def stream_analysis(
    customer_id:    int,
    customer_state: str = Query("DEFAULT"),
):
    """
    SSE endpoint — streams war room progress as each agent completes.
    Connect with EventSource in Next.js for real-time UI updates.
    """
    customer_raw = await asyncio.to_thread(db_customers.get_customer_features, customer_id)
    if customer_raw is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not in test set")

    trace_id = uuid.uuid4().hex
    return StreamingResponse(
        _stream_war_room(customer_id, customer_raw, customer_state, trace_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",   # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )
