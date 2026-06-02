"""
api/routes/analyze.py

POST /api/analyze/{customer_id}         — run full war room, return JSON result
GET  /api/analyze/{customer_id}/stream  — SSE stream, one event per agent node

SSE event format:
  event: agent1_complete
  data: {"risk_score": 0.99, "risk_label": "HIGH", ...}

  event: agent2_complete
  data: {"evidence_report": "...", ...}

  event: agent3_complete
  data: {"retention_offer": "...", ...}

  event: done
  data: {"status": "complete", "risk_label": "HIGH"}

  event: error
  data: {"message": "..."}

LangGraph astream() emits one chunk per node that actually ran.
If risk is LOW, only agent1_complete + done are emitted.
"""

import json
import asyncio
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.schemas import WarRoomResult, ShapDriver
from api.dependencies import AppState, get_state, get_customer_raw
from agents.graph import war_room_graph

_RETENTION_LOG = Path("models/reports/retention_log.jsonl")


def _clear_retention_log() -> None:
    """Wipe previous retention log so each analysis starts fresh."""
    if _RETENTION_LOG.exists():
        _RETENTION_LOG.write_text("", encoding="utf-8")

router = APIRouter(tags=["analysis"])


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
        "messages":         [],
    }


# ── POST: full synchronous result ─────────────────────────────────────────────

@router.post("/analyze/{customer_id}", response_model=WarRoomResult)
async def analyze_customer(
    customer_id:    int,
    customer_state: str = Query("DEFAULT", description="US state for competitor intel"),
    state: AppState = Depends(get_state),
):
    """
    Runs the full 3-agent war room and returns the complete result.
    Takes 20–40 seconds depending on LLM response time.
    For real-time progress, use the /stream endpoint instead.
    """
    customer_raw = get_customer_raw(customer_id, state)
    if customer_raw is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not in test set")

    _clear_retention_log()

    initial = _build_initial_state(customer_id, customer_raw, customer_state)

    try:
        result = await war_room_graph.ainvoke(initial)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"War room failed: {str(e)}")

    return WarRoomResult(
        customer_id=result["customer_id"],
        risk_score=result["risk_score"],
        risk_label=result["risk_label"],
        risk_summary=result["risk_summary"],
        shap_drivers=[ShapDriver(**d) for d in result["shap_drivers"]],
        evidence_report=result.get("evidence_report", ""),
        similar_profiles=result.get("similar_profiles", []),
        churn_reasons=result.get("churn_reasons", []),
        retention_offer=result.get("retention_offer", ""),
        policy=result.get("policy", {}),
        competitor_intel=result.get("competitor_intel", {}),
        crm_log_id=result.get("crm_log_id", ""),
        crm_logged=result.get("crm_logged", False),
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
    "agent3": ["retention_offer", "policy", "competitor_intel", "crm_log_id", "crm_logged"],
}


def _sse(event: str, data: dict) -> str:
    """Format one SSE message."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _stream_war_room(
    customer_id:   int,
    customer_raw:  dict,
    customer_state: str,
) -> AsyncGenerator[str, None]:
    """
    Yields SSE strings as each LangGraph node completes.
    Keeps the connection alive with a comment ping every 15 seconds.
    """
    initial    = _build_initial_state(customer_id, customer_raw, customer_state)
    final_label = "PENDING"

    try:
        # Ping immediately so the client knows the connection is live
        yield ": ping\n\n"

        async for chunk in war_room_graph.astream(initial):
            node_name = next(iter(chunk))         # e.g. "agent1"
            node_data = chunk[node_name]          # state update from that node

            if node_name not in _NODE_TO_EVENT:
                continue                           # skip internal nodes

            # Extract only the fields relevant to this node
            payload = {
                k: node_data[k]
                for k in _NODE_FIELDS[node_name]
                if k in node_data
            }

            if node_name == "agent1":
                final_label = node_data.get("risk_label", "PENDING")

            yield _sse(_NODE_TO_EVENT[node_name], payload)
            await asyncio.sleep(0)                 # yield control back to event loop

        yield _sse("done", {"status": "complete", "risk_label": final_label})

    except Exception as e:
        yield _sse("error", {"message": str(e)})


@router.get("/analyze/{customer_id}/stream")
async def stream_analysis(
    customer_id:    int,
    customer_state: str = Query("DEFAULT"),
    state: AppState = Depends(get_state),
):
    """
    SSE endpoint — streams war room progress as each agent completes.
    Connect with EventSource in Next.js for real-time UI updates.
    """
    customer_raw = get_customer_raw(customer_id, state)
    if customer_raw is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not in test set")

    _clear_retention_log()

    return StreamingResponse(
        _stream_war_room(customer_id, customer_raw, customer_state),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",   # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )