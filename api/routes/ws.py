"""
api/routes/ws.py — WebSocket token streaming for the war room (Step 6).

    WS /api/analyze/{id}/ws?customer_state=<state>

Drives the graph with astream_events(version="v2") and pushes JSON frames:
  {"type":"agent_start","agent":"agent2"}
  {"type":"token","agent":"agent2","text":"..."}        # live narrative tokens
  {"type":"agent_complete","agent":"agent2","data":{...}}
  {"type":"done","risk_label":"HIGH","trace_id":"..."}
  {"type":"error","message":"..."}

Only agent1/agent2/agent3 narrative LLMs stream tokens (the supervisor's
structured-output routing and agent3's tool-deciding calls carry no useful text).
Guardrails + recommendation + audit are applied exactly as on the POST/SSE paths.
"""

import time
import uuid
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from db import customers as db_customers
from db import recommendations as db_recs
from db import audit as db_audit
from guardrails import scan_input, apply_output_guardrails
from agents.graph import war_room_graph
from api.routes.analyze import _build_initial_state, _build_rec, _build_audit

router = APIRouter(tags=["analysis"])

_AGENTS = {"agent1", "agent2", "agent3"}
_NODE_FIELDS = {
    "agent1": ["risk_score", "risk_label", "shap_drivers", "risk_summary"],
    "agent2": ["evidence_report", "similar_profiles", "churn_reasons"],
    "agent3": ["retention_offer", "policy", "competitor_intel", "crm_log_id", "crm_logged", "tools_used"],
}


@router.websocket("/analyze/{customer_id}/ws")
async def analyze_ws(
    websocket:      WebSocket,
    customer_id:    int,
    customer_state: str = Query("DEFAULT"),
):
    await websocket.accept()
    try:
        customer_raw = await asyncio.to_thread(db_customers.get_customer_features, customer_id)
        if customer_raw is None:
            await websocket.send_json({"type": "error", "message": f"Customer {customer_id} not in test set"})
            await websocket.close()
            return

        trace_id   = uuid.uuid4().hex
        input_scan = scan_input(customer_state, customer_raw)
        initial    = _build_initial_state(customer_id, customer_raw, customer_state)
        t0 = time.perf_counter()

        final_label   = "PENDING"
        risk_score    = 0.0
        agent_path:   list = []
        agent2_out:   dict = {}
        agent3_out:   dict = {}
        agent3_guard: dict = {}
        started:      set  = set()

        async for ev in war_room_graph.astream_events(initial, version="v2"):
            etype = ev["event"]
            name  = ev.get("name")
            node  = (ev.get("metadata") or {}).get("langgraph_node")

            # ── live tokens from the agent narrative LLMs ─────────────────────
            if etype == "on_chat_model_stream" and node in _AGENTS:
                chunk = ev["data"].get("chunk")
                text  = getattr(chunk, "content", "") if chunk is not None else ""
                if text:
                    if node not in started:
                        started.add(node)
                        await websocket.send_json({"type": "agent_start", "agent": node})
                    await websocket.send_json({"type": "token", "agent": node, "text": text})

            # ── node completion → structured agent_complete ──────────────────
            elif etype == "on_chain_end" and name in _AGENTS:
                out = ev["data"].get("output")
                if isinstance(out, dict):
                    if name == "agent1":
                        final_label = out.get("risk_label", final_label)
                        risk_score  = out.get("risk_score", risk_score)
                    elif name == "agent2":
                        agent2_out = out
                    elif name == "agent3":
                        safe_offer, agent3_guard = apply_output_guardrails(
                            out.get("retention_offer", ""), out.get("policy", {}))
                        out = {**out, "retention_offer": safe_offer}
                        agent3_out = out

                    payload = {k: out[k] for k in _NODE_FIELDS[name] if k in out}
                    if name == "agent3":
                        payload["guardrails"] = agent3_guard
                    if name not in started:
                        started.add(name)
                        await websocket.send_json({"type": "agent_start", "agent": name})
                    await websocket.send_json({"type": "agent_complete", "agent": name, "data": payload})

            # ── capture routing path from the supervisor ─────────────────────
            elif etype == "on_chain_end" and name == "supervisor":
                out = ev["data"].get("output")
                if isinstance(out, dict) and out.get("agent_path"):
                    agent_path = out["agent_path"]

        latency_ms = int((time.perf_counter() - t0) * 1000)

        if agent3_out.get("crm_log_id"):
            rec = _build_rec(agent3_out["crm_log_id"], f"TEST-{customer_id}",
                             risk_score, agent3_out.get("retention_offer", ""), customer_raw)
            await asyncio.to_thread(db_recs.insert_recommendation, rec)

        audit_state = {
            "customer_id":      f"TEST-{customer_id}",
            "risk_label":       final_label,
            "risk_score":       risk_score,
            "agent_path":       agent_path,
            "tools_used":       agent3_out.get("tools_used", []),
            "similar_profiles": agent2_out.get("similar_profiles", []),
            "churn_reasons":    agent2_out.get("churn_reasons", []),
            "policy":           agent3_out.get("policy", {}),
            "retention_offer":  agent3_out.get("retention_offer", ""),
            "crm_log_id":       agent3_out.get("crm_log_id", ""),
        }
        await asyncio.to_thread(
            db_audit.insert_audit,
            _build_audit(trace_id, f"TEST-{customer_id}", customer_state, audit_state,
                         agent3_guard, input_scan, latency_ms),
        )

        await websocket.send_json({"type": "done", "risk_label": final_label, "trace_id": trace_id})
        await websocket.close()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close()
        except Exception:
            pass
