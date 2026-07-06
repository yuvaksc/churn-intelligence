"""
api/routes/ws.py — WebSocket token streaming for the war room (Step 6).

    WS /api/analyze/{id}/ws?customer_state=<state>

Drives the graph with astream_events(version="v2") and pushes JSON frames:
  {"type":"agent_start","agent":"agent2"}
  {"type":"token","agent":"agent2","text":"..."}        # live narrative tokens
  {"type":"agent_complete","agent":"agent2","data":{...}}
  {"type":"done","risk_label":"HIGH","input_scan":{...}}
  {"type":"error","message":"..."}

Only agent1/agent2/agent3 narrative LLMs stream tokens (the supervisor's
structured-output routing and agent3's tool-deciding calls carry no useful text).
This is the war room's single entry point: it scans the customer input for
prompt-injection, applies output guardrails to the drafted offer, and persists the
recommendation. LangSmith tracing is automatic when LANGSMITH_TRACING=true.
"""

import uuid
import asyncio
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from db import customers as db_customers
from db import recommendations as db_recs
from guardrails import scan_input, apply_output_guardrails
from agents.graph import war_room_graph
from eval import experiment_callbacks, score_run

router = APIRouter(tags=["analysis"])


def _build_initial_state(customer_id: int, customer_raw: dict, customer_state: str) -> dict:
    """The initial WarRoomState the graph is invoked with."""
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

        initial    = _build_initial_state(customer_id, customer_raw, customer_state)
        input_scan = scan_input(customer_state, customer_raw)

        # Trace this run into today's experiment on the churn-eval dataset, linked to
        # the same-label golden archetype (uses the customer's pre-scored risk label).
        # [] when tracing is off / dataset unreachable → the run just traces normally.
        customer_row = await asyncio.to_thread(db_customers.get_customer, customer_id)
        run_id       = uuid.uuid4()
        eval_cbs     = await asyncio.to_thread(experiment_callbacks, (customer_row or {}).get("risk_label", ""), customer_id)

        final_label   = "PENDING"
        risk_score    = 0.0
        agent3_out:   dict = {}
        agent3_guard: dict = {}
        started:      set  = set()
        ws_out:       dict = {}          # WS-shaped run output, fed to the live eval

        async for ev in war_room_graph.astream_events(
            initial, version="v2", config={"run_id": run_id, "callbacks": eval_cbs},
        ):
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
                    ws_out[name] = payload

        if agent3_out.get("crm_log_id"):
            rec = _build_rec(agent3_out["crm_log_id"], f"TEST-{customer_id}",
                             risk_score, agent3_out.get("retention_offer", ""), customer_raw)
            await asyncio.to_thread(db_recs.insert_recommendation, rec)

        await websocket.send_json({"type": "done", "risk_label": final_label, "input_scan": input_scan})

        # live eval — the war room already ran *inside* the experiment (eval_cbs), so
        # this run IS an experiment run on churn-eval with its full waterfall; now attach
        # the four metric feedbacks (completeness / quality + reasoning / guardrails).
        # Best-effort; after `done` so it never delays the UI.
        if eval_cbs:
            ws_out["done"] = {"risk_label": final_label}
            await asyncio.to_thread(score_run, run_id, initial, ws_out)

        await websocket.close()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close()
        except Exception:
            pass
