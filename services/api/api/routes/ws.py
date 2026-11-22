"""
api/routes/ws.py — WebSocket token streaming for the war room.

    WS /api/analyze/{id}/ws?customer_state=<state>

Drives the graph with astream_events(version="v2") and pushes JSON frames:
  {"type":"agent_start","agent":"agent2"}
  {"type":"token","agent":"agent2","text":"..."}        # live narrative tokens
  {"type":"agent_complete","agent":"agent2","data":{...}}
  {"type":"done","risk_label":"HIGH","input_scan":{...}}
  {"type":"error","message":"..."}

This is the war room's single entry point: it rate-limits callers (each run spends LLM
tokens), scans the customer input for prompt injection, applies output guardrails to
the drafted offer, and persists the recommendation.

With LANGSMITH_TRACING=true each analysis is ONE trace ("war-room · customer <id>") whose
waterfall shows every agent, its LLM call, the retrieval and each MCP tool call. After
`done`, churn_evals.live records the analysis as experiments on the churn-warroom and
churn-rag-golden datasets, with all eval metrics (in the background; the UI never waits).
"""

import asyncio
import uuid
from datetime import datetime

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from api.deps import get_customer_repo, get_recommendation_repo
from api.ratelimit import analysis_limiter, client_key
from churn_agents.graph import war_room_graph
from churn_agents.guardrails import apply_output_guardrails, scan_input
from churn_data.db import DEFAULT_ASSIGNED_TO, DEFAULT_STATUS
from churn_agents.prompts import PROMPT_VERSION
from churn_evals.live import log_experiments, tracing_on

router = APIRouter(tags=["analysis"])


def _build_initial_state(customer_id: int, customer_raw: dict, customer_state: str,
                         shap_drivers: list | None = None) -> dict:
    """The initial WarRoomState the graph is invoked with."""
    return {
        "customer_id":      f"TEST-{customer_id}",
        "customer_raw":     customer_raw,
        "customer_state":   customer_state,
        "risk_score":       0.0,
        "risk_label":       "PENDING",
        "shap_drivers":     shap_drivers or [],   # precomputed by the batch SHAP job
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
        "status":         DEFAULT_STATUS,
        "assigned_to":    DEFAULT_ASSIGNED_TO,
    }


_AGENTS = {"agent1", "agent2", "agent3"}
_NODE_FIELDS = {
    "agent1": ["risk_score", "risk_label", "shap_drivers", "risk_summary"],
    "agent2": ["evidence_report", "similar_profiles", "churn_reasons", "evidence_citations"],
    "agent3": ["retention_offer", "policy", "competitor_intel", "crm_log_id", "crm_logged", "tools_used"],
}


@router.websocket("/analyze/{customer_id}/ws")
async def analyze_ws(
    websocket:      WebSocket,
    customer_id:    int,
    customer_state: str = Query("DEFAULT"),
):
    await websocket.accept()

    key = client_key(websocket.headers, websocket.client)
    if not analysis_limiter.allow(key):
        await websocket.send_json({"type": "error", "message":
            f"Rate limit reached for this demo; try again in {analysis_limiter.retry_after(key) // 60 + 1} min."})
        await websocket.close(code=4429)
        return

    customers = get_customer_repo()
    recs      = get_recommendation_repo()
    try:
        customer_raw = await asyncio.to_thread(customers.features, customer_id)
        if customer_raw is None:
            await websocket.send_json({"type": "error", "message": f"Customer {customer_id} not in test set"})
            await websocket.close()
            return

        input_scan = scan_input(customer_state, customer_raw)

        customer_row = await asyncio.to_thread(customers.get, customer_id)
        initial      = _build_initial_state(customer_id, customer_raw, customer_state,
                                            (customer_row or {}).get("top_shap_drivers"))
        # One named trace per analysis; its id is recorded on the analysis's experiments.
        run_id       = uuid.uuid4()
        trace_config = {
            "run_id":   run_id,
            "run_name": f"war-room · customer {customer_id}",
            "tags":     ["war-room", f"prompt-{PROMPT_VERSION}"],
            "metadata": {"customer_id": customer_id, "customer_state": customer_state,
                         "model_risk_label": (customer_row or {}).get("risk_label"),
                         "prompt_version": PROMPT_VERSION},
        }

        final_label   = "PENDING"
        risk_score    = 0.0
        agent3_out:   dict = {}
        agent3_guard: dict = {}
        started:      set  = set()
        analysis:     dict = {}          # every agent's full output, recorded as the experiment
        agent_path:   list = []

        async for ev in war_room_graph.astream_events(
            initial, version="v2", config=trace_config,
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
                    analysis.update(out)
                    analysis["guardrails"] = agent3_guard or analysis.get("guardrails")
                    agent_path.append(name)

        if agent3_out.get("crm_log_id"):
            rec = _build_rec(agent3_out["crm_log_id"], f"TEST-{customer_id}",
                             risk_score, agent3_out.get("retention_offer", ""), customer_raw)
            await asyncio.to_thread(recs.insert, rec)

        await websocket.send_json({"type": "done", "risk_label": final_label, "input_scan": input_scan})

        await websocket.close()

        # Evals: this analysis becomes an experiment on each dataset (judges run here, after the
        # UI already has its answer).
        if tracing_on():
            analysis.update(customer_raw=customer_raw, customer_state=customer_state, agent_path=agent_path)
            await asyncio.to_thread(log_experiments, run_id, customer_id, analysis)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close()
        except Exception:
            pass
