"""
warroom_handler.py — Lambda handler for the war room.
Exposed via Lambda Function URL (BUFFERED invoke mode).
Runs LangGraph war room graph and returns buffered SSE events.

Heavy imports are deferred inside handler() so Lambda's 10s init phase
is not consumed before the function runs. Initialization and graph execution
are wrapped in explicit try/except so any error surfaces in CloudWatch and
in the response body instead of being swallowed (which produces an opaque 500).
"""

import json
import asyncio
import os
import sys
import traceback

_initialized = False


def _ensure_initialized():
    global _initialized
    if not _initialized:
        from api.dependencies import initialize_state
        print("War room Lambda: initializing state...")
        initialize_state()
        _initialized = True
        print("State initialized.")


def handler(event, context):
    try:
        from api.dependencies import get_state, get_customer_raw
        from agents.graph import war_room_graph
        from agents.state import WarRoomState

        _ensure_initialized()

        params = event.get("queryStringParameters") or {}
        customer_id = params.get("customer_id")
        if not customer_id:
            path  = event.get("rawPath", "")
            parts = path.strip("/").split("/")
            if len(parts) >= 2:
                customer_id = parts[-1]

        if not customer_id:
            return {"statusCode": 400, "body": json.dumps({"error": "customer_id required"})}

        try:
            customer_id_int = int(customer_id)
        except ValueError:
            return {"statusCode": 400, "body": json.dumps({"error": "customer_id must be an integer"})}

        state        = get_state()
        customer_raw = get_customer_raw(customer_id_int, state)
        if not customer_raw:
            return {"statusCode": 404, "body": json.dumps({"error": f"Customer {customer_id} not found"})}

        customer_state_val = customer_raw.get("State", "DEFAULT")

        initial_state = WarRoomState(
            customer_id=str(customer_id),
            customer_raw=customer_raw,
            customer_state=customer_state_val,
            risk_score=0.0,
            risk_label="PENDING",
            shap_drivers=[],
            risk_summary="",
            similar_profiles=[],
            churn_reasons=[],
            evidence_report="",
            policy={},
            competitor_intel={},
            retention_offer="",
            crm_log_id="",
            crm_logged=False,
            messages=[],
        )

        sse_chunks = []

        async def run_graph():
            async for chunk in war_room_graph.astream(initial_state, stream_mode="updates"):
                node_name = list(chunk.keys())[0]
                node_data = chunk[node_name]
                safe_data = {}
                for k, v in node_data.items():
                    try:
                        json.dumps(v)
                        safe_data[k] = v
                    except (TypeError, ValueError):
                        safe_data[k] = str(v)
                sse_chunks.append(
                    f"data: {json.dumps({'event': f'{node_name}_complete', 'data': safe_data})}\n\n"
                )
            sse_chunks.append('data: {"event": "done"}\n\n')

        asyncio.get_event_loop().run_until_complete(run_graph())

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type":      "text/event-stream",
                "Cache-Control":     "no-cache",
                "X-Accel-Buffering": "no",
            },
            "body": "".join(sse_chunks),
        }

    except Exception as e:
        tb = traceback.format_exc()
        print("WAR ROOM HANDLER ERROR:", file=sys.stderr)
        print(tb, file=sys.stderr)
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e), "traceback": tb}),
        }