"""
warroom_handler.py — Lambda handler for the war room.
Exposed via Lambda Function URL.
Runs LangGraph war room graph and returns buffered SSE events.

Cold start behaviour:
  Lambda's init phase is capped at 10s. Importing the full ML stack
  (torch, xgboost, sentence-transformers) plus the SageMaker registry
  query and 5 S3 downloads can exceed that window on a brand-new container.

  Fix: _ensure_initialized() is called inside handler() — correctly deferred
  to the request phase. BUT the import chain triggered at module level by
  `from agents.graph import war_room_graph` is heavy enough to push the init
  phase close to its limit before handler() is even called.

  Solution: all heavy imports (agents.graph, api.dependencies) are moved
  inside handler() so they don't execute during the init phase. Only stdlib
  and lightweight modules import at module level.
"""

import json
import asyncio
import os

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
    """
    Lambda Function URL handler.
    Returns buffered SSE response with one event per agent completion.
    """
    # All heavy imports deferred here — keeps module-level init phase light
    # so Lambda's 10s init window is not consumed before handler() runs.
    from api.dependencies import get_state, get_customer_raw
    from agents.graph import war_room_graph
    from agents.state import WarRoomState

    _ensure_initialized()

    # ------------------------------------------------------------------ #
    # Parse customer_id from query params or path                          #
    # ------------------------------------------------------------------ #
    params = event.get("queryStringParameters") or {}
    customer_id = params.get("customer_id")

    if not customer_id:
        path  = event.get("rawPath", "")
        parts = path.strip("/").split("/")
        if len(parts) >= 2:
            customer_id = parts[-1]

    if not customer_id:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "customer_id required"}),
        }

    try:
        customer_id_int = int(customer_id)
    except ValueError:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "customer_id must be an integer"}),
        }

    # ------------------------------------------------------------------ #
    # Load customer from state                                             #
    # ------------------------------------------------------------------ #
    state        = get_state()
    customer_raw = get_customer_raw(customer_id_int, state)

    if not customer_raw:
        return {
            "statusCode": 404,
            "body": json.dumps({"error": f"Customer {customer_id} not found"}),
        }

    customer_state_val = customer_raw.get("State", "DEFAULT")

    # ------------------------------------------------------------------ #
    # Run the war room graph                                               #
    # ------------------------------------------------------------------ #
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
        async for chunk in war_room_graph.astream(
            initial_state, stream_mode="updates"
        ):
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
            "Content-Type":    "text/event-stream",
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",
        },
        "body": "".join(sse_chunks),
    }