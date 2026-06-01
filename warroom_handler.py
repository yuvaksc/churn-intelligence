"""
warroom_handler.py — Lambda streaming handler for the war room.
Exposed via Lambda Function URL with RESPONSE_STREAM enabled.
Runs LangGraph war room graph and streams SSE events per agent.
"""

import json
import asyncio
import os

# Initialize app state (same lazy init as inference Lambda)
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
    Returns SSE stream with one event per agent completion.
    """
    _ensure_initialized()

    # Extract customer_id from query params or path
    params = event.get("queryStringParameters") or {}
    customer_id = params.get("customer_id")

    if not customer_id:
        # Try path parameters
        path = event.get("rawPath", "")
        parts = path.strip("/").split("/")
        # e.g. /analyze/42 → customer_id = "42"
        if len(parts) >= 2:
            customer_id = parts[-1]

    if not customer_id:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "customer_id required"})
        }

    # Load customer data from app state
    from api.dependencies import get_state, get_customer_raw
    state = get_state()

    try:
        customer_id_int = int(customer_id)
    except ValueError:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "customer_id must be an integer"})
        }

    customer_raw = get_customer_raw(customer_id_int, state)
    if not customer_raw:
        return {
            "statusCode": 404,
            "body": json.dumps({"error": f"Customer {customer_id} not found"})
        }

    # Get customer state for MCP competitor lookup
    customer_state_val = customer_raw.get("State", "DEFAULT")

    # Run the war room graph
    from agents.graph import war_room_graph
    from agents.state import WarRoomState

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

    # Collect SSE chunks
    sse_chunks = []

    async def run_graph():
        async for chunk in war_room_graph.astream(
            initial_state, stream_mode="updates"
        ):
            node_name = list(chunk.keys())[0]
            node_data = chunk[node_name]

            # Serialize non-serializable fields
            safe_data = {}
            for k, v in node_data.items():
                try:
                    json.dumps(v)
                    safe_data[k] = v
                except (TypeError, ValueError):
                    safe_data[k] = str(v)

            sse_event = f"data: {json.dumps({'event': f'{node_name}_complete', 'data': safe_data})}\n\n"
            sse_chunks.append(sse_event)

        sse_chunks.append("data: {\"event\": \"done\"}\n\n")

    asyncio.get_event_loop().run_until_complete(run_graph())

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
        "body": "".join(sse_chunks),
    }