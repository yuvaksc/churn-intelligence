"""
mcp_server/server.py — FastMCP server with dual transport support.

LOCAL (stdio):
    python -m mcp_server.server
    MCP_TRANSPORT not set, defaults to stdio

DOCKER / AWS Lambda 2 (HTTP/SSE):
    MCP_TRANSPORT=sse
    MCP_PORT=8001    (default)
    MCP_HOST=0.0.0.0 (default)

Same binary, same tools, same logic — only the transport changes.
The env var is the only switch between local and deployed behaviour.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from fastmcp import FastMCP
from mcp_server.policies import get_policy
from mcp_server.competitors import get_competitor_info, FEATURE_DICTIONARY

_PROJECT_ROOT = Path(__file__).parent.parent
import os

# Use /tmp in Lambda (read-only filesystem), local path otherwise
if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    LOG_PATH = Path("/tmp/retention_log.jsonl")
else:
    LOG_PATH = _PROJECT_ROOT / "models" / "reports" / "retention_log.jsonl"
    
mcp = FastMCP("Churn Retention MCP Server")


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def retention_policy_check(contract_type: str, monthly_charge: float) -> dict:
    """
    Returns the retention discount policy for a customer.
    Always call this BEFORE drafting a retention offer.

    Args:
        contract_type:  'Month-to-month', 'One year', or 'Two year'
        monthly_charge: Customer's current monthly bill in USD
    """
    return get_policy(contract_type, monthly_charge)


@mcp.tool()
def get_competitor_insights(state: str = "DEFAULT", internet_service: str = "") -> dict:
    """
    Returns current competitor intelligence for a US state.

    Args:
        state:            US state name e.g. 'California'. Use 'DEFAULT' if unknown.
        internet_service: Customer's internet type e.g. 'Fiber optic'
    """
    info = get_competitor_info(state)
    info["internet_service_context"] = (
        "HIGH ALERT: Fiber optic customers are the primary target for competitor campaigns."
        if internet_service == "Fiber optic"
        else "Standard competitive pressure — emphasise reliability and support quality."
    )
    return info


@mcp.tool()
def log_retention_action(
    customer_id:    str,
    risk_score:     float,
    offer_text:     str,
    contract_type:  str,
    monthly_charge: float,
) -> dict:
    """
    Logs a retention action to the CRM queue.
    Call this as the FINAL step after the offer has been drafted.
    """
    import time
    log_id = (
        f"RET-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        f"-{str(uuid.uuid4())[:6].upper()}"
    )
    now_iso = datetime.now().isoformat()
    entry = {
        "log_id":         log_id,
        "customer_id":    customer_id,
        "risk_score":     round(risk_score, 4),
        "offer_text":     offer_text,
        "contract_type":  contract_type,
        "monthly_charge": round(monthly_charge, 2),
        "timestamp":      now_iso,
        "status":         "PENDING_CONTACT",
        "assigned_to":    "Retention Team Queue",
    }

    # In Lambda, write to DynamoDB. Locally, fall back to the JSONL file.
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        try:
            import boto3
            from decimal import Decimal
            table_name = os.environ.get("DYNAMODB_TABLE", "churn-prod-main")
            ddb = boto3.resource("dynamodb", region_name=os.environ.get("BEDROCK_REGION", "us-east-1"))
            table = ddb.Table(table_name)
            item = {
                "PK":             f"LOG#{log_id}",
                "SK":             f"CUSTOMER#{customer_id}",
                "log_date":       now_iso[:10],
                "log_id":         log_id,
                "customer_id":    customer_id,
                "risk_score":     Decimal(str(round(risk_score, 4))),
                "offer_text":     offer_text,
                "contract_type":  contract_type,
                "monthly_charge": Decimal(str(round(monthly_charge, 2))),
                "timestamp":      now_iso,
                "status":         "PENDING_CONTACT",
                "assigned_to":    "Retention Team Queue",
                "ttl":            int(time.time()) + 60 * 60 * 24 * 30,
            }
            table.put_item(Item=item)
        except Exception as e:
            import traceback, sys
            print(f"[DYNAMODB ERROR] {e}", file=sys.stderr, flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            return {
                "status":  "log_failed",
                "log_id":  log_id,
                "message": f"DynamoDB write failed: {str(e)}",
            }
    else:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    return {
        "status":  "logged",
        "log_id":  log_id,
        "message": f"Retention action queued for customer {customer_id}",
    }

# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("churn://feature_dictionary")
def feature_dictionary() -> str:
    """Explains all 35 model features. Agents use this to interpret SHAP values."""
    return json.dumps(FEATURE_DICTIONARY, indent=2)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    host      = os.getenv("MCP_HOST", "0.0.0.0")
    port      = int(os.getenv("MCP_PORT", "8001"))

    if transport == "sse":
        print(f"MCP server  [SSE]  →  http://{host}:{port}/sse", flush=True)
        mcp.run(transport="sse", host=host, port=port)
    else:
        # stdio: stdout IS the protocol pipe — no print allowed here
        mcp.run()