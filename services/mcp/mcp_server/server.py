"""
mcp_server/server.py — FastMCP server with dual transport support.

Persistent (HTTP/SSE) — local dev (scripts/dev_api.py) and Docker:
    MCP_TRANSPORT=sse  MCP_PORT=8001  MCP_HOST=0.0.0.0

Per-run subprocess (stdio) — serverless (Vercel), where no long-lived server exists:
    python -m mcp_server.server        (MCP_TRANSPORT unset)

The CRM tools read Postgres (DATABASE_URL), read-only.

Same binary, same tools, same logic — only the transport changes.
The env var is the only switch between local and deployed behaviour.
"""

import json
import os
import uuid
from datetime import datetime

try:
    from fastmcp import FastMCP                 # local / Docker: full FastMCP (SSE server)
except ImportError:                             # serverless: the MCP SDK's built-in FastMCP,
    from mcp.server.fastmcp import FastMCP      # stdio only (keeps the function bundle small)
from mcp_server.policies import get_policy
from mcp_server.competitors import get_competitor_info, FEATURE_DICTIONARY
from churn_data import db

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
    log_id = (
        f"RET-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        f"-{str(uuid.uuid4())[:6].upper()}"
    )
    # Persistence is owned by the api process (Postgres `recommendations` table).
    # This tool just mints the CRM ticket id and returns it; the API writes the
    # full record after the war-room run completes.
    return {
        "status":  "logged",
        "log_id":  log_id,
        "message": f"Retention action queued for customer {customer_id}",
    }


# ── Data-backed CRM tools (read-only, Postgres) ──────────────────────────────

@mcp.tool()
def get_account_history(customer_id: str) -> dict:
    """
    Returns the customer's account snapshot: tenure, contract, monthly/total
    charges, services, payment method, and current account standing.
    Call to ground the retention offer in the customer's actual account.

    Args:
        customer_id: war-room customer id, e.g. 'TEST-4521'
    """
    return db.account_history(customer_id)


@mcp.tool()
def get_open_tickets(customer_id: str) -> list:
    """
    Returns the customer's OPEN support tickets (subject, category, priority).
    Use to spot unresolved pain points before drafting an offer.

    Args:
        customer_id: war-room customer id, e.g. 'TEST-4521'
    """
    return db.open_tickets(customer_id)


@mcp.tool()
def get_prior_recommendations(customer_id: str) -> list:
    """
    Returns prior retention recommendations already made for this customer
    (past offer text and the risk score at the time). Use to avoid repeating
    an offer the customer has already been given.

    Args:
        customer_id: war-room customer id, e.g. 'TEST-4521'
    """
    return db.PgRecommendationRepository().for_customer(customer_id)


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