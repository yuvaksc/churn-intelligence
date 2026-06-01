"""
agents/tools/mcp_tools.py — Transport-aware MCP client.

Reads MCP_TRANSPORT env var at import time:
  stdio (default):  spawns mcp_server/server.py as a subprocess — local dev
  sse:              connects to http://MCP_SERVER_URL/sse — Docker / AWS

Two usage patterns:

  1. Single call (backward compatible):
        result = await _run_mcp_tool("retention_policy_check", {...})

  2. Multi-call session — ONE subprocess / ONE connection for all calls.
     Use this in agent3 to avoid spawning 3 separate processes:
        async with mcp_session() as session:
            policy = await _call(session, "retention_policy_check", {...})
            comp   = await _call(session, "get_competitor_insights", {...})
            log    = await _call(session, "log_retention_action", {...})

The @tool wrappers below exist for backward compatibility with any code
that calls the tools individually. Agent 3 uses mcp_session() directly.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.tools import tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ── Transport configuration ───────────────────────────────────────────────────

MCP_TRANSPORT  = os.getenv("MCP_TRANSPORT", "stdio")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001/sse")

_STDIO_PARAMS  = StdioServerParameters(
    command="python",
    args=["-m", "mcp_server.server"],
    env={**os.environ, "PYTHONPATH": "."},
)


def _get_transport():
    """Return the correct transport context manager based on MCP_TRANSPORT."""
    if MCP_TRANSPORT == "sse":
        from mcp.client.sse import sse_client
        return sse_client(MCP_SERVER_URL)
    return stdio_client(_STDIO_PARAMS)


def _extract_text(result) -> str:
    """Pull text content out of an MCP CallToolResult."""
    if not result.content:
        return "{}"
    parts = []
    for block in result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
        elif hasattr(block, "data"):
            parts.append(str(block.data))
    return "".join(parts)


# ── Core async primitives ─────────────────────────────────────────────────────

@asynccontextmanager
async def mcp_session():
    """
    Open one MCP session (one subprocess or one HTTP connection).
    Use this in agent3 to make all 3 tool calls over a single connection.

    Usage:
        async with mcp_session() as session:
            result = await _call(session, "tool_name", {"arg": value})
    """
    async with _get_transport() as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def _call(session: ClientSession, tool_name: str, arguments: dict[str, Any]) -> str:
    """Make one tool call on an already-open session and return text."""
    result = await session.call_tool(tool_name, arguments)
    return _extract_text(result)


async def _run_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """
    Open a fresh session, make one call, close.
    Used by the @tool wrappers for single standalone calls.
    """
    async with mcp_session() as session:
        return await _call(session, tool_name, arguments)


def _sync(coro) -> str:
    """
    Run an async coroutine from sync context.
    Handles both 'no loop' and 'loop already running' cases.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an existing event loop (e.g. Jupyter, some test runners)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── LangChain @tool wrappers (sync, backward compatible) ─────────────────────

@tool
def retention_policy_check(contract_type: str, monthly_charge: float) -> str:
    """
    Returns the retention discount policy for a customer based on their
    contract type and monthly charge. Always call before drafting an offer.
    """
    return _sync(_run_mcp_tool("retention_policy_check", {
        "contract_type":  contract_type,
        "monthly_charge": monthly_charge,
    }))


@tool
def get_competitor_insights(state: str = "DEFAULT", internet_service: str = "") -> str:
    """
    Returns competitor intelligence for a US state to frame the retention offer.
    """
    return _sync(_run_mcp_tool("get_competitor_insights", {
        "state":            state,
        "internet_service": internet_service,
    }))


@tool
def log_retention_action(
    customer_id:    str,
    risk_score:     float,
    offer_text:     str,
    contract_type:  str,
    monthly_charge: float,
) -> str:
    """Logs a retention action to the CRM queue. Call as the final step."""
    return _sync(_run_mcp_tool("log_retention_action", {
        "customer_id":    customer_id,
        "risk_score":     risk_score,
        "offer_text":     offer_text,
        "contract_type":  contract_type,
        "monthly_charge": monthly_charge,
    }))


ALL_MCP_TOOLS = [retention_policy_check, get_competitor_insights, log_retention_action]