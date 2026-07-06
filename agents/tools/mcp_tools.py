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


async def list_tool_specs(session: ClientSession) -> list[dict]:
    """
    Discover the server's tools (MCP Capability Exchange) and return them as
    OpenAI-format function specs for llm.bind_tools(). The LLM then decides which
    to call at runtime.
    """
    result = await session.list_tools() 
    specs: list[dict] = []
    for t in result.tools:
        specs.append({
            "type": "function",
            "function": {
                "name":        t.name,
                "description": t.description or "",
                "parameters":  t.inputSchema or {"type": "object", "properties": {}},
            },
        })
    return specs
