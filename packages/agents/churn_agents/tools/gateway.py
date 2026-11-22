"""churn_agents.tools.gateway — the tool-calling contract Agent 3 depends on.

`ToolGateway` is all Agent 3 sees: discover tool specs, call a tool by name.
`McpToolGateway` is the real one (an open MCP session); tests use a fake that records
calls, which is how "policy checked before any offer" is asserted.
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Protocol


class ToolGateway(Protocol):
    async def list_specs(self) -> list[dict]: ...
    async def call(self, name: str, args: dict[str, Any]) -> str: ...


class McpToolGateway:
    def __init__(self, session):
        self._session = session

    async def list_specs(self) -> list[dict]:
        from churn_agents.tools.mcp_tools import list_tool_specs
        return await list_tool_specs(self._session)

    async def call(self, name: str, args: dict[str, Any]) -> str:
        from churn_agents.tools.mcp_tools import _call
        return await _call(self._session, name, args)


@asynccontextmanager
async def mcp_gateway() -> AsyncIterator[McpToolGateway]:
    """Open one MCP session for a whole Agent 3 run."""
    from churn_agents.tools.mcp_tools import mcp_session
    async with mcp_session() as session:
        yield McpToolGateway(session)
