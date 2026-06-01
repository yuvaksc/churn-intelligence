"""
mcp_server/lambda_handler.py — Lambda entry point for MCP server.
Wraps FastMCP ASGI app with Mangum.
"""
from mcp_server.server import mcp
from mangum import Mangum

# FastMCP renamed the ASGI app method across versions.
# Try them in order of likelihood.
if hasattr(mcp, "sse_app"):
    asgi_app = mcp.sse_app()
elif hasattr(mcp, "http_app"):
    asgi_app = mcp.http_app()
elif hasattr(mcp, "get_asgi_app"):
    asgi_app = mcp.get_asgi_app()
else:
    raise RuntimeError(f"No known ASGI method on FastMCP: {dir(mcp)}")

handler = Mangum(asgi_app, lifespan="off")