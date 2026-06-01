"""
mcp_server/lambda_handler.py — Lambda entry point for MCP server.
Wraps FastMCP ASGI app with Mangum.
"""
from mcp_server.server import mcp
from mangum import Mangum

# Get the ASGI app from FastMCP
asgi_app = mcp.get_asgi_app()

# Lambda handler
handler = Mangum(asgi_app, lifespan="off")