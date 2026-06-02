FROM python:3.12-slim

WORKDIR /app

# curl for the HEALTHCHECK below
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before copying code (layer-cache friendly)
COPY mcp_server/requirements.txt ./mcp_requirements.txt
RUN pip install --no-cache-dir -r mcp_requirements.txt

COPY mcp_server/ ./mcp_server/

# Pre-create the log directory so the bind mount lands correctly and
# log_retention_action() can write without a mkdir race on first call
RUN mkdir -p /app/models/reports

ENV MCP_TRANSPORT=sse
ENV MCP_PORT=8001
ENV MCP_HOST=0.0.0.0
ENV PYTHONPATH=/app

EXPOSE 8001

# FastMCP exposes /sse (SSE stream), not /health.
# curl returns 200 as soon as the server sends the response header,
# even though the body streams indefinitely — so --max-time 3 is enough.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD sh -c 'curl -so /dev/null --max-time 3 http://localhost:8001/sse; e=$?; [ $e -eq 0 ] || [ $e -eq 28 ]'

CMD ["python", "-m", "mcp_server.server"]
