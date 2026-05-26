# docker/api.Dockerfile
#
# This container runs FastAPI + LangGraph + all the ML dependencies.
# It's larger than the MCP container because it carries:
#   - xgboost, shap, sklearn (the ML stack)
#   - langchain, langgraph (the agent stack)
#   - chromadb + sentence-transformers (the RAG stack)
#   - fastapi + uvicorn (the API layer)
#
# models/ and data/ are NOT copied into this image.
# They are bind-mounted at runtime from your host machine.
# This means: when you retrain the model, the container picks up
# the new pkl files automatically on next restart — no rebuild needed.

FROM python:3.12-slim

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# Some Python packages (numpy, sentence-transformers) need C build tools.
# Install them first, then remove build tools to keep the image lean.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy all three requirements files, install together.
# One RUN command = one layer. Fewer layers = smaller image.
COPY requirements.txt requirements_agents.txt requirements_api.txt ./
RUN pip install --no-cache-dir \
    -r requirements.txt \
    -r requirements_agents.txt \
    -r requirements_api.txt

# ── Application code ──────────────────────────────────────────────────────────
# Copy only the Python packages the API needs.
# models/ and data/ are excluded — they'll be bind-mounted via docker-compose.
COPY src/        ./src/
COPY api/        ./api/
COPY agents/     ./agents/
COPY rag/        ./rag/
COPY mcp_server/ ./mcp_server/

# Create mount-point directories so bind mounts land in the right place
RUN mkdir -p /app/models /app/data

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
# MCP_TRANSPORT=sse: agent3 connects to mcp-server container over HTTP,
# not by spawning a subprocess (which won't work across containers).
ENV MCP_TRANSPORT=sse
# MCP_SERVER_URL is set in docker-compose (points to the mcp-server service).
# We don't hardcode it here so the same image works locally and on AWS.

EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
# Docker uses this to know when the container is ready.
# --interval: check every 30s
# --timeout: fail if no response in 10s
# --start-period: give the app 60s to start before checking (ML loading is slow)
# --retries: mark unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
