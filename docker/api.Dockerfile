# docker/api.Dockerfile
#
# Runs FastAPI + LangGraph + the ML/RAG stack.
# models/ and data/ are bind-mounted at runtime, not copied in.

FROM python:3.12-slim

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# gcc/g++  : compile native Python wheels
# libgomp1 : OpenMP runtime required by xgboost on Linux
# curl     : used by the HEALTHCHECK below
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── CPU-only PyTorch (CRITICAL) ───────────────────────────────────────────────
# sentence-transformers (used by ChromaDB embeddings) depends on torch.
# By default pip installs the CUDA build — ~2.5GB of NVIDIA packages you do
# NOT need on CPU. That bloat is what crashes the build with an OOM/EOF error.
# Installing the CPU-only wheel FIRST means sentence-transformers finds torch
# already satisfied and won't pull the CUDA version.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# ── Remaining Python dependencies ─────────────────────────────────────────────
COPY requirements.txt requirements_agents.txt requirements_api.txt ./
RUN pip install --no-cache-dir \
    -r requirements.txt \
    -r requirements_agents.txt \
    -r requirements_api.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY src/        ./src/
COPY api/        ./api/
COPY agents/     ./agents/
COPY rag/        ./rag/
COPY mcp_server/ ./mcp_server/

# Mount points for bind-mounted volumes
RUN mkdir -p /app/models /app/data

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV MCP_TRANSPORT=sse
# MCP_SERVER_URL is injected by docker-compose

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
