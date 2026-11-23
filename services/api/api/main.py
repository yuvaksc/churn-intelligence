"""
api/main.py — FastAPI application.

Run locally (also starts the MCP server):
    python scripts/dev_api.py

OpenAPI docs: http://localhost:8000/docs

Data lives in Postgres (DATABASE_URL — Supabase in production, the pgvector container
locally). Schema: supabase/migrations/. Data: `python -m churn_ml.scoring --seed`.
"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import customers, health, logs, metrics, ws


def _warm_models() -> None:
    from churn_agents.deps import default_deps
    from churn_rag.retrieve import warm_up
    default_deps().predictor.threshold      # first attribute access builds the shared predictor
    warm_up()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the churn model and the RAG models in the background at startup, so the first
    # analysis doesn't pay for them (Agent 1 ~10 s, Agent 2 ~20 s on a laptop).
    # Off by default (tests, serverless cold starts).
    warm = None
    if os.getenv("WARM_MODELS", "").lower() in ("1", "true", "yes"):
        warm = asyncio.create_task(asyncio.to_thread(_warm_models))
        warm.add_done_callback(lambda t: print(
            "Models loaded (churn model + RAG)." if t.cancelled() or not t.exception()
            else f"RAG warm-up failed: {t.exception()}", flush=True))
    print("\nChurn Intelligence API ready.\n")
    yield
    if warm and not warm.done():
        warm.cancel()


app = FastAPI(
    title="Churn Intelligence API",
    description=(
        "Multi-agent churn prediction and retention. Calibrated XGBoost with batch SHAP, "
        "pgvector hybrid RAG evidence, MCP-powered retention policy and CRM tools."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# CORS: local dev origins plus deployed origins from CORS_ORIGINS (comma-separated).
# On Vercel the web app and the API share one origin, so CORS is not involved there.
_origins = ["http://localhost:3000", "http://localhost:3100"] + [
    o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",   # any local dev port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)                            # /health
app.include_router(customers.router,   prefix="/api")        # /api/customers
app.include_router(logs.router,        prefix="/api")        # /api/logs
app.include_router(metrics.router,     prefix="/api")        # /api/metrics
app.include_router(ws.router,          prefix="/api")        # /api/analyze/{id}/ws
