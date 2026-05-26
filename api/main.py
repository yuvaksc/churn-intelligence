"""
api/main.py — FastAPI application.

Run locally:
    uvicorn api.main:app --reload --port 8000

OpenAPI docs auto-generated at:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import initialize_state
from api.routes import health, customers, analyze, logs, metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: warm up all shared state before accepting requests.
    Runs initialize_state() in a thread since it does blocking I/O (joblib, CSV reads).
    """
    print("\nChurn Intelligence API — starting up...")
    await asyncio.to_thread(initialize_state)
    print("API ready.\n")
    yield
    print("API shutting down.")


app = FastAPI(
    title="Churn Intelligence API",
    description=(
        "Multi-agent churn prediction and retention system. "
        "XGBoost + SHAP scoring, ChromaDB RAG evidence, "
        "MCP-powered retention policy enforcement."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Next.js dev server and production domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://your-production-domain.com",  # replace at deploy time
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router)                          # /health
app.include_router(customers.router, prefix="/api")        # /api/customers
app.include_router(analyze.router,   prefix="/api")        # /api/analyze
app.include_router(logs.router,      prefix="/api")        # /api/logs
app.include_router(metrics.router,   prefix="/api")        # /api/metrics