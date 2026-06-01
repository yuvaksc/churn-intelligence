"""
api/main.py — FastAPI application.

Run locally:
    uvicorn api.main:app --reload --port 8000

OpenAPI docs auto-generated at:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import initialize_state
from api.routes import health, customers, analyze, logs, metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: warm up all shared state before accepting requests.
    (Used by local uvicorn; Lambda uses lazy init via middleware below.)
    """
    print("\nChurn Intelligence API — starting up...")
    if not os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
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
        "https://your-production-domain.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-initialize state on first request in Lambda
# (Lambda's init phase has a hard 10s limit; loading our state takes longer)
_lambda_initialized = False

@app.middleware("http")
async def ensure_initialized(request: Request, call_next):
    global _lambda_initialized
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") and not _lambda_initialized:
        print("Lambda: initializing state on first request...")
        initialize_state()
        _lambda_initialized = True
        print("State initialized.")
    return await call_next(request)

# Routes
app.include_router(health.router)                          # /health
app.include_router(customers.router, prefix="/api")        # /api/customers
app.include_router(analyze.router,   prefix="/api")        # /api/analyze
app.include_router(logs.router,      prefix="/api")        # /api/logs
app.include_router(metrics.router,   prefix="/api")        # /api/metrics

# Lambda handler — wraps FastAPI app for AWS Lambda + API Gateway
from mangum import Mangum
lambda_handler = Mangum(app, lifespan="off")