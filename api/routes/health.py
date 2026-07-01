"""api/routes/health.py — GET /health"""

import asyncio

from fastapi import APIRouter

from api.schemas import HealthResponse
from db import customers as db_customers

router = APIRouter(tags=["system"])


def _counts() -> tuple[int, int, float]:
    """Bundle the three SQLite reads into one thread hop."""
    return (
        db_customers.count_customers(),
        db_customers.count_high_risk(),
        db_customers.get_threshold(),
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    import chromadb

    total, high_risk, threshold = await asyncio.to_thread(_counts)

    try:
        client      = chromadb.PersistentClient(path="data/chroma_db")
        collections = [c.name for c in client.list_collections()]
    except Exception:
        collections = []

    return HealthResponse(
        status="ok",
        model="xgb_pipeline.pkl",
        threshold=round(threshold, 4),
        test_set_size=total,
        high_risk_count=high_risk,
        chroma_collections=collections,
    )
