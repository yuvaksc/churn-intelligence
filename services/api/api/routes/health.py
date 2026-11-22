"""api/routes/health.py — GET /health (public; used by Docker, Vercel and the smoke test)."""

import asyncio

from fastapi import APIRouter, Depends

from api.deps import get_customer_repo
from api.schemas import HealthResponse
from churn_data.ports import CustomerRepository

router = APIRouter(tags=["system"])


def _snapshot(repo: CustomerRepository) -> tuple[int, int, float, str, int]:
    """One thread hop for all reads; a database outage reports as status=degraded."""
    from churn_data import db
    with db.tx() as c:
        docs = c.execute("select count(*) as n from rag_documents").fetchone()
    return (repo.count(), repo.count_high_risk(), repo.threshold(),
            db.get_meta("model_version") or "unknown", int(docs["n"]) if docs else 0)


@router.get("/health", response_model=HealthResponse)
async def health(repo: CustomerRepository = Depends(get_customer_repo)):
    try:
        total, high_risk, threshold, model, docs = await asyncio.to_thread(_snapshot, repo)
        status = "ok"
    except Exception:
        total, high_risk, threshold, model, docs, status = 0, 0, 0.5, "unknown", 0, "degraded"
    return HealthResponse(status=status, model=model, threshold=round(threshold, 4),
                          test_set_size=total, high_risk_count=high_risk, rag_documents=docs)
