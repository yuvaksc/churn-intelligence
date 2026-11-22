"""api/routes/logs.py — GET /api/logs"""

import asyncio

from fastapi import APIRouter, Depends, Query

from api.deps import get_recommendation_repo
from api.schemas import RetentionLogEntry
from churn_data.ports import RecommendationRepository

router = APIRouter(tags=["logs"])


@router.get("/logs", response_model=list[RetentionLogEntry])
async def get_logs(
    limit:  int = Query(20, ge=1, le=200),
    status: str = Query("", description="Filter by status e.g. PENDING_CONTACT"),
    repo: RecommendationRepository = Depends(get_recommendation_repo),
):
    """The most recent retention offers, newest first."""
    return await asyncio.to_thread(repo.list, limit, status)
