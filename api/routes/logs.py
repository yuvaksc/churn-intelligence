"""api/routes/logs.py — GET /api/logs"""

import asyncio

from fastapi import APIRouter, Query

from api.schemas import RetentionLogEntry
from db import recommendations as db_recs

router = APIRouter(tags=["logs"])


@router.get("/logs", response_model=list[RetentionLogEntry])
async def get_logs(
    limit:  int = Query(20, ge=1, le=200),
    status: str = Query("", description="Filter by status e.g. PENDING_CONTACT"),
):
    """Returns the most recent retention log entries, newest first."""
    return await asyncio.to_thread(db_recs.list_recommendations, limit, status)
