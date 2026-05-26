"""api/routes/logs.py — GET /api/logs"""

import json
from pathlib import Path

from fastapi import APIRouter, Query, HTTPException
from api.schemas import RetentionLogEntry

router   = APIRouter(tags=["logs"])
LOG_PATH = Path("models/reports/retention_log.jsonl")


@router.get("/logs", response_model=list[RetentionLogEntry])
async def get_logs(
    limit:  int = Query(20, ge=1, le=200),
    status: str = Query("", description="Filter by status e.g. PENDING_CONTACT"),
):
    """Returns the most recent retention log entries, newest first."""
    if not LOG_PATH.exists():
        return []

    lines = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    entries = []

    for line in reversed(lines):
        try:
            entry = json.loads(line)
            if status and entry.get("status") != status:
                continue
            entries.append(RetentionLogEntry(**entry))
            if len(entries) >= limit:
                break
        except (json.JSONDecodeError, Exception):
            continue

    return entries