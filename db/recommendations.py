"""db/recommendations.py — retention/CRM action log (raw SQL, sync).

The api process is the single writer: after a war-room run it inserts one row
built from the final state. Rows accumulate across runs (no per-run wipe). The
status/assigned_to defaults match the values the MCP tool used to stamp.

All functions are blocking — async callers wrap them in asyncio.to_thread.
"""

from db.connection import connect

DEFAULT_STATUS      = "PENDING_CONTACT"
DEFAULT_ASSIGNED_TO = "Retention Team Queue"

# Column order shared by INSERT and the RetentionLogEntry response shape.
_FIELDS = (
    "log_id", "customer_id", "risk_score", "offer_text", "contract_type",
    "monthly_charge", "timestamp", "status", "assigned_to",
)


def insert_recommendation(rec: dict) -> None:
    with connect() as conn:
        conn.execute(
            f"""INSERT INTO recommendations ({", ".join(_FIELDS)})
                VALUES ({", ".join("?" for _ in _FIELDS)})""",
            tuple(rec[f] for f in _FIELDS),
        )
        conn.commit()


def list_recommendations(limit: int, status: str = "") -> list[dict]:
    """Most recent entries first (matches the old JSONL reverse-read)."""
    sql    = f"SELECT {', '.join(_FIELDS)} FROM recommendations"
    params: list = []
    if status:
        sql += " WHERE status = ?"
        params.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_recommendations_for_customer(customer_id: str, limit: int = 5) -> list[dict]:
    """Prior recommendations for one customer, newest first (the 'memory' read by
    the get_prior_recommendations MCP tool)."""
    sql = f"SELECT {', '.join(_FIELDS)} FROM recommendations WHERE customer_id = ? ORDER BY id DESC LIMIT ?"
    with connect() as conn:
        rows = conn.execute(sql, (customer_id, limit)).fetchall()
    return [dict(r) for r in rows]
