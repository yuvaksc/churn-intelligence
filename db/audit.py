"""db/audit.py — append-only audit trail: one row per war-room analysis (raw SQL).

Captures everything needed to reconstruct a run: routing path, retrieved docs,
tools called, policy, offer, guardrail results and latency. Written by the api
(single writer); JSON-typed columns are (de)serialized here.
"""

import json

from db.connection import connect

_FIELDS = (
    "trace_id", "customer_id", "customer_state", "risk_label", "risk_score",
    "agent_path", "tools_used", "retrieved_profiles", "churn_reasons_count",
    "policy", "retention_offer", "crm_log_id", "guardrails", "input_scan",
    "latency_ms", "created_at",
)
_JSON_FIELDS = {"agent_path", "tools_used", "retrieved_profiles", "policy",
                "guardrails", "input_scan"}


def insert_audit(row: dict) -> None:
    vals = [json.dumps(row.get(f)) if f in _JSON_FIELDS else row.get(f) for f in _FIELDS]
    with connect() as conn:
        conn.execute(
            f"INSERT INTO audit_log ({', '.join(_FIELDS)}) "
            f"VALUES ({', '.join('?' for _ in _FIELDS)})",
            vals,
        )
        conn.commit()


def _hydrate(row) -> dict:
    d = dict(row)
    for f in _JSON_FIELDS:
        if d.get(f) is not None:
            try:
                d[f] = json.loads(d[f])
            except Exception:
                pass
    return d


def get_audit(trace_id: str) -> dict | None:
    with connect() as conn:
        r = conn.execute("SELECT * FROM audit_log WHERE trace_id = ?", (trace_id,)).fetchone()
    return _hydrate(r) if r else None


def list_audit(limit: int = 20) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_hydrate(r) for r in rows]
