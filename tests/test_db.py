from db.connection import connect
from db import recommendations as R
from db import audit as A


def test_schema_has_tables(db):
    with connect() as conn:
        tables = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"customers", "recommendations", "meta", "audit_log", "crm_tickets"} <= tables


def test_recommendations_roundtrip(db):
    rec = {
        "log_id": "RET-T1", "customer_id": "TEST-1", "risk_score": 0.9,
        "offer_text": "10% loyalty discount", "contract_type": "Month-to-month",
        "monthly_charge": 50.0, "timestamp": "2026-01-01T00:00:00",
        "status": R.DEFAULT_STATUS, "assigned_to": R.DEFAULT_ASSIGNED_TO,
    }
    R.insert_recommendation(rec)
    rows = R.list_recommendations(10)
    assert rows and rows[0]["log_id"] == "RET-T1"

    mem = R.get_recommendations_for_customer("TEST-1")
    assert mem and mem[0]["customer_id"] == "TEST-1"


def test_audit_roundtrip_hydrates_json(db):
    row = {
        "trace_id": "trace-xyz", "customer_id": "TEST-1", "customer_state": "DEFAULT",
        "risk_label": "HIGH", "risk_score": 0.9,
        "agent_path": ["agent1", "agent2", "FINISH"],
        "tools_used": ["retention_policy_check"],
        "retrieved_profiles": [{"similarity": 0.5}], "churn_reasons_count": 3,
        "policy": {"max_discount_pct": 25}, "retention_offer": "an offer",
        "crm_log_id": "RET-T1", "guardrails": {"passed": True},
        "input_scan": {"ok": True}, "latency_ms": 1200,
        "created_at": "2026-01-01T00:00:00",
    }
    A.insert_audit(row)

    got = A.get_audit("trace-xyz")
    assert got is not None
    assert got["agent_path"] == ["agent1", "agent2", "FINISH"]   # JSON hydrated back to list
    assert got["guardrails"]["passed"] is True
    assert "trace-xyz" in {r["trace_id"] for r in A.list_audit(10)}
