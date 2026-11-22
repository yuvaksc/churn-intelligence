"""Repositories, CRM reads and batch-score writes against real Postgres."""

from churn_data.db import PgCustomerRepository, PgRecommendationRepository


def _rec(log_id: str, customer: str = "TEST-1") -> dict:
    return {"log_id": log_id, "customer_id": customer, "risk_score": 0.9, "offer_text": "10% off",
            "contract_type": "Month-to-month", "monthly_charge": 71.0, "timestamp": "2026-11-20T00:00:00",
            "status": "PENDING_CONTACT", "assigned_to": "queue"}


def test_customer_repository_reads(pg):
    repo = PgCustomerRepository()
    assert repo.count() == 3 and repo.count_high_risk() == 1
    assert [r["customer_id"] for r in repo.list(10, 0, True)] == [1]
    assert repo.get(1)["all_features"]["Contract"] == "Month-to-month"
    assert repo.features(99) is None
    assert repo.threshold() == 0.5
    assert len(repo.scoring_rows()) == 3


def test_recommendations_newest_first_and_per_customer(pg):
    repo = PgRecommendationRepository()
    repo.insert(_rec("RET-1"))
    repo.insert(_rec("RET-2", "TEST-2"))
    assert [r["log_id"] for r in repo.list()] == ["RET-2", "RET-1"]
    assert [r["log_id"] for r in repo.for_customer("TEST-1")] == ["RET-1"]
    assert repo.list(status="CONTACTED") == []


def test_crm_reads_for_mcp_tools(pg):
    pg.replace_crm_tickets([(1, "Billing concern", "Billing", "Open", "High", "2026-01-01"),
                            (1, "Old issue", "Technical", "Resolved", "Low", "2026-01-02")])
    assert [t["subject"] for t in pg.open_tickets("TEST-1")] == ["Billing concern"]
    assert pg.account_history("TEST-1")["account_standing"] == "at-risk"
    assert pg.account_history("TEST-99")["error"] == "customer not found"


def test_batch_predictions_update_customer_rows(pg):
    pg.insert_predictions([{"customer_id": 2, "model_version": "m:2", "risk_score": 0.8, "risk_label": "HIGH",
                            "threshold": 0.5, "shap_drivers": [{"feature": "Contract"}]}])
    row = PgCustomerRepository().get(2)
    assert row["risk_label"] == "HIGH" and row["top_shap_drivers"] == [{"feature": "Contract"}]
    with pg.tx() as c:
        assert c.execute("select count(*) as n from predictions").fetchone()["n"] == 1


def test_meta_round_trip(pg):
    pg.set_meta("model_version", "m:2")
    assert pg.get_meta("model_version") == "m:2"


def test_public_api_roles_cannot_read_tables(pg):
    """RLS on, no policies: Supabase's anon/authenticated roles see nothing."""
    import psycopg
    with psycopg.connect(pg.database_url(), autocommit=True) as conn:
        for role in ("anon", "authenticated"):
            conn.execute(f"do $$ begin if not exists (select 1 from pg_roles where rolname = '{role}') "
                         f"then create role {role} nologin; end if; end $$")
            conn.execute(f"grant usage on schema public to {role}; grant select on customers to {role}")
        with conn.transaction():
            conn.execute("set local role anon")
            assert conn.execute("select count(*) from customers").fetchone()[0] == 0
