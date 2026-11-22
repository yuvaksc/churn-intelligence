"""Integration tests against a real Postgres with pgvector.

Set TEST_DATABASE_URL to a THROWAWAY database (CI starts a pgvector service; locally
`docker compose up db` exposes one on port 5433). Each session drops and recreates the
`public` schema from supabase/migrations — never point this at Supabase.
Skipped when TEST_DATABASE_URL is unset.
"""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = sorted((ROOT / "supabase" / "migrations").glob("*.sql"))


def pytest_collection_modifyitems(config, items):
    if os.getenv("TEST_DATABASE_URL"):
        return
    skip = pytest.mark.skip(reason="TEST_DATABASE_URL not set")
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def pg_url():
    import psycopg

    url = os.environ["TEST_DATABASE_URL"]
    if "supabase" in url:
        pytest.exit("Refusing to run destructive integration tests against Supabase", returncode=2)
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("drop schema if exists public cascade")
        conn.execute("create schema public")
        for m in MIGRATIONS:
            conn.execute(m.read_text(encoding="utf-8"))
    return url


@pytest.fixture()
def pg(pg_url, monkeypatch):
    """DATABASE_URL pointed at the test DB, tables emptied, three customers loaded."""
    import psycopg

    monkeypatch.setenv("DATABASE_URL", pg_url)
    from churn_data import db

    with psycopg.connect(pg_url, autocommit=True) as conn:
        conn.execute("truncate customers, recommendations, meta, rag_documents restart identity cascade")
    db.upsert_customers([_customer(i) for i in range(1, 4)])
    db.set_meta("threshold", "0.5")
    return db


def _customer(i: int) -> dict:
    score = 0.9 if i == 1 else 0.2
    return {
        "customer_id": i, "risk_score": score, "risk_label": "HIGH" if score >= 0.5 else "LOW",
        "contract": "Month-to-month", "monthly_charges": 70.0 + i, "tenure_months": i * 3,
        "internet_service": "Fiber optic", "payment_method": "Electronic check", "services_count": 3,
        "high_risk_flag": int(score >= 0.5), "true_label": 1 if i == 1 else 0,
        "all_features": {"Contract": "Month-to-month", "Monthly Charges": 70.0 + i, "Tech Support": "No"},
        "top_shap_drivers": [],
    }
