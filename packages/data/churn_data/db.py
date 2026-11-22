"""churn_data.db — Postgres data access (Supabase in production, pgvector Postgres locally).

Raw SQL with psycopg 3, no ORM. One short transaction per call; functions are
blocking, so async callers wrap them in `asyncio.to_thread`.

Connection: DATABASE_URL. With Supabase use the pooler URI; `prepare_threshold=None`
keeps psycopg from creating server-side prepared statements, which the transaction
pooler does not support.

Repositories implement the interfaces in churn_data.ports, so the API and agents
can be tested with in-memory fakes.
"""

from __future__ import annotations

import builtins
import os
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

DEFAULT_STATUS = "PENDING_CONTACT"
DEFAULT_ASSIGNED_TO = "Retention Team Queue"

_SUMMARY_COLS = ("customer_id", "risk_score", "risk_label", "contract", "monthly_charges",
                 "tenure_months", "internet_service", "services_count", "high_risk_flag", "true_label")
_REC_FIELDS = ("log_id", "customer_id", "risk_score", "offer_text", "contract_type",
               "monthly_charge", "timestamp", "status", "assigned_to")


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (see .env.example)")
    return url


@contextmanager
def tx() -> Iterator[psycopg.Connection[dict[str, Any]]]:
    """One connection + transaction; rows come back as dicts."""
    with psycopg.connect(database_url(), row_factory=dict_row, prepare_threshold=None) as conn:
        with conn.transaction():
            yield conn


# ── Customers ─────────────────────────────────────────────────────────────────

class PgCustomerRepository:
    def list(self, limit: int, offset: int, risk_only: bool) -> builtins.list[dict]:
        sql = f"select {', '.join(_SUMMARY_COLS)} from customers"
        if risk_only:
            sql += " where risk_label = 'HIGH'"
        sql += " order by risk_score desc nulls last limit %s offset %s"
        with tx() as c:
            return c.execute(sql, [limit, offset]).fetchall()

    def get(self, customer_id: int) -> dict | None:
        cols = ", ".join(_SUMMARY_COLS + ("all_features", "top_shap_drivers"))
        with tx() as c:
            return c.execute(f"select {cols} from customers where customer_id = %s", [customer_id]).fetchone()

    def features(self, customer_id: int) -> dict | None:
        with tx() as c:
            row = c.execute("select all_features from customers where customer_id = %s", [customer_id]).fetchone()
        return row["all_features"] if row else None

    def count(self) -> int:
        with tx() as c:
            row = c.execute("select count(*) as n from customers").fetchone()
        return int(row["n"]) if row else 0

    def count_high_risk(self) -> int:
        with tx() as c:
            row = c.execute("select count(*) as n from customers where risk_label = 'HIGH'").fetchone()
        return int(row["n"]) if row else 0

    def threshold(self, default: float = 0.5) -> float:
        value = get_meta("threshold")
        return float(value) if value is not None else default

    def scoring_rows(self) -> builtins.list[dict]:
        with tx() as c:
            return c.execute("select risk_score, true_label, monthly_charges from customers "
                             "where risk_score is not null").fetchall()


# ── Recommendations (offers) ─────────────────────────────────────────────────

class PgRecommendationRepository:
    def insert(self, rec: dict) -> None:
        with tx() as c:
            c.execute(f"insert into recommendations ({', '.join(_REC_FIELDS)}) "
                      f"values ({', '.join(['%s'] * len(_REC_FIELDS))})", [rec[f] for f in _REC_FIELDS])

    def list(self, limit: int = 200, status: str = "") -> builtins.list[dict]:
        sql = f"select {', '.join(_REC_FIELDS)} from recommendations"
        params: builtins.list[object] = []
        if status:
            sql += " where status = %s"
            params.append(status)
        sql += " order by id desc limit %s"
        params.append(limit)
        with tx() as c:
            return c.execute(sql, params).fetchall()

    def for_customer(self, customer_id: str, limit: int = 5) -> builtins.list[dict]:
        with tx() as c:
            return c.execute(f"select {', '.join(_REC_FIELDS)} from recommendations where customer_id = %s "
                             "order by id desc limit %s", [customer_id, limit]).fetchall()


# ── CRM reads (MCP server tools) ─────────────────────────────────────────────

def account_history(customer_id: str) -> dict:
    from churn_data.crm import parse_customer_id
    idx = parse_customer_id(customer_id)
    if idx is None:
        return {"error": "could not parse customer id", "customer_id": customer_id}
    with tx() as c:
        row = c.execute("select * from customers where customer_id = %s", [idx]).fetchone()
    if row is None:
        return {"error": "customer not found", "customer_id": customer_id}
    feats = row["all_features"] or {}
    return {
        "customer_id": customer_id, "tenure_months": row["tenure_months"], "contract": row["contract"],
        "monthly_charges": row["monthly_charges"], "total_charges": feats.get("Total Charges"),
        "internet_service": row["internet_service"], "services_count": row["services_count"],
        "payment_method": feats.get("Payment Method"), "tech_support": feats.get("Tech Support"),
        "senior_citizen": feats.get("Senior Citizen"), "partner": feats.get("Partner"),
        "dependents": feats.get("Dependents"),
        "account_standing": "at-risk" if row["high_risk_flag"] else "stable",
    }


def open_tickets(customer_id: str) -> list[dict]:
    from churn_data.crm import parse_customer_id
    idx = parse_customer_id(customer_id)
    if idx is None:
        return []
    with tx() as c:
        return c.execute("select subject, category, status, priority, opened_at from crm_tickets "
                         "where customer_id = %s and status = 'Open' order by id", [idx]).fetchall()


# ── Writes (seeding, batch scoring) ──────────────────────────────────────────

def get_meta(key: str) -> str | None:
    with tx() as c:
        row = c.execute("select value from meta where key = %s", [key]).fetchone()
    return row["value"] if row else None


def set_meta(key: str, value: str) -> None:
    with tx() as c:
        c.execute("insert into meta (key, value) values (%s, %s) "
                  "on conflict (key) do update set value = excluded.value", [key, value])


def upsert_customers(rows: list[dict]) -> int:
    cols = ("customer_id", "risk_score", "risk_label", "contract", "monthly_charges", "tenure_months",
            "internet_service", "payment_method", "services_count", "high_risk_flag", "true_label",
            "all_features", "top_shap_drivers", "model_version")
    updates = ", ".join(f"{c} = excluded.{c}" for c in cols[1:])
    sql = (f"insert into customers ({', '.join(cols)}) values ({', '.join(['%s'] * len(cols))}) "
           f"on conflict (customer_id) do update set {updates}")
    params = [[r["customer_id"], r.get("risk_score"), r.get("risk_label"), r["contract"], r["monthly_charges"],
               r["tenure_months"], r["internet_service"], r.get("payment_method"), r.get("services_count", 0),
               r.get("high_risk_flag", 0), r.get("true_label"), Jsonb(r["all_features"]),
               Jsonb(r.get("top_shap_drivers", [])), r.get("model_version")] for r in rows]
    with tx() as c:
        with c.cursor() as cur:
            cur.executemany(sql, params)
    return len(rows)


def insert_predictions(preds: list[dict]) -> int:
    """Append to `predictions` and keep the latest score/SHAP on `customers`."""
    with tx() as c:
        with c.cursor() as cur:
            cur.executemany(
                "insert into predictions (customer_id, model_version, risk_score, risk_label, threshold, "
                "shap_drivers) values (%s, %s, %s, %s, %s, %s)",
                [[p["customer_id"], p["model_version"], p["risk_score"], p["risk_label"], p["threshold"],
                  Jsonb(p["shap_drivers"])] for p in preds])
            cur.executemany(
                "update customers set risk_score = %s, risk_label = %s, top_shap_drivers = %s, "
                "model_version = %s where customer_id = %s",
                [[p["risk_score"], p["risk_label"], Jsonb(p["shap_drivers"]), p["model_version"],
                  p["customer_id"]] for p in preds])
    return len(preds)


def replace_crm_tickets(tickets: list[tuple]) -> int:
    """tickets: (customer_id, subject, category, status, priority, opened_at)."""
    with tx() as c:
        c.execute("delete from crm_tickets")
        with c.cursor() as cur:
            cur.executemany("insert into crm_tickets (customer_id, subject, category, status, priority, opened_at) "
                            "values (%s, %s, %s, %s, %s, %s)", tickets)
    return len(tickets)

