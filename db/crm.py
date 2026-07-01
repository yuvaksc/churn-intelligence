"""
db/crm.py — synthetic CRM data + read helpers (raw SQL, sqlite-only).

Backs the data-backed MCP tools (get_account_history, get_open_tickets). The read
helpers are import-light (no pandas/ML) so the mcp-server container can use them.
seed_crm() runs in the api process at startup, deriving deterministic support
tickets from the already-seeded customers table.
"""

import re
import json

from db.connection import connect

_CRM_SEED_VERSION = "1"


def _parse_idx(customer_id) -> int | None:
    """Extract the integer customer index from ids like 'TEST-4521' or 4521."""
    m = re.search(r"(\d+)\s*$", str(customer_id))
    return int(m.group(1)) if m else None


# ── Seeding ──────────────────────────────────────────────────────────────────

def _tickets_for(idx: int, feats: dict) -> list[tuple]:
    """Deterministically derive 0–3 support tickets from a customer's features."""
    contract = str(feats.get("Contract", ""))
    internet = str(feats.get("Internet Service", ""))
    tenure   = int(feats.get("Tenure Months", 0) or 0)
    monthly  = float(feats.get("Monthly Charges", 0) or 0)
    tech_sup = str(feats.get("Tech Support", ""))
    payment  = str(feats.get("Payment Method", ""))

    rules: list[tuple[str, str, str]] = []  # (subject, category, priority)
    if internet == "Fiber optic" and monthly > 80:
        rules.append(("Billing concern: monthly charge higher than expected", "Billing", "High"))
    if tenure < 6:
        rules.append(("New-customer onboarding question about setup", "Onboarding", "Medium"))
    if tech_sup == "No":
        rules.append(("Reported intermittent connection drops", "Technical", "Medium"))
    if "Electronic check" in payment:
        rules.append(("Requested help switching to automatic payments", "Account", "Low"))
    if contract == "Month-to-month" and monthly > 70:
        rules.append(("Asked about contract options and pricing", "Retention", "Medium"))

    tickets: list[tuple] = []
    for i, (subject, category, priority) in enumerate(rules[:3]):
        status    = "Resolved" if (idx + i) % 3 == 0 else "Open"
        opened_at = f"2026-{((idx + i) % 12) + 1:02d}-{((idx * 7 + i) % 28) + 1:02d}"
        tickets.append((idx, subject, category, status, priority, opened_at))
    return tickets


def seed_crm(force: bool = False) -> int:
    """Populate crm_tickets from the customers table if empty (or force-rebuild).
    Returns the ticket count. Blocking — call via asyncio.to_thread."""
    with connect() as conn:
        if force:
            conn.execute("DELETE FROM crm_tickets")
            conn.commit()
        else:
            existing = conn.execute("SELECT COUNT(*) AS n FROM crm_tickets").fetchone()["n"]
            if existing > 0:
                return existing

        customers = conn.execute("SELECT customer_id, all_features FROM customers").fetchall()
        tickets: list[tuple] = []
        for row in customers:
            tickets.extend(_tickets_for(row["customer_id"], json.loads(row["all_features"])))

        conn.executemany(
            """INSERT INTO crm_tickets (customer_id, subject, category, status, priority, opened_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            tickets,
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('crm_seed_version', ?)",
            (_CRM_SEED_VERSION,),
        )
        conn.commit()
        return len(tickets)


# ── Reads (used by the MCP server) ────────────────────────────────────────────

def get_account_history(customer_id: str) -> dict:
    """Account snapshot for one customer (tenure, contract, charges, services, standing)."""
    idx = _parse_idx(customer_id)
    if idx is None:
        return {"error": "could not parse customer id", "customer_id": customer_id}
    with connect() as conn:
        row = conn.execute("SELECT * FROM customers WHERE customer_id = ?", (idx,)).fetchone()
    if row is None:
        return {"error": "customer not found", "customer_id": customer_id}
    feats = json.loads(row["all_features"])
    return {
        "customer_id":      customer_id,
        "tenure_months":    row["tenure_months"],
        "contract":         row["contract"],
        "monthly_charges":  row["monthly_charges"],
        "total_charges":    feats.get("Total Charges"),
        "internet_service": row["internet_service"],
        "services_count":   row["services_count"],
        "payment_method":   feats.get("Payment Method"),
        "tech_support":     feats.get("Tech Support"),
        "senior_citizen":   feats.get("Senior Citizen"),
        "partner":          feats.get("Partner"),
        "dependents":       feats.get("Dependents"),
        "account_standing": "at-risk" if row["high_risk_flag"] else "stable",
    }


def get_open_tickets(customer_id: str) -> list[dict]:
    """Open support tickets for one customer (subject, category, priority, opened_at)."""
    idx = _parse_idx(customer_id)
    if idx is None:
        return []
    with connect() as conn:
        rows = conn.execute(
            """SELECT subject, category, status, priority, opened_at
               FROM crm_tickets WHERE customer_id = ? AND status = 'Open' ORDER BY id""",
            (idx,),
        ).fetchall()
    return [dict(r) for r in rows]
