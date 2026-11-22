"""churn_data.crm — deterministic synthetic CRM support tickets.

The public dataset has no support history, so each customer gets 0–3 tickets derived
from their own features (e.g. fiber + high bill → billing complaint). Deterministic,
so every environment seeds identical data. Read by the MCP `get_open_tickets` tool.
"""

from __future__ import annotations

import re


def parse_customer_id(customer_id) -> int | None:
    """Extract the integer index from ids like 'TEST-4521' or 4521."""
    m = re.search(r"(\d+)\s*$", str(customer_id))
    return int(m.group(1)) if m else None


def tickets_for(idx: int, feats: dict) -> list[tuple]:
    contract = str(feats.get("Contract", ""))
    internet = str(feats.get("Internet Service", ""))
    tenure = int(feats.get("Tenure Months", 0) or 0)
    monthly = float(feats.get("Monthly Charges", 0) or 0)
    tech_support = str(feats.get("Tech Support", ""))
    payment = str(feats.get("Payment Method", ""))

    rules: list[tuple[str, str, str]] = []  # (subject, category, priority)
    if internet == "Fiber optic" and monthly > 80:
        rules.append(("Billing concern: monthly charge higher than expected", "Billing", "High"))
    if tenure < 6:
        rules.append(("New-customer onboarding question about setup", "Onboarding", "Medium"))
    if tech_support == "No":
        rules.append(("Reported intermittent connection drops", "Technical", "Medium"))
    if "Electronic check" in payment:
        rules.append(("Requested help switching to automatic payments", "Account", "Low"))
    if contract == "Month-to-month" and monthly > 70:
        rules.append(("Asked about contract options and pricing", "Retention", "Medium"))

    tickets = []
    for i, (subject, category, priority) in enumerate(rules[:3]):
        status = "Resolved" if (idx + i) % 3 == 0 else "Open"
        opened_at = f"2026-{((idx + i) % 12) + 1:02d}-{((idx * 7 + i) % 28) + 1:02d}"
        tickets.append((idx, subject, category, status, priority, opened_at))
    return tickets
