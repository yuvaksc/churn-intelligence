"""MCP tool logic — the retention-policy + competitor-intelligence the MCP server
exposes to Agent 3 (mcp_server/policies.py, mcp_server/competitors.py)."""

from mcp_server.policies import get_policy
from mcp_server.competitors import get_competitor_info


def test_policy_month_to_month_ceiling():
    p = get_policy("Month-to-month", 100.0)
    assert p["contract_type"] == "Month-to-month"
    assert p["max_discount_pct"] == 25
    # dollar cap = min(absolute $20, 25% of $100 = $25) → $20
    assert p["max_discount_dollars"] == 20.0


def test_policy_dollar_cap_uses_percentage_when_lower():
    # Two year: 10% of $50 = $5, below the $8 absolute cap → $5
    p = get_policy("Two year", 50.0)
    assert p["max_discount_pct"] == 10
    assert p["max_discount_dollars"] == 5.0


def test_policy_unknown_contract_falls_back_to_month_to_month():
    assert get_policy("Weekly", 80.0)["max_discount_pct"] == 25


def test_competitor_known_state():
    assert get_competitor_info("Texas")["top_competitor"] == "Lone Star Broadband"


def test_competitor_unknown_state_falls_back_to_default():
    assert get_competitor_info("Atlantis") == get_competitor_info("DEFAULT")
