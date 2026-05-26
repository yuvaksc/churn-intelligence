"""
Retention discount policies by contract type.
The MCP server exposes these via retention_policy_check tool.
Agent 3 must stay within these limits when drafting offers.
"""

RETENTION_POLICIES = {
    "Month-to-month": {
        "max_discount_pct":      25,
        "max_discount_absolute": 20,
        "allowed_offers": [
            "One-time bill credit",
            "Free month of service",
            "Upgrade to fiber at same monthly price",
            "Free add-on service (Online Security or Backup)",
            "Loyalty discount for 6 months",
        ],
        "contract_incentive": (
            "Offer 10% ongoing discount if customer commits to 1-year contract"
        ),
        "constraint": (
            "Cannot exceed 25% monthly discount. "
            "No multi-year upfront discounts without manager approval."
        ),
    },
    "One year": {
        "max_discount_pct":      15,
        "max_discount_absolute": 12,
        "allowed_offers": [
            "Loyalty renewal discount",
            "Free service upgrade (speed tier)",
            "Waive one month fee",
            "Add streaming service at no cost for 3 months",
        ],
        "contract_incentive": (
            "Offer 5% off for upgrading to 2-year contract at renewal"
        ),
        "constraint": (
            "Cannot exceed 15% discount. "
            "Customer is already committed — focus on value, not deep discounts."
        ),
    },
    "Two year": {
        "max_discount_pct":      10,
        "max_discount_absolute":  8,
        "allowed_offers": [
            "Premium add-on at no cost (device protection, tech support)",
            "Speed upgrade at same price",
            "Priority support tier upgrade",
        ],
        "contract_incentive": (
            "Minimal discount needed. Emphasize service quality and reliability."
        ),
        "constraint": (
            "Cannot exceed 10% discount. "
            "Customer is on best plan — lead with service quality messaging."
        ),
    },
}


def get_policy(contract_type: str, monthly_charge: float) -> dict:
    """
    Returns the full policy dict for a given contract type,
    with dollar ceiling computed from the customer's actual charge.
    """
    policy = RETENTION_POLICIES.get(
        contract_type,
        RETENTION_POLICIES["Month-to-month"],
    ).copy()

    dollar_cap = min(
        policy["max_discount_absolute"],
        monthly_charge * policy["max_discount_pct"] / 100,
    )
    policy["max_discount_dollars"] = round(dollar_cap, 2)
    policy["monthly_charge"]       = round(monthly_charge, 2)
    policy["contract_type"]        = contract_type
    return policy