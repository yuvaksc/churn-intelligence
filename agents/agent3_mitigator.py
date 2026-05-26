"""
agents/agent3_mitigator.py — Mitigation Architect (Agent 3).  [async]

Changes from sync version:
  - async def agent3_node
  - asyncio.run() wrapper removed — mcp_session() is natively awaited
  - llm.invoke() -> await llm.ainvoke()

One MCP session, three tool calls, one LLM call — all in one async context.
"""

import json

from agents.state import WarRoomState
from agents.llm_config import get_creative_llm
from agents.tools.mcp_tools import mcp_session, _call


_OFFER_PROMPT = """\
You are a senior retention specialist drafting a personalised offer for a high-risk customer.

-- RISK SUMMARY --
{risk_summary}

-- EVIDENCE REPORT --
{evidence_report}

-- RETENTION POLICY --
Contract type:       {contract_type}
Maximum discount:    {max_discount_pct}% (${max_discount_dollars:.2f}/month cap)
Allowed offer types: {allowed_offers}
Contract incentive:  {contract_incentive}
Constraint:          {constraint}

-- COMPETITOR INTELLIGENCE --
Main competitor:     {top_competitor}
Their pricing:       {avg_competitor_price}
Their promotions:    {current_promotions}
Their strengths:     {competitor_strengths}
Our advantage:       {our_advantage}

-- CUSTOMER CONTEXT --
Monthly charge:      ${monthly_charge:.2f}
Tenure:              {tenure_months} months
Internet service:    {internet_service}

Draft a retention offer in exactly 3 labelled parts:

HOOK:    (1 sentence) Acknowledge the customer's tenure or specific situation.
         Reference their actual tenure and service type, not generic language.

OFFER:   (2 sentences) Specific offer using allowed offer types.
         Stay within the policy discount ceiling.
         Counter {top_competitor}'s promotion implicitly — frame as what the
         customer would LOSE by switching, not just what they gain by staying.

URGENCY: (1 sentence) Time-bound call to action with a specific deadline.

Rules:
  - Write in second person
  - Do not exceed the discount ceiling
  - Do not name the competitor directly to the customer
  - Keep total length under 120 words
"""


async def agent3_node(state: WarRoomState) -> dict:
    print("\n" + "-" * 52)
    print("  [Agent 3 - Mitigation Architect]")
    print("-" * 52)

    customer_raw   = state["customer_raw"]
    contract_type  = str(customer_raw.get("Contract", "Month-to-month"))
    monthly_charge = float(customer_raw.get("Monthly Charges", 0.0))
    tenure_months  = int(customer_raw.get("Tenure Months", 0))
    internet_svc   = str(customer_raw.get("Internet Service", ""))
    customer_state = state.get("customer_state", "DEFAULT")

    async with mcp_session() as session:

        print("  [MCP] retention_policy_check...")
        policy_raw = await _call(session, "retention_policy_check", {
            "contract_type":  contract_type,
            "monthly_charge": monthly_charge,
        })
        policy = json.loads(policy_raw)
        print(f"  Policy:     max {policy['max_discount_pct']}%  (${policy['max_discount_dollars']:.2f}/month)")

        print("  [MCP] get_competitor_insights...")
        comp_raw = await _call(session, "get_competitor_insights", {
            "state":            customer_state,
            "internet_service": internet_svc,
        })
        comp = json.loads(comp_raw)
        print(f"  Competitor: {comp['top_competitor']}  ({comp['avg_competitor_price']})")

        print("  [LLM] Drafting personalised offer...")
        prompt = _OFFER_PROMPT.format(
            risk_summary=state["risk_summary"],
            evidence_report=state["evidence_report"],
            contract_type=contract_type,
            max_discount_pct=policy["max_discount_pct"],
            max_discount_dollars=policy["max_discount_dollars"],
            allowed_offers=", ".join(policy["allowed_offers"]),
            contract_incentive=policy["contract_incentive"],
            constraint=policy["constraint"],
            top_competitor=comp["top_competitor"],
            avg_competitor_price=comp["avg_competitor_price"],
            current_promotions=comp["current_promotions"],
            competitor_strengths=", ".join(comp.get("competitor_strengths", [])),
            our_advantage=comp["our_advantage"],
            monthly_charge=monthly_charge,
            tenure_months=tenure_months,
            internet_service=internet_svc,
        )
        retention_offer = (await get_creative_llm().ainvoke(prompt)).content.strip()
        print(f"  Offer drafted  ({len(retention_offer)} chars)")

        print("  [MCP] log_retention_action...")
        log_raw = await _call(session, "log_retention_action", {
            "customer_id":    state["customer_id"],
            "risk_score":     state["risk_score"],
            "offer_text":     retention_offer,
            "contract_type":  contract_type,
            "monthly_charge": monthly_charge,
        })
        log_result = json.loads(log_raw)
        print(f"  CRM log ID: {log_result['log_id']}")

    return {
        "policy":           policy,
        "competitor_intel": comp,
        "retention_offer":  retention_offer,
        "crm_log_id":       log_result["log_id"],
        "crm_logged":       True,
    }