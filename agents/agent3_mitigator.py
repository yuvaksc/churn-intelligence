"""
agents/agent3_mitigator.py — Mitigation Architect (Agent 3).  [async, ReAct]

Instead of a hardcoded tool sequence, agent3 runs a bounded ReAct loop: it
discovers the MCP server's tools at runtime (list_tool_specs → MCP Capability
Exchange), binds them to the Groq LLM, and the model decides which tools to call
(policy, competitor intel, account history, open tickets, prior recommendations)
before drafting the offer. Reliability backstops guarantee the policy fetch and
the CRM log so persistence + the frontend contract stay intact.
"""

import json

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from agents.state import WarRoomState
from agents.llm_config import get_creative_llm
from agents.tools.mcp_tools import mcp_session, _call, list_tool_specs

_MAX_TOOL_ITERS = 6

_SYSTEM_PROMPT = """You are a senior retention specialist drafting a personalised offer for a high-risk telecom customer.

You have tools available (discovered at runtime). Use them as needed to gather context:
  - retention_policy_check(contract_type, monthly_charge) — the discount ceiling. REQUIRED before any offer.
  - get_competitor_insights(state, internet_service) — what competitors are offering.
  - get_account_history(customer_id) — the customer's account snapshot.
  - get_open_tickets(customer_id) — unresolved support tickets (pain points).
  - get_prior_recommendations(customer_id) — offers already made (do NOT repeat a prior offer).

Decide which tools you actually need — you do not have to call them all. When you have enough context,
STOP calling tools and output the final offer in exactly three labelled parts:

HOOK:    (1 sentence) acknowledge the customer's tenure or situation.
OFFER:   (2 sentences) a specific offer within the policy discount ceiling, countering the competitor implicitly.
URGENCY: (1 sentence) a time-bound call to action.

Rules: never exceed the policy discount ceiling; write in second person; do not name the competitor directly;
keep the offer under 120 words."""


def _user_context(state: WarRoomState) -> str:
    c = state["customer_raw"]
    return (
        f"Customer ID: {state['customer_id']}\n"
        f"US state: {state.get('customer_state', 'DEFAULT')}\n"
        f"Contract: {c.get('Contract')}\n"
        f"Monthly charge: ${float(c.get('Monthly Charges', 0) or 0):.2f}\n"
        f"Tenure: {c.get('Tenure Months')} months\n"
        f"Internet service: {c.get('Internet Service')}\n\n"
        f"RISK SUMMARY:\n{state.get('risk_summary', '')}\n\n"
        f"EVIDENCE REPORT:\n{state.get('evidence_report', '')}\n\n"
        "Draft the retention offer. Call the tools you need, then output HOOK / OFFER / URGENCY."
    )


def _maybe_json(text: str) -> dict | None:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


async def agent3_node(state: WarRoomState) -> dict:
    print("\n" + "─" * 52)
    print("  [Agent 3 — Mitigation Architect]  (ReAct)")
    print("─" * 52)

    customer_raw   = state["customer_raw"]
    contract_type  = str(customer_raw.get("Contract", "Month-to-month"))
    monthly_charge = float(customer_raw.get("Monthly Charges", 0.0) or 0.0)

    messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=_user_context(state))]
    policy: dict = {}
    competitor_intel: dict = {}
    tools_used: list[str] = []
    retention_offer = ""

    async with mcp_session() as session:
        specs = await list_tool_specs(session)
        print(f"  [MCP] discovered {len(specs)} tools: {[s['function']['name'] for s in specs]}")
        llm = get_creative_llm().bind_tools(specs)

        for _ in range(_MAX_TOOL_ITERS):
            ai = await llm.ainvoke(messages)
            messages.append(ai)

            if not ai.tool_calls:
                retention_offer = (ai.content or "").strip()
                break

            for tc in ai.tool_calls:
                name = tc["name"]
                args = tc.get("args", {}) or {}
                tools_used.append(name)
                print(f"  [MCP] LLM → {name}({args})")
                result = await _call(session, name, args)

                parsed = _maybe_json(result)
                if name == "retention_policy_check" and parsed:
                    policy = parsed
                elif name == "get_competitor_insights" and parsed:
                    competitor_intel = parsed

                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        else:
            # Iteration cap reached — force a final draft with whatever context we have
            ai = await llm.ainvoke(
                messages + [HumanMessage(content="Stop calling tools. Write the final HOOK / OFFER / URGENCY offer now.")]
            )
            retention_offer = (ai.content or "").strip()

        # Backstop 1: policy must be populated (frontend display + Phase-6 guardrail rely on it)
        if not policy:
            print("  [MCP] policy not fetched by LLM → backstop retention_policy_check")
            policy = _maybe_json(await _call(session, "retention_policy_check", {
                "contract_type":  contract_type,
                "monthly_charge": monthly_charge,
            })) or {}

        # Backstop 2: always log the action (mints crm_log_id; api persists the recommendation)
        log_result = _maybe_json(await _call(session, "log_retention_action", {
            "customer_id":    state["customer_id"],
            "risk_score":     state["risk_score"],
            "offer_text":     retention_offer,
            "contract_type":  contract_type,
            "monthly_charge": monthly_charge,
        })) or {}
        crm_log_id = log_result.get("log_id", "")

    print(f"  Offer drafted ({len(retention_offer)} chars) · tools used: {tools_used}")

    return {
        "policy":           policy,
        "competitor_intel": competitor_intel,
        "retention_offer":  retention_offer,
        "crm_log_id":       crm_log_id,
        "crm_logged":       bool(crm_log_id),
        "tools_used":       tools_used,
    }
