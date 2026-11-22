"""
churn_agents/agent3_mitigator.py — Mitigation Architect (Agent 3).  [async, ReAct]

Instead of a hardcoded tool sequence, agent3 runs a bounded ReAct loop: it
discovers the MCP server's tools at runtime (list_tool_specs → MCP Capability
Exchange), binds them to the Groq LLM, and the model decides which tools to call
(policy, competitor intel, account history, open tickets, prior recommendations)
before drafting the offer. Reliability backstops guarantee the policy fetch and
the CRM log so persistence + the frontend contract stay intact.
"""

import json

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import trace

from churn_agents.state import WarRoomState
from churn_agents import prompts
from churn_agents.deps import WarRoomDeps

# Token budget: Groq's free tier allows ~8k tokens/minute, and every ReAct round resends the
# whole conversation. So: few rounds, all tools requested at once, compact schemas + results.
_MAX_TOOL_ITERS = 3
_MAX_RESULT_CHARS = 700
_MAX_CONTEXT_CHARS = {"risk_summary": 500, "evidence_report": 900}
# The CRM log is always written by the backstop below, so the LLM never needs to spend a
# round on it.
_HIDDEN_FROM_LLM = {"log_retention_action"}

def _user_context(state: WarRoomState) -> str:
    c = state["customer_raw"]
    return prompts.MITIGATION_USER.format(
        customer_id=state["customer_id"],
        customer_state=state.get("customer_state", "DEFAULT"),
        contract=c.get("Contract"),
        monthly_charge=float(c.get("Monthly Charges", 0) or 0),
        tenure=c.get("Tenure Months"),
        internet=c.get("Internet Service"),
        risk_summary=_clip(state.get("risk_summary", ""), _MAX_CONTEXT_CHARS["risk_summary"]),
        evidence_report=_clip(state.get("evidence_report", ""), _MAX_CONTEXT_CHARS["evidence_report"]),
    )


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def _compact_spec(spec: dict) -> dict:
    """First sentence of the tool description only; the JSON schema carries the rest."""
    fn = dict(spec["function"])
    desc = " ".join((fn.get("description") or "").split())
    fn["description"] = _clip(desc.split(". ")[0], 160)
    return {**spec, "function": fn}


def _compact_result(text: str) -> str:
    """Re-serialise JSON without whitespace and cap the size fed back to the LLM."""
    try:
        text = json.dumps(json.loads(text), separators=(",", ":"), ensure_ascii=False)
    except Exception:
        pass
    return text if len(text) <= _MAX_RESULT_CHARS else text[:_MAX_RESULT_CHARS] + "…(truncated)"


async def _call_tool(gateway, name: str, args: dict) -> str:
    """One MCP tool call, recorded as a `tool` span named after the tool in the trace."""
    with trace(name, run_type="tool", inputs=args) as span:
        result = await gateway.call(name, args)
        span.end(outputs={"output": result})
    return result


def _maybe_json(text: str) -> dict | None:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


async def agent3_node(state: WarRoomState, deps: WarRoomDeps) -> dict:
    print("\n" + "─" * 52)
    print("  [Agent 3 — Mitigation Architect]  (ReAct)")
    print("─" * 52)

    customer_raw   = state["customer_raw"]
    contract_type  = str(customer_raw.get("Contract", "Month-to-month"))
    monthly_charge = float(customer_raw.get("Monthly Charges", 0.0) or 0.0)

    messages = [SystemMessage(content=prompts.MITIGATION_SYSTEM), HumanMessage(content=_user_context(state))]
    policy: dict = {}
    competitor_intel: dict = {}
    tools_used: list[str] = []
    retention_offer = ""

    async with deps.tools() as gateway:
        specs = [_compact_spec(s) for s in await gateway.list_specs()
                 if s["function"]["name"] not in _HIDDEN_FROM_LLM]
        print(f"  [MCP] {len(specs)} tools offered: {[s['function']['name'] for s in specs]}")
        llm = deps.creative_llm().bind_tools(specs)
        cache: dict[str, str] = {}   # identical repeat calls are answered without a round trip

        for _ in range(_MAX_TOOL_ITERS):
            ai = await llm.ainvoke(messages)
            messages.append(ai)

            if not ai.tool_calls:
                retention_offer = (ai.content or "").strip()
                break

            for tc in ai.tool_calls:
                name = tc["name"]
                args = tc.get("args", {}) or {}
                key = name + json.dumps(args, sort_keys=True, default=str)
                if key in cache:
                    messages.append(ToolMessage(content="(same result as before)", tool_call_id=tc["id"]))
                    continue
                tools_used.append(name)
                print(f"  [MCP] LLM → {name}({args})")
                result = await _call_tool(gateway, name, args)

                parsed = _maybe_json(result)
                if name == "retention_policy_check" and parsed:
                    policy = parsed
                elif name == "get_competitor_insights" and parsed:
                    competitor_intel = parsed

                result = _compact_result(result)
                cache[key] = result
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        else:
            # Iteration cap reached — force a final draft with whatever context we have
            ai = await llm.ainvoke(
                messages + [HumanMessage(content=prompts.MITIGATION_FINAL_NUDGE)]
            )
            retention_offer = (ai.content or "").strip()

        # Backstop 1: policy must be populated (frontend display + Phase-6 guardrail rely on it)
        if not policy:
            print("  [MCP] policy not fetched by LLM → backstop retention_policy_check")
            policy = _maybe_json(await _call_tool(gateway, "retention_policy_check", {
                "contract_type":  contract_type,
                "monthly_charge": monthly_charge,
            })) or {}

        # Backstop 2: always log the action (mints crm_log_id; api persists the recommendation)
        log_result = _maybe_json(await _call_tool(gateway, "log_retention_action", {
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
