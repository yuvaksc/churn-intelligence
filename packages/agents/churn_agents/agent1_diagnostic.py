"""
churn_agents/agent1_diagnostic.py — Diagnostic Lead (Agent 1).  [async]

Scores the customer with the calibrated champion model (via the Predictor interface),
reads the top SHAP drivers precomputed by the batch job, and — for HIGH-risk customers
only — asks the LLM for a plain-language risk summary. LOW-risk customers get a
deterministic summary and no LLM call.
"""

import asyncio

from langsmith import traceable

from churn_agents.deps import WarRoomDeps
from churn_agents import prompts
from churn_agents.state import WarRoomState


@traceable(name="churn_model.predict", run_type="chain",
           process_inputs=lambda i: {"customer": i.get("customer_raw")})
def _predict(predictor, customer_raw: dict) -> dict:
    return predictor.predict(customer_raw)


def _format_shap_table(shap_drivers: list) -> str:
    lines = []
    for d in shap_drivers:
        arrow = "▲ raises risk" if d["shap_value"] > 0 else "▼ lowers risk"
        lines.append(
            f"  {d['feature']:<35} {arrow}  "
            f"(SHAP={d['shap_value']:+.4f},  value={d['raw_value']})"
        )
    return "\n".join(lines)


async def agent1_node(state: WarRoomState, deps: WarRoomDeps) -> dict:
    print("\n" + "─" * 52)
    print("  [Agent 1 — Diagnostic Lead]")
    print("─" * 52)

    # Blocking: model load + inference → run in thread pool (its own span in the trace)
    result       = await asyncio.to_thread(_predict, deps.predictor, state["customer_raw"])
    risk_score   = result["risk_score"]
    risk_label   = result["risk_label"]
    # The calibrated pipeline serves probabilities only; SHAP drivers come from the
    # batch job (stored on the customer row and passed in via state).
    shap_drivers = result["shap_drivers"] or state.get("shap_drivers") or []
    threshold    = result["threshold"]

    print(f"  Risk score:  {risk_score:.1%}  →  {risk_label}")
    print(f"  Threshold:   {threshold:.1%}")
    if shap_drivers:
        print(f"  Top driver:  {shap_drivers[0]['feature']}  "
              f"({shap_drivers[0]['direction']},  SHAP={shap_drivers[0]['shap_value']:+.4f})")

    if risk_label == "LOW":
        risk_summary = (
            f"Customer scores {risk_score:.1%} churn probability, "
            f"below the {threshold:.1%} intervention threshold. "
            f"No retention action required at this time."
        )
    else:
        c = state["customer_raw"]
        user = prompts.DIAGNOSTIC_USER.format(
            risk_score=risk_score,
            threshold=threshold,
            customer_line=(f"{c.get('Tenure Months')} months tenure, {c.get('Contract')} contract, "
                           f"{c.get('Internet Service')} internet, ${float(c.get('Monthly Charges', 0) or 0):.2f}/month"),
            shap_table=_format_shap_table(shap_drivers) or "  (no drivers available)",
        )
        response     = await deps.analytical_llm().ainvoke(prompts.messages(prompts.DIAGNOSTIC_SYSTEM, user))
        risk_summary = response.content.strip()

    print(f"  Summary:     {risk_summary[:100]}...")

    return {
        "risk_score":   risk_score,
        "risk_label":   risk_label,
        "shap_drivers": shap_drivers,
        "risk_summary": risk_summary,
    }