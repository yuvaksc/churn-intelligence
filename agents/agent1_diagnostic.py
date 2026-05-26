"""
agents/agent1_diagnostic.py — Diagnostic Lead (Agent 1).  [async]

Changes from sync version:
  - async def agent1_node
  - predict_customer() wrapped in asyncio.to_thread (blocking: joblib + XGBoost + SHAP)
  - llm.invoke() → await llm.ainvoke()
"""

import asyncio

from agents.state import WarRoomState
from agents.tools.predict_tool import predict_customer
from agents.llm_config import get_analytical_llm


_RISK_SUMMARY_PROMPT = """\
You are a senior churn analyst reviewing a model output for a retention specialist.

Customer churn probability: {risk_score:.1%}  (threshold for action: {threshold:.1%})

Top SHAP risk drivers (what is PUSHING this score up or down):
{shap_table}

Write a concise Risk Summary of 2–3 sentences:
  • State the exact probability and whether it triggers intervention
  • Name the single largest driver and its direction
  • Note if a second driver compounds or contradicts the first
  • If Charge Ratio is the top driver and Tenure Months ≤ 3, note the customer
    is in the highest-risk new-customer window

Rules:
  - Be specific with numbers, not vague
  - Do NOT recommend any retention action
  - Write for a non-technical retention specialist
  - No bullet points — flowing sentences only
"""


def _format_shap_table(shap_drivers: list) -> str:
    lines = []
    for d in shap_drivers:
        arrow = "▲ raises risk" if d["shap_value"] > 0 else "▼ lowers risk"
        lines.append(
            f"  {d['feature']:<35} {arrow}  "
            f"(SHAP={d['shap_value']:+.4f},  value={d['raw_value']})"
        )
    return "\n".join(lines)


async def agent1_node(state: WarRoomState) -> dict:
    print("\n" + "─" * 52)
    print("  [Agent 1 — Diagnostic Lead]")
    print("─" * 52)

    # Blocking: joblib loads + XGBoost inference + SHAP → run in thread pool
    result       = await asyncio.to_thread(predict_customer, state["customer_raw"])
    risk_score   = result["risk_score"]
    risk_label   = result["risk_label"]
    shap_drivers = result["shap_drivers"]
    threshold    = result["threshold"]

    print(f"  Risk score:  {risk_score:.1%}  →  {risk_label}")
    print(f"  Threshold:   {threshold:.1%}")
    print(f"  Top driver:  {shap_drivers[0]['feature']}  "
          f"({shap_drivers[0]['direction']},  SHAP={shap_drivers[0]['shap_value']:+.4f})")

    if risk_label == "LOW":
        risk_summary = (
            f"Customer scores {risk_score:.1%} churn probability, "
            f"below the {threshold:.1%} intervention threshold. "
            f"No retention action required at this time."
        )
    else:
        prompt   = _RISK_SUMMARY_PROMPT.format(
            risk_score=risk_score,
            threshold=threshold,
            shap_table=_format_shap_table(shap_drivers),
        )
        response     = await get_analytical_llm().ainvoke(prompt)
        risk_summary = response.content.strip()

    print(f"  Summary:     {risk_summary[:100]}...")

    return {
        "risk_score":   risk_score,
        "risk_label":   risk_label,
        "shap_drivers": shap_drivers,
        "risk_summary": risk_summary,
    }