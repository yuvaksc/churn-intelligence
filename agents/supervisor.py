"""
agents/supervisor.py — LangGraph routing logic.

route_after_diagnostic() is a pure Python conditional edge function.
No LLM call — routing is deterministic based on the threshold from threshold.pkl.

Flow:
    agent1 → [route_after_diagnostic] → agent2   (if HIGH risk)
                                      → END       (if LOW risk)
    agent2 → agent3 → END             (always sequential)
"""

import joblib
from pathlib import Path
from functools import lru_cache
from langgraph.graph import END

from agents.state import WarRoomState


@lru_cache(maxsize=1)
def _threshold() -> float:
    cfg = joblib.load(Path("models/threshold.pkl"))
    return cfg["best_threshold"]


def route_after_diagnostic(state: WarRoomState) -> str:
    """
    Reads risk_score from state and routes:
      ≥ threshold  →  "agent2"  (escalate to full war room)
      < threshold  →  END       (log as low risk, no action)
    """
    score     = state.get("risk_score", 0.0)
    threshold = _threshold()

    if score >= threshold:
        print(f"\n  [Supervisor] {score:.1%} ≥ {threshold:.1%} → escalating to Agent 2")
        return "agent2"
    else:
        print(f"\n  [Supervisor] {score:.1%} < {threshold:.1%} → LOW RISK, no action")
        return END