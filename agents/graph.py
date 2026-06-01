"""
agents/graph.py — Assembles the War Room StateGraph.

Topology:
    [agent1] → route_after_diagnostic → [agent2] → [agent3] → END
                                     ↘
                                       END  (low risk customers)

Compiled once at import time and reused across all run_agents.py calls.
"""

from langgraph.graph import StateGraph, END

from agents.state import WarRoomState
from agents.agent1_diagnostic import agent1_node
from agents.agent2_researcher import agent2_node
from agents.agent3_mitigator import agent3_node
from agents.supervisor import route_after_diagnostic


def build_graph():
    workflow = StateGraph(WarRoomState)

    # Register nodes
    workflow.add_node("agent1", agent1_node)
    workflow.add_node("agent2", agent2_node)
    workflow.add_node("agent3", agent3_node)

    # Entry point
    workflow.set_entry_point("agent1")

    # Conditional: agent1 → agent2 or END (supervisor gate)
    workflow.add_conditional_edges(
        "agent1",
        route_after_diagnostic,
        {"agent2": "agent2", END: END},
    )

    # Sequential: agent2 → agent3 → END
    workflow.add_edge("agent2", "agent3")
    workflow.add_edge("agent3", END)

    return workflow.compile()


# Singleton — import this in run_agents.py
war_room_graph = build_graph()