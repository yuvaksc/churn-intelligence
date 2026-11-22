"""Evaluators (no network: LangSmith and the LLM judges are not called)."""

import pytest

from churn_evals import evaluators as ev
from churn_evals import live

QRELS = {"a": 2, "b": 1, "c": 1}


def test_recall_uses_reachable_denominator():
    assert ev.recall_at_k(["a", "x", "b"], QRELS, 3) == pytest.approx(2 / 3)
    assert ev.recall_at_k(["a"], {"a": 1, **{f"r{i}": 1 for i in range(50)}}, 1) == 1.0


def test_precision_and_mrr():
    assert ev.precision_at_k(["x", "a", "y"], QRELS, 3) == pytest.approx(1 / 3)
    assert ev.mrr(["x", "y", "b"], QRELS) == pytest.approx(1 / 3)
    assert ev.mrr(["x"], QRELS) == 0.0


def test_ndcg_rewards_graded_order():
    perfect = ev.ndcg_at_k(["a", "b", "c"], QRELS, 3)
    assert perfect == pytest.approx(1.0) and ev.ndcg_at_k(["b", "a", "c"], QRELS, 3) < perfect
    assert ev.ndcg_at_k(["x", "y"], QRELS, 2) == 0.0


def test_retrieval_evaluator_reports_every_metric():
    scores = {r["key"]: r["score"] for r in
              ev.retrieval_metrics({"retrieved_ids": ["a", "b", "x"], "qrels": QRELS})["results"]}
    assert scores["mrr"] == 1.0
    assert {"recall@5", "precision@5", "ndcg@5", "ndcg@10"} <= scores.keys()


HIGH_RUN = {"risk_label": "HIGH", "agent_path": ["agent1", "agent2", "agent3"],
            "retention_offer": "HOOK: ...", "tools_used": ["retention_policy_check", "get_open_tickets"],
            "guardrails": {"passed": True, "policy_ceiling": {"ok": True}, "grounding": {"ok": True}},
            "evidence_report": "x [train_1]",
            "evidence_citations": {"ok": True, "unknown": [], "citation_accuracy": 1.0}}
LOW_RUN = {"risk_label": "LOW", "agent_path": ["agent1"], "retention_offer": ""}


def test_warroom_checks():
    assert ev.routing(HIGH_RUN)["score"] == 1
    assert ev.routing(LOW_RUN)["score"] == 1
    assert ev.routing({**LOW_RUN, "agent_path": ["agent1", "agent2"]})["score"] == 0
    assert ev.policy_before_offer(HIGH_RUN)["score"] == 1
    assert ev.policy_before_offer({**HIGH_RUN, "tools_used": ["get_open_tickets"]})["score"] == 0
    assert ev.guardrails(HIGH_RUN)["score"] == 1
    assert ev.citations_ok(HIGH_RUN)["score"] == 1
    assert ev.citation_accuracy(HIGH_RUN)["score"] == 1.0


def test_offer_checks_do_not_apply_to_low_risk():
    for evaluator in (ev.policy_before_offer, ev.guardrails, ev.citations_ok):
        assert evaluator(LOW_RUN)["score"] is None


def test_no_experiments_without_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert live.log_experiments("trace", 1, HIGH_RUN) == []
