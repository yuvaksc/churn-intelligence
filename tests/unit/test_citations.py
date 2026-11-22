"""Citation guardrail: evidence must cite only documents retrieved for this run."""

from churn_agents.guardrails.citations import check_citations

IDS = ["train_1", "train_2"]


def test_valid_citations_pass():
    r = check_citations("Pattern is price [train_1], echoed by [train_2].", IDS)
    assert r["ok"] and r["cited"] == ["train_1", "train_2"] and r["citation_accuracy"] == 1.0


def test_fabricated_id_fails():
    r = check_citations("Customers leave over price [train_9].", IDS)
    assert not r["ok"] and r["unknown"] == ["train_9"] and r["citation_accuracy"] == 0.0


def test_claims_without_citations_fail():
    assert not check_citations("Most similar customers left over price.", IDS)["ok"]


def test_explicit_refusal_is_allowed():
    assert check_citations("Insufficient historical evidence to establish a pattern.", IDS)["ok"]
