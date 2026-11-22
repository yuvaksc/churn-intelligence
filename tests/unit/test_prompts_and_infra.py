"""Prompt structure, rate limiter, CRM ticket rules — small pure units."""

import pytest

from api.ratelimit import SlidingWindowLimiter, client_key
from churn_agents import prompts
from churn_data.crm import parse_customer_id, tickets_for

SYSTEM_PROMPTS = {
    "diagnostic": prompts.DIAGNOSTIC_SYSTEM,
    "evidence": prompts.EVIDENCE_SYSTEM,
    "supervisor": prompts.SUFFICIENCY_SYSTEM,
    "mitigation": prompts.MITIGATION_SYSTEM,
}


@pytest.mark.parametrize("name", SYSTEM_PROMPTS)
def test_every_system_prompt_has_the_standard_sections(name):
    text = SYSTEM_PROMPTS[name]
    for section in ("ROLE", "TASK", "STEPS", "CONSTRAINTS", "OUTPUT FORMAT"):
        assert f"{section}\n" in text, f"{name} prompt is missing {section}"


def test_agent_prompts_keep_reasoning_private_and_include_an_example():
    for text in (prompts.DIAGNOSTIC_SYSTEM, prompts.EVIDENCE_SYSTEM, prompts.MITIGATION_SYSTEM):
        assert "output ONLY the final answer" in text
        assert "EXAMPLE (illustrative only" in text


def test_user_templates_format_with_run_data():
    msgs = prompts.messages(prompts.DIAGNOSTIC_SYSTEM, prompts.DIAGNOSTIC_USER.format(
        risk_score=0.72, threshold=0.13, customer_line="2 months", shap_table="  Contract ▲"))
    assert msgs[0].type == "system" and msgs[1].type == "human"
    assert "72.0%" in msgs[1].content and "13.0%" in msgs[1].content


def test_rate_limiter_window():
    now = [0.0]
    lim = SlidingWindowLimiter(2, window_seconds=60, clock=lambda: now[0])
    assert lim.allow("a") and lim.allow("a") and not lim.allow("a")
    assert lim.allow("b")                      # per-client
    now[0] = 61
    assert lim.allow("a")                      # window slid
    assert SlidingWindowLimiter(0).allow("x")  # 0 disables


def test_client_key_prefers_forwarded_for():
    class Peer:
        host = "10.0.0.1"
    assert client_key({"x-forwarded-for": "1.2.3.4, 5.6.7.8"}, Peer()) == "1.2.3.4"
    assert client_key({}, Peer()) == "10.0.0.1"


def test_crm_tickets_are_deterministic_and_capped():
    feats = {"Contract": "Month-to-month", "Internet Service": "Fiber optic", "Tenure Months": 2,
             "Monthly Charges": 95.0, "Tech Support": "No", "Payment Method": "Electronic check"}
    first = tickets_for(7, feats)
    assert first == tickets_for(7, feats) and len(first) == 3
    assert first[0][1].startswith("Billing concern")
    assert tickets_for(7, {"Contract": "Two year", "Tenure Months": 60}) == []


def test_parse_customer_id():
    assert parse_customer_id("TEST-4521") == 4521
    assert parse_customer_id(17) == 17
    assert parse_customer_id("abc") is None
