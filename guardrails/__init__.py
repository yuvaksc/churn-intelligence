"""
guardrails/ — backend guardrails applied around the war-room I/O.

Input:  prompt-injection / jailbreak heuristic on free-text fields.
Output: PII masking, policy-ceiling verification, grounding/citation check.

This is an internal analyst tool, so nothing is hard-blocked — violations are
flagged and attached to the response + the audit record (and the offer is
PII-masked before it is stored or returned).
"""

from guardrails.pii import mask_pii
from guardrails.policy import check_policy_ceiling
from guardrails.grounding import check_grounding
from guardrails.injection import detect_injection


def scan_input(customer_state: str, customer_raw: dict | None) -> dict:
    """Input guardrail — scan free-text fields for prompt-injection patterns."""
    parts = [str(customer_state or "")]
    parts += [str(v) for v in (customer_raw or {}).values() if isinstance(v, str)]
    return detect_injection(" ".join(parts))


def apply_output_guardrails(offer_text: str, policy: dict) -> tuple[str, dict]:
    """Mask PII + verify policy ceiling + grounding. Returns (safe_offer, report)."""
    masked, pii_count = mask_pii(offer_text or "")
    policy_check = check_policy_ceiling(masked, policy)
    grounding    = check_grounding(masked, policy)
    report = {
        "pii_masked":     pii_count,
        "policy_ceiling": policy_check,
        "grounding":      grounding,
        "passed":         bool(policy_check["ok"] and grounding["ok"]),
    }
    return masked, report
