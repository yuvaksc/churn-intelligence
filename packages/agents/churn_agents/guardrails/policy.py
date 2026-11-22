"""guardrails/policy.py — verify a drafted offer respects the retention policy ceiling."""

import re

_PCT = re.compile(r"(\d{1,3})\s*%")


def check_policy_ceiling(offer_text: str, policy: dict) -> dict:
    """Flag any discount percentage in the offer that exceeds policy.max_discount_pct.

    Percentages are the reliable signal (offers read like '10% discount'); raw
    dollar amounts are noisy (they also appear as the monthly charge), so they are
    intentionally not parsed here — the dollar cap is enforced upstream by the
    policy the LLM was given.
    """
    max_pct = (policy or {}).get("max_discount_pct")
    if not offer_text or max_pct is None:
        return {"ok": True, "max_pct": max_pct, "violations": []}
    violations = [
        f"{int(p)}% exceeds {int(max_pct)}% ceiling"
        for p in _PCT.findall(offer_text)
        if int(p) > int(max_pct)
    ]
    return {"ok": not violations, "max_pct": max_pct, "violations": violations}
