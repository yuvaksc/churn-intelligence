"""guardrails/grounding.py — heuristic check that the offer is grounded in the policy."""

import re

_STOP = {"the", "a", "an", "of", "for", "to", "at", "on", "or", "and", "with",
         "same", "one", "free", "your", "you", "month", "service", "price"}


def _keywords(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3 and w not in _STOP}


def check_grounding(offer_text: str, policy: dict) -> dict:
    """Does the offer reference at least one of the policy's allowed offer types?

    A cheap citation/grounding signal for the live path; the deeper LLM-judged
    faithfulness/groundedness scores live in the offline eval (eval/run_eval.py).
    """
    allowed = (policy or {}).get("allowed_offers") or []
    if not offer_text or not allowed:
        return {"ok": True, "matched": []}
    offer_kw = _keywords(offer_text)
    matched = [a for a in allowed if _keywords(a) & offer_kw]
    return {"ok": bool(matched), "matched": matched}
