"""guardrails/pii.py — mask obvious PII (emails, phone numbers) in generated text."""

import re

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")


def mask_pii(text: str) -> tuple[str, int]:
    """Replace emails / phone numbers with [REDACTED]. Returns (masked, count)."""
    if not text:
        return text, 0
    count = 0

    def _sub(_m):
        nonlocal count
        count += 1
        return "[REDACTED]"

    masked = _EMAIL.sub(_sub, text)
    masked = _PHONE.sub(_sub, masked)
    return masked, count
