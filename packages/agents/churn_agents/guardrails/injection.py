"""guardrails/injection.py — lightweight prompt-injection / jailbreak heuristic."""

import re

_PATTERNS = [
    r"ignore (?:all |the )?(?:previous|prior|above) instructions",
    r"disregard (?:all |the )?(?:previous|prior|above)",
    r"you are now",
    r"system prompt",
    r"reveal (?:your |the )?(?:prompt|instructions|system)",
    r"\bjailbreak\b",
    r"do anything now",
]
_RE = [re.compile(p, re.I) for p in _PATTERNS]


def detect_injection(text: str) -> dict:
    """Flag free-text that looks like a prompt-injection attempt."""
    if not text:
        return {"ok": True, "matches": []}
    matches = [r.pattern for r in _RE if r.search(text)]
    return {"ok": not matches, "matches": matches}
