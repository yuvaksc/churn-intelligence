"""guardrails/citations.py — every citation in an evidence report must point at a
document that was actually retrieved for this run.

Citations are written as [doc_id] (e.g. [train_1234]). Reports:
  cited        ids the text cites
  unknown      cited ids that were NOT retrieved (fabricated or stale) → fails
  uncited      the report makes claims but cites nothing → fails, unless it says
               evidence is insufficient (the allowed refusal)
"""

import re

_CITE = re.compile(r"\[([A-Za-z0-9_:\-.]+)\]")
_REFUSAL = "insufficient historical evidence"


def check_citations(text: str, retrieved_ids: list[str] | set[str]) -> dict:
    retrieved = set(retrieved_ids)
    cited = list(dict.fromkeys(_CITE.findall(text or "")))
    unknown = [c for c in cited if c not in retrieved]
    refusal = _REFUSAL in (text or "").lower()
    ok = not unknown and (bool(cited) or refusal or not (text or "").strip())
    accuracy = (len(cited) - len(unknown)) / len(cited) if cited else (1.0 if refusal else 0.0)
    return {"ok": ok, "cited": cited, "unknown": unknown, "refusal": refusal,
            "citation_accuracy": round(accuracy, 4)}
