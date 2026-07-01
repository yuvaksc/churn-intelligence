"""
rag/bm25.py — minimal in-memory BM25 (Okapi), zero dependencies.

Built once from a corpus of short documents and queried at retrieval time as the
sparse half of hybrid search. Standard BM25 with non-negative IDF, k1=1.5, b=0.75
and a simple \\w+ lowercase tokenizer (good for the pipe-delimited profile strings
and one-word churn reasons in this project).
"""

import math
import re
from collections import Counter

_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(
        self,
        documents: list[str],
        ids: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if len(documents) != len(ids):
            raise ValueError("documents and ids must be the same length")
        self.ids = ids
        self.k1 = k1
        self.b = b

        self._tokens = [_tokenize(d) for d in documents]
        self._len = [len(t) for t in self._tokens]
        self._avgdl = (sum(self._len) / len(self._len)) if self._len else 0.0
        self._tf = [Counter(t) for t in self._tokens]

        df: Counter = Counter()
        for toks in self._tokens:
            df.update(set(toks))
        n = len(documents)
        # Non-negative BM25 IDF: log(1 + (N - df + 0.5) / (df + 0.5))
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def _score(self, q_tokens: list[str], i: int) -> float:
        tf = self._tf[i]
        dl = self._len[i]
        norm = self.k1 * (1 - self.b + self.b * (dl / self._avgdl if self._avgdl else 0.0))
        score = 0.0
        for term in q_tokens:
            f = tf.get(term, 0)
            if f:
                score += self._idf.get(term, 0.0) * (f * (self.k1 + 1)) / (f + norm)
        return score

    def search(
        self,
        query: str,
        top_k: int = 30,
        allowed_ids: set | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to top_k (id, score) pairs, highest score first. If
        allowed_ids is given, only those ids are scored (mirrors a metadata filter)."""
        q_tokens = _tokenize(query)
        scored: list[tuple[str, float]] = []
        for i, doc_id in enumerate(self.ids):
            if allowed_ids is not None and doc_id not in allowed_ids:
                continue
            s = self._score(q_tokens, i)
            if s > 0:
                scored.append((doc_id, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
