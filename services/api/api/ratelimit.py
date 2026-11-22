"""api/ratelimit.py — per-client sliding-window limit for expensive endpoints.

The public demo has no login, so without a limit anyone could run war-room analyses
(each one spends LLM tokens) in a loop. This is an in-memory limiter: good enough for a
single process or a demo deployment. On serverless each instance keeps its own window,
so treat it as a guard against accidents, not a hard quota; a shared store (e.g. Redis)
would be the production version.

    ANALYSIS_LIMIT_PER_HOUR   default 20   (0 disables)
"""

import os
import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float = 3600.0, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window_seconds, clock
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now, hits = self.clock(), self._hits[key]
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True

    def retry_after(self, key: str) -> int:
        hits = self._hits.get(key)
        if not hits:
            return 0
        return max(0, int(self.window - (self.clock() - hits[0])) + 1)


analysis_limiter = SlidingWindowLimiter(int(os.getenv("ANALYSIS_LIMIT_PER_HOUR", "20")))


def client_key(headers, client) -> str:
    """Caller identity: first X-Forwarded-For hop (set by Vercel's proxy), else the socket peer."""
    forwarded = headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return client.host if client else "unknown"
