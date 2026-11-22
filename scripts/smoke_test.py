"""Post-deploy smoke test.

    python scripts/smoke_test.py https://your-app.vercel.app

Checks: /health reports ok with seeded data and an indexed RAG corpus; the customer
list and metrics load; the war-room WebSocket accepts a connection and starts
streaming. Exits non-zero on any failure.
"""

import asyncio
import json
import sys

import httpx


def main(base: str) -> int:
    base = base.rstrip("/")
    if not base.startswith("http"):
        base = "https://" + base
    failures = []

    def check(name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
        if not ok:
            failures.append(name)

    with httpx.Client(timeout=60, follow_redirects=True) as c:
        h = c.get(f"{base}/health").json()
        check("health ok", h.get("status") == "ok", str(h.get("status")))
        check("customers seeded", h.get("test_set_size", 0) > 0, str(h.get("test_set_size")))
        check("rag indexed", h.get("rag_documents", 0) > 0, str(h.get("rag_documents")))
        r = c.get(f"{base}/api/customers?limit=1&risk_only=true")
        check("customer list", r.status_code == 200 and len(r.json()) == 1, str(r.status_code))
        check("metrics", c.get(f"{base}/api/metrics").status_code == 200)
        if r.status_code == 200 and r.json():
            check("websocket streams", asyncio.run(_ws(base, r.json()[0]["customer_id"])))
    return 1 if failures else 0


async def _ws(base: str, cid: int) -> bool:
    import websockets
    url = base.replace("http", "ws", 1) + f"/api/analyze/{cid}/ws"
    try:
        async with websockets.connect(url, open_timeout=60) as ws:
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
            return first.get("type") in ("agent_start", "token", "agent_complete")
    except Exception as exc:
        print(f"     websocket error: {exc}")
        return False


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"))
