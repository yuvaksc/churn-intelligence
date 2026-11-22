"""Run the API locally for development, with the MCP tool server alongside it.

    docker compose up -d db              # Postgres + pgvector on localhost:5433 (once)
    python scripts/dev_api.py            # API on http://localhost:8000

Defaults (any variable already set in your environment wins):
    DATABASE_URL   postgresql://postgres:postgres@localhost:5433/churn   (the compose db)
    MCP            started ONCE in the background over SSE on :8001 and reused by every
                   analysis (instead of a ~9 s stdio spawn per run); stopped on exit.
                   Set MCP_TRANSPORT=stdio for the per-run behaviour serverless uses.
"""

import atexit
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# Agents log with box-drawing characters; a Windows console that isn't UTF-8 would raise
# UnicodeEncodeError mid-analysis. PYTHONIOENCODING only affects child processes.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
PACKAGE_ROOTS = [str(ROOT / rel) for rel in (
    "packages/data", "packages/ml", "packages/rag", "packages/agents",
    "packages/evals", "services/api", "services/mcp")]
sys.path[:0] = PACKAGE_ROOTS
os.environ["PYTHONPATH"] = os.pathsep.join(PACKAGE_ROOTS)

# Optional overrides when the default ports are taken:  --port 8200 --mcp-port 8201
_args = sys.argv[1:]
for flag, var in (("--port", "PORT"), ("--mcp-port", "MCP_PORT")):
    if flag in _args and _args.index(flag) + 1 < len(_args):
        os.environ[var] = _args[_args.index(flag) + 1]

MCP_PORT = os.environ.get("MCP_PORT", "8001")
for key, value in {
    "DATABASE_URL": "postgresql://postgres:postgres@localhost:5433/churn",
    "MCP_TRANSPORT": "sse",
    "MCP_SERVER_URL": f"http://127.0.0.1:{MCP_PORT}/sse",
    "PYTHONIOENCODING": "utf-8",
    "ANALYSIS_LIMIT_PER_HOUR": "0",                      # no rate limit locally
    "WARM_MODELS": "1",                                  # load RAG models at startup
    "PORT": "8000",
}.items():
    os.environ.setdefault(key, value)


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_mcp_server() -> None:
    """Start the MCP server over SSE unless one is already listening."""
    port = int(MCP_PORT)
    if _port_open(port):
        print(f"MCP server already running on :{port} (reusing it)")
        return
    env = {**os.environ, "MCP_TRANSPORT": "sse", "MCP_HOST": "127.0.0.1", "MCP_PORT": str(port)}
    log = open(ROOT / "mcp_server.log", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "mcp_server.server"], env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    atexit.register(lambda: (proc.terminate(), log.close()))
    for _ in range(180):                      # up to ~90 s for imports + startup (slow laptops)
        if proc.poll() is not None:
            raise SystemExit(f"MCP server exited (code {proc.returncode}); see mcp_server.log")
        if _port_open(port):
            print(f"MCP server ready on :{port} (SSE, pid {proc.pid})")
            return
        time.sleep(0.5)
    raise SystemExit("MCP server did not start in 90 s; see mcp_server.log")


if __name__ == "__main__":
    if os.environ["MCP_TRANSPORT"] == "sse":
        start_mcp_server()

    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=int(os.environ["PORT"]))
