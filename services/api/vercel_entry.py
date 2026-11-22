"""Vercel entrypoint for the FastAPI service (see vercel.json → services.api).

The API imports the shared libraries in packages/* and spawns the MCP server
(services/mcp) over stdio, so the Vercel service is rooted at the repository and this
module puts those package roots on sys.path before importing the app.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for rel in ("packages/data", "packages/ml", "packages/rag", "packages/agents",
            "packages/evals", "services/api", "services/mcp"):
    p = str(_ROOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)

from api.main import app  # noqa: E402,F401
