"""Shared pytest setup.

Unit tests (tests/unit) never touch a network or a database: repositories, the
predictor, the retriever, the LLMs and the MCP tools are all faked (tests/fakes.py).
Integration tests (tests/integration) run against a real Postgres + pgvector and are
skipped unless TEST_DATABASE_URL is set. Package roots come from pytest.ini.
"""

import os

# Unit tests must never reach a real database by accident.
os.environ.pop("DATABASE_URL", None)
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("ANALYSIS_LIMIT_PER_HOUR", "1000")
