"""Shared pytest setup.

Points the SQLite layer at an isolated temp DB *before* db.connection is
imported, and puts the repo root on sys.path.
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Must be set before importing db.connection (it binds the path at import time).
_DB = Path(tempfile.gettempdir()) / "churn_pytest.db"
os.environ["APP_DB_PATH"] = str(_DB)

import pytest                              # noqa: E402
from db.connection import init_db          # noqa: E402


@pytest.fixture()
def db():
    """Fresh, initialized SQLite schema for a single test."""
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(_DB) + suffix)
        if p.exists():
            p.unlink()
    init_db()
    yield
