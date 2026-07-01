"""db/connection.py — synchronous SQLite access (stdlib sqlite3, raw SQL, no ORM).

Functions here are blocking; async callers wrap them with asyncio.to_thread —
the same pattern the agents use for model inference and the API uses at startup.
One short-lived connection per call; WAL mode + a busy timeout suit the
single-writer (api) / occasional-reader workload.

DB location comes from APP_DB_PATH (default: data/app.db). The parent directory
is created on demand so a clean checkout works with no setup.
"""

import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(os.getenv("APP_DB_PATH", "data/app.db"))
_SCHEMA = Path(__file__).parent / "schema.sql"


@contextmanager
def connect():
    """Yield a sqlite3 connection with rows accessible by column name."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create tables from schema.sql (idempotent) and enable WAL.
    Blocking — call via `await asyncio.to_thread(init_db)`."""
    schema = _SCHEMA.read_text(encoding="utf-8")
    with connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(schema)
        conn.commit()
