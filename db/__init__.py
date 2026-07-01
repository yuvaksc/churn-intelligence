"""db/ — raw-SQL SQLite data-access layer (stdlib sqlite3, no ORM).

All persistent application data (pre-scored customers, retention
recommendations, key/value meta) lives in a single SQLite file. Each module
holds hand-written, parameterized SQL for one domain. Functions are synchronous
and blocking; async callers wrap them with asyncio.to_thread (the same pattern
the agents use for model inference). The api process is the only writer.
"""
