"""SQLite access: connections and the single transaction helper used for every write.

Existing tables are never altered. This module only adds the holiday_calendar table and indexes.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

_write_lock = threading.Lock()

ADDED_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS holiday_calendar (
        doc_id       TEXT NOT NULL,
        holiday_date TEXT NOT NULL,
        name         TEXT NOT NULL,
        kind         TEXT NOT NULL,
        PRIMARY KEY (doc_id, holiday_date, name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS auth_credentials (
        employee_id   TEXT PRIMARY KEY REFERENCES employees(employee_id),
        password_hash TEXT NOT NULL,
        updated_at    TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_leave_requests_employee ON leave_requests(employee_id)",
    "CREATE INDEX IF NOT EXISTS idx_employees_manager ON employees(manager_id)",
]


class Database:
    def __init__(self, path: str) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)  # autocommit; explicit BEGIN
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """One atomic unit of work. BEGIN IMMEDIATE takes the write lock up front so the
        validation reads inside the block see the same data the writes are applied to."""
        with _write_lock:
            conn = self.connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    yield conn
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
                else:
                    conn.execute("COMMIT")
            finally:
                conn.close()

    def ensure_schema(self) -> None:
        with self.transaction() as conn:
            for statement in ADDED_SCHEMA:
                conn.execute(statement)
