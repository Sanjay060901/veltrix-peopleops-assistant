"""Credentials (added DB table), login sessions and brute-force throttling for AUTH_MODE=login."""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime

from app.auth.passwords import burn_time, hash_password, verify_password


# --- credentials -----------------------------------------------------------------------------------

def set_password(conn: sqlite3.Connection, employee_id: str, password: str) -> None:
    now = datetime.now().replace(microsecond=0).isoformat()
    conn.execute(
        "INSERT INTO auth_credentials (employee_id, password_hash, updated_at) VALUES (?, ?, ?)"
        " ON CONFLICT(employee_id) DO UPDATE SET password_hash = excluded.password_hash, updated_at = excluded.updated_at",
        (employee_id, hash_password(password), now),
    )


def remove_password(conn: sqlite3.Connection, employee_id: str) -> bool:
    return conn.execute("DELETE FROM auth_credentials WHERE employee_id = ?", (employee_id,)).rowcount == 1


def check_credentials(conn: sqlite3.Connection, employee_id: str, password: str) -> bool:
    """True only for an existing employee with a matching password. Same cost either way."""
    row = conn.execute(
        "SELECT c.password_hash FROM auth_credentials c JOIN employees e ON e.employee_id = c.employee_id"
        " WHERE c.employee_id = ?",
        (employee_id,),
    ).fetchone()
    if row is None:
        burn_time(password)
        return False
    return verify_password(password, row["password_hash"])


def credential_version(conn: sqlite3.Connection, employee_id: str) -> str | None:
    """Changes whenever the password is reset or removed; sessions remember the version they
    were created with, so a reset (even from the admin CLI process) ends existing sessions."""
    row = conn.execute("SELECT password_hash FROM auth_credentials WHERE employee_id = ?", (employee_id,)).fetchone()
    return hashlib.sha256(row["password_hash"].encode()).hexdigest()[:24] if row else None


def employees_with_credentials(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT employee_id FROM auth_credentials ORDER BY employee_id")]


# --- login sessions --------------------------------------------------------------------------------

@dataclass
class LoginSession:
    employee_id: str
    credential_version: str
    created: float
    last_seen: float


class LoginSessions:
    """Server-side sessions. The cookie holds a random token; only its SHA-256 is kept here."""

    def __init__(self, idle_seconds: int, absolute_seconds: int) -> None:
        self.idle = idle_seconds
        self.absolute = absolute_seconds
        self._sessions: dict[str, LoginSession] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def create(self, employee_id: str, credential_version: str) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._lock:
            # One active login per employee: a new sign-in ends older ones.
            for key in [k for k, s in self._sessions.items() if s.employee_id == employee_id]:
                del self._sessions[key]
            self._sessions[self._key(token)] = LoginSession(employee_id, credential_version, now, now)
        return token

    def resolve(self, token: str | None) -> LoginSession | None:
        if not token:
            return None
        key = self._key(token)
        now = time.time()
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                return None
            if now - session.last_seen > self.idle or now - session.created > self.absolute:
                del self._sessions[key]
                return None
            session.last_seen = now
            return session

    def end(self, token: str | None) -> None:
        if token:
            with self._lock:
                self._sessions.pop(self._key(token), None)

    def end_all_for(self, employee_id: str) -> None:
        with self._lock:
            for key in [k for k, s in self._sessions.items() if s.employee_id == employee_id]:
                del self._sessions[key]


# --- throttling ------------------------------------------------------------------------------------

class LoginThrottle:
    """Locks an employee ID, and separately a client address, after repeated failures."""

    def __init__(self, max_failures: int, window_seconds: int, lockout_seconds: int) -> None:
        self.max_failures = max_failures
        self.window = window_seconds
        self.lockout = lockout_seconds
        self._failures: dict[str, list[float]] = {}
        self._locked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def locked(self, *keys: str) -> bool:
        now = time.time()
        with self._lock:
            return any(self._locked_until.get(k, 0) > now for k in keys)

    def failure(self, *keys: str) -> None:
        now = time.time()
        with self._lock:
            for key in keys:
                recent = [t for t in self._failures.get(key, []) if now - t < self.window] + [now]
                self._failures[key] = recent
                if len(recent) >= self.max_failures:
                    self._locked_until[key] = now + self.lockout
                    self._failures[key] = []

    def success(self, *keys: str) -> None:
        with self._lock:
            for key in keys:
                self._failures.pop(key, None)
