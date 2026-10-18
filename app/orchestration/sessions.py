"""In-memory sessions: owner, short history, and at most one pending write proposal."""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import date


@dataclass
class PendingProposal:
    tool: str  # apply_leave | cancel_leave
    args: dict  # typed values: leave_type/start/end or request_id
    summary: str  # LLM-safe description (no personal data)


@dataclass
class Session:
    session_id: str
    owner_id: str
    history: deque = field(default_factory=lambda: deque(maxlen=12))  # (role, llm-safe text)
    pending: PendingProposal | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionOwnershipError(Exception):
    pass


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str, employee_id: str) -> Session:
        """A session belongs to the employee who sent its first message (TASK §7.1)."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = Session(session_id, employee_id)
                self._sessions[session_id] = session
            elif session.owner_id != employee_id:
                raise SessionOwnershipError(session_id)
            return session


def describe_apply(leave_type: str, start: date, end: date, working_days: int | None = None) -> str:
    days = f", {working_days} working days" if working_days is not None else ""
    return f"apply_leave {leave_type} leave from {start.isoformat()} to {end.isoformat()}{days}"
