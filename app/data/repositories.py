"""Repositories over the provided tables plus the added holiday_calendar table."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Employee:
    employee_id: str
    full_name: str
    email: str
    phone: str
    region: str
    join_date: str
    level: str
    role: str
    manager_id: str | None


@dataclass(frozen=True)
class LeaveRequest:
    request_id: str
    employee_id: str
    leave_type: str
    start_date: date
    end_date: date
    working_days: int
    status: str
    created_at: str

    def overlaps(self, start: date, end: date) -> bool:
        return self.start_date <= end and start <= self.end_date


def _employee(row: sqlite3.Row) -> Employee:
    return Employee(**{k: row[k] for k in row.keys()})


def _request(row: sqlite3.Row) -> LeaveRequest:
    return LeaveRequest(
        request_id=row["request_id"],
        employee_id=row["employee_id"],
        leave_type=row["leave_type"],
        start_date=date.fromisoformat(row["start_date"]),
        end_date=date.fromisoformat(row["end_date"]),
        working_days=int(row["working_days"]),
        status=row["status"],
        created_at=row["created_at"],
    )


# --- employees -----------------------------------------------------------------------------

def get_employee(conn: sqlite3.Connection, employee_id: str) -> Employee | None:
    row = conn.execute("SELECT * FROM employees WHERE employee_id = ?", (employee_id,)).fetchone()
    return _employee(row) if row else None


def direct_reports(conn: sqlite3.Connection, manager_id: str) -> list[Employee]:
    rows = conn.execute(
        "SELECT * FROM employees WHERE manager_id = ? ORDER BY employee_id", (manager_id,)
    ).fetchall()
    return [_employee(r) for r in rows]


def all_employees(conn: sqlite3.Connection) -> list[Employee]:
    return [_employee(r) for r in conn.execute("SELECT * FROM employees ORDER BY employee_id")]


# --- balances ------------------------------------------------------------------------------

def get_balances(conn: sqlite3.Connection, employee_id: str, leave_type: str | None = None) -> dict[str, float]:
    sql = "SELECT leave_type, balance_days FROM leave_balances WHERE employee_id = ?"
    params: list = [employee_id]
    if leave_type:
        sql += " AND leave_type = ?"
        params.append(leave_type)
    sql += " ORDER BY leave_type"
    return {r["leave_type"]: float(r["balance_days"]) for r in conn.execute(sql, params)}


def adjust_balance(conn: sqlite3.Connection, employee_id: str, leave_type: str, delta: float) -> None:
    cur = conn.execute(
        "UPDATE leave_balances SET balance_days = balance_days + ? WHERE employee_id = ? AND leave_type = ?",
        (delta, employee_id, leave_type),
    )
    if cur.rowcount != 1:
        raise LookupError("no balance row to update")


# --- requests ------------------------------------------------------------------------------

def list_requests(
    conn: sqlite3.Connection, employee_id: str, statuses: Sequence[str] | None = None
) -> list[LeaveRequest]:
    sql = "SELECT * FROM leave_requests WHERE employee_id = ?"
    params: list = [employee_id]
    if statuses:
        sql += f" AND status IN ({','.join('?' * len(statuses))})"
        params.extend(statuses)
    sql += " ORDER BY start_date, request_id"
    return [_request(r) for r in conn.execute(sql, params)]


def get_request(conn: sqlite3.Connection, request_id: str) -> LeaveRequest | None:
    row = conn.execute("SELECT * FROM leave_requests WHERE request_id = ?", (request_id,)).fetchone()
    return _request(row) if row else None


def insert_request(conn: sqlite3.Connection, req: LeaveRequest) -> None:
    conn.execute(
        "INSERT INTO leave_requests (request_id, employee_id, leave_type, start_date, end_date,"
        " working_days, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            req.request_id, req.employee_id, req.leave_type, req.start_date.isoformat(),
            req.end_date.isoformat(), req.working_days, req.status, req.created_at,
        ),
    )


def set_request_status(conn: sqlite3.Connection, request_id: str, status: str) -> None:
    conn.execute("UPDATE leave_requests SET status = ? WHERE request_id = ?", (status, request_id))


def team_requests_overlapping(
    conn: sqlite3.Connection, manager_id: str, start: date, end: date
) -> list[LeaveRequest]:
    rows = conn.execute(
        "SELECT r.* FROM leave_requests r JOIN employees e ON e.employee_id = r.employee_id"
        " WHERE e.manager_id = ? AND r.status IN ('pending', 'approved')"
        " AND r.start_date <= ? AND r.end_date >= ? ORDER BY r.start_date, r.employee_id",
        (manager_id, end.isoformat(), start.isoformat()),
    ).fetchall()
    return [_request(r) for r in rows]


# --- holidays (added table) ----------------------------------------------------------------

def replace_holidays(conn: sqlite3.Connection, records: Iterable[tuple[str, str, str, str]]) -> None:
    conn.execute("DELETE FROM holiday_calendar")
    conn.executemany(
        "INSERT OR IGNORE INTO holiday_calendar (doc_id, holiday_date, name, kind) VALUES (?, ?, ?, ?)",
        list(records),
    )


def ensure_holidays(conn: sqlite3.Connection, records: Iterable[tuple[str, str, str, str]]) -> bool:
    """Recreate and refill the holiday table if it is missing (e.g. the database file was replaced
    while the service was running). Returns True when it had to be rebuilt."""
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'holiday_calendar'").fetchone()
    if exists:
        return False
    from app.data.db import ADDED_SCHEMA

    for statement in ADDED_SCHEMA:
        conn.execute(statement)
    replace_holidays(conn, records)
    return True


def holidays_for_docs(conn: sqlite3.Connection, doc_ids: Sequence[str], kind: str = "closure") -> list[tuple[str, date, str]]:
    if not doc_ids:
        return []
    rows = conn.execute(
        f"SELECT doc_id, holiday_date, name FROM holiday_calendar WHERE kind = ?"
        f" AND doc_id IN ({','.join('?' * len(doc_ids))}) ORDER BY holiday_date",
        [kind, *doc_ids],
    ).fetchall()
    return [(r["doc_id"], date.fromisoformat(r["holiday_date"]), r["name"]) for r in rows]
