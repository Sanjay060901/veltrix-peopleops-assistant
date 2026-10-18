"""Access-control decisions for leave data. One function per decision, called inside each tool.

Every decision uses the requester loaded from the X-Employee-Id header, never anything the user
typed or the LLM chose.
"""
from __future__ import annotations

import sqlite3

from app.data import repositories as repo
from app.data.repositories import Employee, LeaveRequest

# Same text whether the employee does not exist or is not visible to the requester (TASK §7.3).
EMPLOYEE_REFUSAL = "I can't share information about that employee."


class PermissionDenied(Exception):
    def __init__(self, message: str = EMPLOYEE_REFUSAL) -> None:
        super().__init__(message)
        self.message = message


def can_view_employee_leave(conn: sqlite3.Connection, requester: Employee, target_id: str) -> bool:
    """employee: self only; manager: self and direct reports; hr: any existing employee."""
    if target_id == requester.employee_id:
        return True
    target = repo.get_employee(conn, target_id)
    if target is None:
        return False
    if requester.role == "hr":
        return True
    if requester.role == "manager":
        return target.manager_id == requester.employee_id
    return False


def can_view_team_leave(requester: Employee) -> bool:
    """get_team_leave is for managers only (TASK §6.1)."""
    return requester.role == "manager"


def can_modify_request(requester: Employee, request: LeaveRequest | None) -> bool:
    """apply/cancel act on the requester's own leave only, for every role."""
    return request is not None and request.employee_id == requester.employee_id


def require_view_employee_leave(conn: sqlite3.Connection, requester: Employee, target_id: str) -> None:
    if not can_view_employee_leave(conn, requester, target_id):
        raise PermissionDenied()


def require_team_leave(requester: Employee) -> None:
    if not can_view_team_leave(requester):
        raise PermissionDenied("Team leave is only available to managers for their direct reports.")
