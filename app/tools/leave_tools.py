"""The five leave tools (TASK §6). Names and argument names match the spec exactly.

Each tool performs its own permission check with the requester from the request header.
Write tools validate first; the write itself happens in one transaction that re-validates,
because data can change between proposal and confirmation.
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from app.data import repositories as repo
from app.data.db import Database
from app.data.repositories import Employee, LeaveRequest
from app.ingestion.models import DocMeta
from app.tools import permissions as perm
from app.tools.working_days import choose_calendar, count_working_days

LEAVE_TYPES = ("annual", "sick")
STATUSES = ("pending", "approved", "cancelled")
ACTIVE = ("pending", "approved")


@dataclass
class ToolContext:
    db: Database
    docs: dict[str, DocMeta]
    requester: Employee
    today: date
    leave_year: int


@dataclass
class ToolOutcome:
    name: str
    args: dict[str, Any]
    status: str  # success | error | pending_confirmation
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class LeaveRuleError(Exception):
    """A specific, user-facing reason why a leave action is rejected."""


def _fmt(d: date) -> str:
    return f"{d.strftime('%a')} {d.day} {d.strftime('%b %Y')}"


# --- read tools ------------------------------------------------------------------------------

def get_leave_balance(ctx: ToolContext, employee_id: str, leave_type: str | None = None) -> ToolOutcome:
    args: dict[str, Any] = {"employee_id": employee_id}
    if leave_type:
        args["leave_type"] = leave_type
    with ctx.db.read() as conn:
        perm.require_view_employee_leave(conn, ctx.requester, employee_id)
        if leave_type and leave_type not in LEAVE_TYPES:
            return ToolOutcome("get_leave_balance", args, "error", error=f"Unknown leave type '{leave_type}'. Leave types are annual and sick.")
        balances = repo.get_balances(conn, employee_id, leave_type)
    return ToolOutcome("get_leave_balance", args, "success", {"employee_id": employee_id, "balances": balances})


def list_leave_requests(ctx: ToolContext, employee_id: str, status: str | None = None) -> ToolOutcome:
    args: dict[str, Any] = {"employee_id": employee_id}
    if status:
        args["status"] = status
    with ctx.db.read() as conn:
        perm.require_view_employee_leave(conn, ctx.requester, employee_id)
        if status and status not in STATUSES:
            return ToolOutcome("list_leave_requests", args, "error", error=f"Unknown status '{status}'. Use pending, approved or cancelled.")
        requests = repo.list_requests(conn, employee_id, [status] if status else None)
    return ToolOutcome("list_leave_requests", args, "success", {"employee_id": employee_id, "requests": requests})


def get_team_leave(ctx: ToolContext, start_date: date, end_date: date) -> ToolOutcome:
    args = {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
    perm.require_team_leave(ctx.requester)
    if end_date < start_date:
        return ToolOutcome("get_team_leave", args, "error", error="The end date is before the start date.")
    with ctx.db.read() as conn:
        reports = {e.employee_id: e for e in repo.direct_reports(conn, ctx.requester.employee_id)}
        requests = repo.team_requests_overlapping(conn, ctx.requester.employee_id, start_date, end_date)
    entries = [(reports[r.employee_id], r) for r in requests if r.employee_id in reports]
    return ToolOutcome("get_team_leave", args, "success", {"entries": entries, "team_size": len(reports)})


# --- validation shared by proposal and execution --------------------------------------------

def compute_working_days(conn: sqlite3.Connection, ctx: ToolContext, start: date, end: date) -> tuple[int, tuple[str, ...]]:
    choice = choose_calendar(conn, ctx.docs.values(), ctx.requester.role, ctx.requester.region, start.year, start)
    return count_working_days(start, end, choice.holidays), choice.doc_ids


def validate_apply(conn: sqlite3.Connection, ctx: ToolContext, leave_type: str, start: date, end: date) -> dict[str, Any]:
    if leave_type not in LEAVE_TYPES:
        raise LeaveRuleError(f"'{leave_type}' leave can't be requested here. Only annual or sick leave can be applied for.")
    if start.year != ctx.leave_year or end.year != ctx.leave_year:
        raise LeaveRuleError(f"Leave can only be requested for dates in {ctx.leave_year}.")
    if start < ctx.today:
        raise LeaveRuleError(f"The start date {_fmt(start)} is in the past. Leave must start today ({_fmt(ctx.today)}) or later.")
    if end < start:
        raise LeaveRuleError(f"The end date {_fmt(end)} is before the start date {_fmt(start)}.")
    working_days, calendars = compute_working_days(conn, ctx, start, end)
    if working_days == 0:
        raise LeaveRuleError(
            f"{_fmt(start)} to {_fmt(end)} contains no working days (only weekends or public holidays), so there is nothing to request."
        )
    balance = repo.get_balances(conn, ctx.requester.employee_id, leave_type).get(leave_type)
    if balance is None:
        raise LeaveRuleError(f"You have no {leave_type} leave balance on record.")
    if working_days > balance:
        raise LeaveRuleError(
            f"This request needs {working_days} working days of {leave_type} leave, but your balance is {balance:g} days."
        )
    for existing in repo.list_requests(conn, ctx.requester.employee_id, ACTIVE):
        if existing.overlaps(start, end):
            raise LeaveRuleError(
                f"It overlaps your existing {existing.status} {existing.leave_type} leave from "
                f"{_fmt(existing.start_date)} to {_fmt(existing.end_date)}."
            )
    return {"working_days": working_days, "balance": balance, "calendars": calendars}


def validate_cancel(conn: sqlite3.Connection, ctx: ToolContext, request_id: str) -> LeaveRequest:
    request = repo.get_request(conn, request_id)
    if not perm.can_modify_request(ctx.requester, request):
        # Same message for "does not exist" and "belongs to someone else".
        raise LeaveRuleError("I couldn't find that leave request among your own requests.")
    assert request is not None
    if request.status not in ACTIVE:
        raise LeaveRuleError(f"That request is already {request.status}.")
    if request.start_date <= ctx.today:
        raise LeaveRuleError(
            f"That leave started on {_fmt(request.start_date)}. Only requests that start after today can be cancelled."
        )
    return request


# --- write tools -----------------------------------------------------------------------------

def apply_leave(ctx: ToolContext, leave_type: str, start_date: date, end_date: date, execute: bool) -> ToolOutcome:
    """execute=False validates and returns a proposal; execute=True writes in one transaction."""
    args = {"leave_type": leave_type, "start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
    try:
        if not execute:
            with ctx.db.read() as conn:
                info = validate_apply(conn, ctx, leave_type, start_date, end_date)
            return ToolOutcome("apply_leave", args, "pending_confirmation", info)
        with ctx.db.transaction() as conn:
            info = validate_apply(conn, ctx, leave_type, start_date, end_date)
            request = LeaveRequest(
                request_id=str(uuid.uuid4()),
                employee_id=ctx.requester.employee_id,
                leave_type=leave_type,
                start_date=start_date,
                end_date=end_date,
                working_days=info["working_days"],
                status="pending",
                created_at=datetime.now().replace(microsecond=0).isoformat(),
            )
            repo.insert_request(conn, request)
            repo.adjust_balance(conn, ctx.requester.employee_id, leave_type, -info["working_days"])
            new_balance = repo.get_balances(conn, ctx.requester.employee_id, leave_type)[leave_type]
        return ToolOutcome("apply_leave", args, "success", {**info, "request_id": request.request_id, "new_balance": new_balance})
    except LeaveRuleError as exc:
        return ToolOutcome("apply_leave", args, "error", error=str(exc))


def cancel_leave(ctx: ToolContext, request_id: str, execute: bool) -> ToolOutcome:
    args = {"request_id": request_id}
    try:
        if not execute:
            with ctx.db.read() as conn:
                request = validate_cancel(conn, ctx, request_id)
            return ToolOutcome("cancel_leave", args, "pending_confirmation", {"request": request})
        with ctx.db.transaction() as conn:
            request = validate_cancel(conn, ctx, request_id)
            repo.set_request_status(conn, request_id, "cancelled")
            repo.adjust_balance(conn, ctx.requester.employee_id, request.leave_type, request.working_days)
            new_balance = repo.get_balances(conn, ctx.requester.employee_id, request.leave_type)[request.leave_type]
        return ToolOutcome("cancel_leave", args, "success", {"request": request, "new_balance": new_balance})
    except LeaveRuleError as exc:
        return ToolOutcome("cancel_leave", args, "error", error=str(exc))
