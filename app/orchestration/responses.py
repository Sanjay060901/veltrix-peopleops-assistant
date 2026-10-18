"""Deterministic answer text for tool results, refusals and clarifications.

Tool answers are templated rather than LLM-written: the numbers come straight from the tools,
the text is identical across runs, and no personal data has to be sent to the LLM.
"""
from __future__ import annotations

from datetime import date

from app.data.repositories import Employee, LeaveRequest

CAPABILITIES = (
    "I can answer questions about Veltrix HR policies, leave rules and holidays that apply to you, "
    "show leave balances and requests you are allowed to see, and apply for or cancel your own annual or sick leave."
)
OFF_TOPIC = "Sorry, I can only help with HR topics at Veltrix. " + CAPABILITIES
MANIPULATION = "I can't help with that request. I follow fixed rules about what I can access and share. " + CAPABILITIES
BULK = (
    "I can't share leave data for many employees at once. I can look up one employee at a time "
    "where you're allowed to see their data, and managers can check which of their direct reports are on leave in a date range."
)
NO_ANSWER = (
    "The HR documents available to you don't cover this question, so I can't answer it reliably. "
    "Please contact the HR team for help."
)
LLM_UNAVAILABLE = "Sorry, I can't process that request right now because the language service is unavailable. Please try again later."
DISCARDED = "OK, I haven't made any changes."


def fmt_date(d: date) -> str:
    return f"{d.strftime('%a')} {d.day} {d.strftime('%b %Y')}"


def fmt_days(value: float) -> str:
    return f"{value:g} day" + ("" if value == 1 else "s")


def who(target_id: str, requester: Employee, target_name: str | None) -> str:
    if target_id == requester.employee_id:
        return "You have"
    label = f"{target_name} ({target_id})" if target_name else target_id
    return f"{label} has"


def balance_answer(target_id: str, requester: Employee, target_name: str | None, balances: dict[str, float]) -> str:
    if not balances:
        return f"I couldn't find any leave balance on record for {'you' if target_id == requester.employee_id else target_id}."
    parts = [f"{fmt_days(v)} of {k} leave" for k, v in balances.items()]
    return f"{who(target_id, requester, target_name)} {' and '.join(parts)} remaining."


def request_line(r: LeaveRequest) -> str:
    span = fmt_date(r.start_date) if r.start_date == r.end_date else f"{fmt_date(r.start_date)} to {fmt_date(r.end_date)}"
    return f"- {r.leave_type.title()} leave, {span}, {fmt_days(r.working_days).replace('day', 'working day')}, {r.status} (request ID {r.request_id})"


def requests_answer(target_id: str, requester: Employee, target_name: str | None, requests: list[LeaveRequest], status: str | None) -> str:
    subject = "You have" if target_id == requester.employee_id else who(target_id, requester, target_name)
    qualifier = f"{status} " if status else ""
    if not requests:
        return f"{subject} no {qualifier}leave requests."
    lines = [f"{subject} {len(requests)} {qualifier}leave request{'s' if len(requests) != 1 else ''}:"]
    lines.extend(request_line(r) for r in requests)
    return "\n".join(lines)


def fmt_span(start: date, end: date) -> str:
    return fmt_date(start) if start == end else f"{fmt_date(start)} to {fmt_date(end)}"


def team_answer(entries: list[tuple[Employee, LeaveRequest]], start: date, end: date) -> str:
    span = fmt_span(start, end)
    if not entries:
        when = f"on {span}" if start == end else f"from {span}"
        return f"None of your direct reports have pending or approved leave {when}."
    lines = [f"Pending or approved leave of your direct reports overlapping {span}:"]
    for emp, r in entries:
        lines.append(
            f"- {emp.full_name} ({emp.employee_id}): {r.leave_type} leave {fmt_date(r.start_date)} to {fmt_date(r.end_date)}, {r.status}"
        )
    return "\n".join(lines)


def apply_proposal(leave_type: str, start: date, end: date, working_days: int, balance: float) -> str:
    span = fmt_date(start) if start == end else f"{fmt_date(start)} to {fmt_date(end)}"
    return (
        f"Here is your leave request:\n- Type: {leave_type}\n- Dates: {span}\n- Working days: {working_days}\n"
        f"Your {leave_type} balance would go from {balance:g} to {balance - working_days:g} days. "
        "Shall I submit it? Reply \"yes\" to confirm."
    )


def apply_done(leave_type: str, start: date, end: date, working_days: int, request_id: str, new_balance: float) -> str:
    span = fmt_date(start) if start == end else f"{fmt_date(start)} to {fmt_date(end)}"
    return (
        f"Done. Your {leave_type} leave for {span} ({working_days} working day{'s' if working_days != 1 else ''}) "
        f"has been submitted with status pending (request ID {request_id}). Your remaining {leave_type} balance is {new_balance:g} days."
    )


def cancel_proposal(r: LeaveRequest) -> str:
    return (
        "You're about to cancel this request:\n"
        f"{request_line(r)}\n"
        f"{fmt_days(r.working_days).replace('day', 'working day')} would be returned to your {r.leave_type} balance. "
        "Shall I cancel it? Reply \"yes\" to confirm."
    )


def cancel_done(r: LeaveRequest, new_balance: float) -> str:
    return (
        f"Done. Your {r.leave_type} leave for {fmt_span(r.start_date, r.end_date)} has been cancelled "
        f"and {fmt_days(r.working_days).replace('day', 'working day')} returned. Your {r.leave_type} balance is now {new_balance:g} days."
    )


def rejected(action: str, reason: str) -> str:
    return f"I can't {action}: {reason}"


def ask_for(missing: list[str], extra: str = "") -> str:
    if len(missing) == 1:
        need = missing[0]
    else:
        need = ", ".join(missing[:-1]) + " and " + missing[-1]
    text = f"To continue, please tell me {need}."
    return f"{extra} {text}".strip() if extra else text
