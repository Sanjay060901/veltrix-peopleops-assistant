"""apply_leave / cancel_leave rules (TASK §6.2, §6.3) and read-tool access control per role."""
from datetime import date

import pytest

from app.data import repositories as repo
from app.tools import leave_tools as T
from app.tools.permissions import EMPLOYEE_REFUSAL, PermissionDenied

TODAY = date(2026, 9, 15)


def ctx_for(db, corpus, employee_id, today=TODAY):
    with db.read() as conn:
        requester = repo.get_employee(conn, employee_id)
    return T.ToolContext(db, corpus.docs, requester, today, 2026)


def balance(db, employee_id, leave_type):
    with db.read() as conn:
        return repo.get_balances(conn, employee_id, leave_type)[leave_type]


# --- apply_leave -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "leave_type,start,end,reason",
    [
        ("parental", date(2026, 9, 22), date(2026, 9, 22), "Only annual or sick"),
        ("annual", date(2026, 12, 31), date(2027, 1, 4), "dates in 2026"),
        ("annual", date(2026, 9, 14), date(2026, 9, 16), "in the past"),
        ("annual", date(2026, 9, 24), date(2026, 9, 22), "before the start date"),
        ("annual", date(2026, 9, 19), date(2026, 9, 20), "no working days"),  # a weekend
        ("annual", date(2026, 10, 3), date(2026, 10, 4), "no working days"),  # Sat holiday + Sun
        ("annual", date(2026, 11, 2), date(2026, 11, 6), "balance is 4"),  # 5 days > 4
        ("annual", date(2026, 10, 6), date(2026, 10, 7), "overlaps"),  # existing pending request
    ],
)
def test_apply_rejections(db, corpus, leave_type, start, end, reason):
    ctx = ctx_for(db, corpus, "X201")
    before = balance(db, "X201", "annual")
    outcome = T.apply_leave(ctx, leave_type, start, end, execute=True)
    assert outcome.status == "error" and reason in outcome.error
    assert balance(db, "X201", "annual") == before


def test_holiday_reduces_working_days(db, corpus):
    india = ctx_for(db, corpus, "X201")
    germany = ctx_for(db, corpus, "X301")
    # Fri 18 .. Mon 21 Sep spans a weekend; the Monday is a holiday in IN only.
    outcome = T.apply_leave(india, "annual", date(2026, 9, 18), date(2026, 9, 21), execute=False)
    assert outcome.status == "pending_confirmation" and outcome.data["working_days"] == 1
    assert T.apply_leave(germany, "annual", date(2026, 9, 18), date(2026, 9, 21), execute=False).data["working_days"] == 2
    # Tue 22 Sep is a holiday in DE only; the optional IN holiday on Wed 23 still counts as a working day.
    assert T.apply_leave(germany, "annual", date(2026, 9, 22), date(2026, 9, 23), execute=False).data["working_days"] == 1
    assert T.apply_leave(india, "annual", date(2026, 9, 22), date(2026, 9, 23), execute=False).data["working_days"] == 2


def test_proposal_changes_nothing(db, corpus):
    ctx = ctx_for(db, corpus, "X201")
    with db.read() as conn:
        before = len(repo.list_requests(conn, "X201"))
    T.apply_leave(ctx, "annual", date(2026, 9, 16), date(2026, 9, 17), execute=False)
    with db.read() as conn:
        assert len(repo.list_requests(conn, "X201")) == before
    assert balance(db, "X201", "annual") == 4.0


def test_apply_success_creates_pending_request_and_subtracts(db, corpus):
    ctx = ctx_for(db, corpus, "X201")
    outcome = T.apply_leave(ctx, "annual", date(2026, 9, 16), date(2026, 9, 17), execute=True)
    assert outcome.status == "success"
    with db.read() as conn:
        created = repo.get_request(conn, outcome.data["request_id"])
    assert created.status == "pending" and created.working_days == 2 and created.employee_id == "X201"
    assert len(created.request_id) == 36  # uuid4
    assert balance(db, "X201", "annual") == 2.0


def test_start_today_is_allowed(db, corpus):
    ctx = ctx_for(db, corpus, "X202")
    assert T.apply_leave(ctx, "sick", TODAY, TODAY, execute=False).status == "pending_confirmation"


def test_apply_is_one_transaction(db, corpus, monkeypatch):
    ctx = ctx_for(db, corpus, "X201")

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(repo, "adjust_balance", boom)
    with pytest.raises(RuntimeError):
        T.apply_leave(ctx, "annual", date(2026, 9, 16), date(2026, 9, 17), execute=True)
    with db.read() as conn:
        assert not [r for r in repo.list_requests(conn, "X201") if r.start_date == date(2026, 9, 16)]
    assert balance(db, "X201", "annual") == 4.0


# --- cancel_leave ----------------------------------------------------------------------------

def test_cancel_success_returns_days(db, corpus):
    ctx = ctx_for(db, corpus, "X201")
    outcome = T.cancel_leave(ctx, "req-future", execute=True)
    assert outcome.status == "success"
    with db.read() as conn:
        assert repo.get_request(conn, "req-future").status == "cancelled"
    assert balance(db, "X201", "annual") == 6.0


@pytest.mark.parametrize(
    "employee,request_id,reason",
    [
        ("X201", "req-past", "started on"),
        ("X201", "req-cancelled", "already cancelled"),
        ("X201", "req-other", "couldn't find"),  # someone else's request
        ("X201", "does-not-exist", "couldn't find"),
        ("X200", "req-future", "couldn't find"),  # managers can't cancel for reports
    ],
)
def test_cancel_rejections(db, corpus, employee, request_id, reason):
    outcome = T.cancel_leave(ctx_for(db, corpus, employee), request_id, execute=True)
    assert outcome.status == "error" and reason in outcome.error


def test_cancel_request_starting_today_is_rejected(db, corpus):
    ctx = ctx_for(db, corpus, "X201", today=date(2026, 10, 5))
    assert T.cancel_leave(ctx, "req-future", execute=False).status == "error"


# --- read tools: access per role ---------------------------------------------------------------

def test_employee_sees_only_self(db, corpus):
    ctx = ctx_for(db, corpus, "X201")
    assert T.get_leave_balance(ctx, "X201").data["balances"]["annual"] == 4.0
    for other in ("X202", "X200", "NOPE1"):
        with pytest.raises(PermissionDenied) as exc:
            T.get_leave_balance(ctx, other)
        assert exc.value.message == EMPLOYEE_REFUSAL
        with pytest.raises(PermissionDenied):
            T.list_leave_requests(ctx, other)


def test_manager_sees_direct_reports_only(db, corpus):
    ctx = ctx_for(db, corpus, "X200")
    assert T.get_leave_balance(ctx, "X202").status == "success"
    assert T.list_leave_requests(ctx, "X201", "pending").data["requests"][0].request_id == "req-future"
    with pytest.raises(PermissionDenied) as forbidden:
        T.get_leave_balance(ctx, "X301")  # another manager's report
    with pytest.raises(PermissionDenied) as missing:
        T.get_leave_balance(ctx, "X999")
    assert forbidden.value.message == missing.value.message == EMPLOYEE_REFUSAL


def test_hr_sees_any_single_employee(db, corpus):
    ctx = ctx_for(db, corpus, "X100")
    assert T.get_leave_balance(ctx, "X301").status == "success"
    with pytest.raises(PermissionDenied) as missing:
        T.get_leave_balance(ctx, "X999")
    assert missing.value.message == EMPLOYEE_REFUSAL


def test_team_leave_managers_only(db, corpus):
    mgr = ctx_for(db, corpus, "X200")
    outcome = T.get_team_leave(mgr, date(2026, 9, 21), date(2026, 10, 9))
    ids = sorted(r.request_id for _, r in outcome.data["entries"])
    assert ids == ["req-future", "req-other"]  # cancelled and other teams excluded
    for role_holder in ("X201", "X100"):
        with pytest.raises(PermissionDenied):
            T.get_team_leave(ctx_for(db, corpus, role_holder), date(2026, 9, 21), date(2026, 10, 9))
