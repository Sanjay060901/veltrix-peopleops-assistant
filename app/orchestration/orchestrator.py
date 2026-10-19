"""Turn handling: routing, slot filling, the two-step write confirmation and response shaping.

Business rules live here and in the tools, not in prompts. The LLM is used to understand the
message (planner) and to write document answers (rag); every permission, validation and route
decision is made in code.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date

from app.config import Settings
from app.data import repositories as repo
from app.data.db import Database
from app.data.repositories import Employee
from app.ingestion.pipeline import Corpus
from app.llm.base import LLMClient, LLMError, request_deadline
from app.orchestration import responses as R
from app.orchestration.dates import parse_iso, resolve_dates
from app.orchestration.planner import Plan, is_confirmation, looks_like_manipulation, plan_message
from app.orchestration.rag import answer_from_documents
from app.orchestration.sessions import PendingProposal, Session, describe_apply
from app.privacy.pii import redact_for_log, scrub_answer
from app.tools import leave_tools as T
from app.tools.permissions import EMPLOYEE_REFUSAL, PermissionDenied

log = logging.getLogger(__name__)

ROUTES = ("rag", "tool", "hybrid", "clarify", "no_answer", "refuse")
REQUEST_DEADLINE_SECONDS = 27.0  # the spec allows 30 s per request; keep a margin
_EMPLOYEE_ID = re.compile(r"\b([A-Za-z]{1,3}\d{2,})\b")
_SELF_WORDS = {"", "me", "myself", "my", "i", "mine", "self", "my own"}


@dataclass
class ChatResult:
    answer: str
    route: str
    citations: list[tuple[str, str]] = field(default_factory=list)
    tool_calls: list[T.ToolOutcome] = field(default_factory=list)
    pending_confirmation: bool = False


class Clarify(Exception):
    def __init__(self, question: str) -> None:
        super().__init__(question)
        self.question = question


def enforce_route_contract(result: ChatResult) -> ChatResult:
    """Make the response satisfy the route table in TASK §8.1 whatever happened upstream."""
    route = result.route if result.route in ROUTES else "no_answer"
    if route in ("clarify", "no_answer", "refuse"):
        result.citations, result.tool_calls, result.pending_confirmation = [], [], False
    elif route == "rag":
        result.tool_calls = []
        if not result.citations:
            route, result.answer = "no_answer", R.NO_ANSWER
    elif route == "tool":
        result.citations = []
        if not result.tool_calls:
            route = "no_answer"
    elif route == "hybrid":
        if not result.citations and result.tool_calls:
            route = "tool"
        elif result.citations and not result.tool_calls:
            route = "rag"
        elif not result.citations and not result.tool_calls:
            route = "no_answer"
    if route != "tool":
        result.pending_confirmation = False
    result.route = route
    return result


class Orchestrator:
    def __init__(self, settings: Settings, db: Database, corpus: Corpus, llm: LLMClient | None) -> None:
        self.settings = settings
        self.db = db
        self.corpus = corpus
        self.llm = llm

    # --- entry point ---------------------------------------------------------------------

    def handle(self, requester: Employee, session: Session, message: str) -> ChatResult:
        with session.lock, request_deadline(REQUEST_DEADLINE_SECONDS):
            try:
                result = self._handle(requester, session, message)
            except PermissionDenied as exc:
                result = ChatResult(exc.message, "refuse")
            except Clarify as exc:
                result = ChatResult(exc.question, "clarify")
            except LLMError:
                log.warning("llm failure employee=%s", requester.employee_id)
                result = ChatResult(R.LLM_UNAVAILABLE, "no_answer")
            result = enforce_route_contract(result)
            result.answer = scrub_answer(result.answer)
            if not result.pending_confirmation:
                session.pending = None  # at most one proposal, and only right after it was made
            session.history.append(("user", message))
            # History is sent to the planner later, so names from tool answers are removed.
            session.history.append(("assistant", redact_for_log(result.answer)))
            log.info(
                "chat employee=%s route=%s tools=%s citations=%s pending=%s",
                requester.employee_id, result.route, [t.name for t in result.tool_calls],
                [c[0] for c in result.citations], result.pending_confirmation,
            )
            return result

    def _ctx(self, requester: Employee) -> T.ToolContext:
        return T.ToolContext(self.db, self.corpus.docs, requester, self.settings.today(), self.settings.leave_year)

    def _handle(self, requester: Employee, session: Session, message: str) -> ChatResult:
        ctx = self._ctx(requester)
        pending = session.pending

        if pending and is_confirmation(message):
            session.pending = None
            return self._execute(ctx, pending)
        if looks_like_manipulation(message):
            session.pending = None
            return ChatResult(R.MANIPULATION, "refuse")
        if self.llm is None:
            return ChatResult(R.LLM_UNAVAILABLE, "no_answer")

        history = list(session.history)[-2 * self.settings.history_turns :]
        plan = plan_message(self.llm, message, history, pending.summary if pending else "", ctx.today, ctx.leave_year)
        log.info("plan employee=%s intent=%s", requester.employee_id, plan.intent)

        if pending:
            if plan.intent == "confirm":
                session.pending = None
                return self._execute(ctx, pending)
            if plan.intent == "decline":
                session.pending = None
                return ChatResult(R.DISCARDED, "clarify")
            if plan.intent == "modify_pending" and pending.tool == "apply_leave":
                session.pending = None
                return self._apply(ctx, session, plan, defaults=pending.args)
            # Anything else discards the proposal and is handled normally (TASK §8.2).
            session.pending = None

        return self._dispatch(ctx, session, plan, message)

    def _dispatch(self, ctx: T.ToolContext, session: Session, plan: Plan, message: str) -> ChatResult:
        intent = plan.intent
        if intent == "manipulation":
            return ChatResult(R.MANIPULATION, "refuse")
        if intent == "off_topic":
            return ChatResult(R.OFF_TOPIC, "refuse")
        if intent == "bulk_data" or (plan.multiple_employees and intent in ("leave_balance", "leave_requests")):
            return ChatResult(R.BULK, "refuse")
        if intent == "smalltalk":
            return ChatResult("Hello! " + R.CAPABILITIES, "clarify")
        if intent == "unclear":
            return ChatResult(plan.clarification_question or "Could you tell me a bit more about what you need?", "clarify")
        if intent in ("confirm", "decline", "modify_pending"):
            return ChatResult("There is no pending request to confirm or change. " + R.CAPABILITIES, "clarify")
        if intent == "policy_question":
            return self._policy(ctx, plan, message)
        if intent == "leave_balance":
            return self._balance(ctx, plan, message)
        if intent == "leave_requests":
            return self._requests(ctx, plan)
        if intent == "team_leave":
            return self._team(ctx, plan)
        if intent == "apply_leave":
            return self._apply(ctx, session, plan)
        if intent == "cancel_leave":
            return self._cancel(ctx, session, plan)
        return ChatResult("Could you rephrase that?", "clarify")

    # --- documents -----------------------------------------------------------------------

    def _reference_date(self, plan: Plan, today: date) -> date:
        ref = parse_iso(plan.reference_date)
        return ref if ref and ref < today else today

    def _context_lines(self, ctx: T.ToolContext, reference: date) -> list[str]:
        r = ctx.requester
        lines = [f"Today: {ctx.today.isoformat()}", f"Office region: {r.region}", f"Role: {r.role}"]
        if reference != ctx.today:
            lines.append(f"The question is about the rules in force on {reference.isoformat()}.")
        return lines

    def _rag(self, ctx: T.ToolContext, plan: Plan, message: str, tool_lines: list[str], extra_context: list[str]):
        reference = self._reference_date(plan, ctx.today)
        return answer_from_documents(
            self.llm, self.corpus, message, plan.search_query or message, ctx.requester.role, ctx.requester.region,
            ctx.today, reference, self._context_lines(ctx, reference) + extra_context, tool_lines,
            self.settings.corpus_token_budget,
        )

    def _policy(self, ctx: T.ToolContext, plan: Plan, message: str) -> ChatResult:
        rag = self._rag(ctx, plan, message, [], [])
        if rag.status != "answered":
            return ChatResult(R.NO_ANSWER, "no_answer")
        return ChatResult(rag.answer, "rag", citations=rag.citations)

    # --- read tools ----------------------------------------------------------------------

    def resolve_target(self, requester: Employee, text: str) -> str:
        """Map 'me', an employee ID or a name to an employee ID. Name lookup only searches
        employees the requester may see, so a miss and a forbidden employee look the same."""
        cleaned = re.sub(r"'s\b", "", text.strip(), flags=re.I).strip()
        if cleaned.lower() in _SELF_WORDS:
            return requester.employee_id
        match = _EMPLOYEE_ID.search(cleaned)
        if match:
            return match.group(1).upper()
        tokens = [t for t in re.findall(r"[a-z]+", cleaned.lower()) if t not in {"employee", "colleague", "mr", "ms", "mrs", "dr"}]
        if not tokens:
            return requester.employee_id
        with self.db.read() as conn:
            if requester.role == "hr":
                visible = repo.all_employees(conn)
            elif requester.role == "manager":
                visible = [requester] + repo.direct_reports(conn, requester.employee_id)
            else:
                visible = [requester]
        found = [e for e in visible if all(t in e.full_name.lower().split() for t in tokens)]
        if not found:
            raise PermissionDenied(EMPLOYEE_REFUSAL)
        if len(found) > 1:
            options = ", ".join(f"{e.full_name} ({e.employee_id})" for e in found[:6])
            raise Clarify(f"Which employee do you mean: {options}?")
        return found[0].employee_id

    def _target_name(self, ctx: T.ToolContext, target_id: str) -> str | None:
        """Only called after the tool's permission check succeeded for this target."""
        if target_id == ctx.requester.employee_id:
            return None
        with self.db.read() as conn:
            emp = repo.get_employee(conn, target_id)
        return emp.full_name if emp else None

    def _balance(self, ctx: T.ToolContext, plan: Plan, message: str) -> ChatResult:
        target_id = self.resolve_target(ctx.requester, plan.target_employee)
        leave_type = plan.leave_type if plan.leave_type in T.LEAVE_TYPES else None
        outcome = T.get_leave_balance(ctx, target_id, leave_type)
        if outcome.status == "error":
            return ChatResult(outcome.error or "That balance could not be read.", "tool", tool_calls=[outcome])
        name = self._target_name(ctx, target_id)
        balances = outcome.data["balances"]
        text = R.balance_answer(target_id, ctx.requester, name, balances)
        if not plan.needs_policy_documents:
            return ChatResult(text, "tool", tool_calls=[outcome])

        with self.db.read() as conn:
            target = repo.get_employee(conn, target_id)
        tool_lines = [f"Current {k} leave balance on {ctx.today.isoformat()}: {v:g} days" for k, v in balances.items()]
        extra = [f"Employee joined on: {target.join_date}", f"Employee office region: {target.region}"] if target else []
        rag = self._rag(ctx, plan, message, tool_lines, extra)
        if rag.status != "answered":
            return ChatResult(
                text + " The documents available to you don't cover how this balance will change, so I can't project it. "
                "Please contact the HR team for help.",
                "tool",
                tool_calls=[outcome],
            )
        return ChatResult(rag.answer, "hybrid", citations=rag.citations, tool_calls=[outcome])

    def _requests(self, ctx: T.ToolContext, plan: Plan) -> ChatResult:
        target_id = self.resolve_target(ctx.requester, plan.target_employee)
        status = plan.status_filter or None
        outcome = T.list_leave_requests(ctx, target_id, status)
        if outcome.status == "error":
            return ChatResult(outcome.error or "Those requests could not be read.", "tool", tool_calls=[outcome])
        name = self._target_name(ctx, target_id)
        text = R.requests_answer(target_id, ctx.requester, name, outcome.data["requests"], status)
        return ChatResult(text, "tool", tool_calls=[outcome])

    def _team(self, ctx: T.ToolContext, plan: Plan) -> ChatResult:
        # The tool re-checks the role; checking first avoids asking a non-manager for dates.
        if ctx.requester.role != "manager":
            raise PermissionDenied("Team leave is only available to managers for their direct reports.")
        dates = resolve_dates(
            plan.start_date_text, plan.end_date_text, plan.start_date, plan.end_date, ctx.today, ctx.leave_year, allow_past=True
        )
        if dates.problem:
            raise Clarify(R.ask_for(["the date range you want to check"], f"I couldn't use that date: {dates.problem}."))
        if not dates.start or not dates.end:
            raise Clarify(R.ask_for(["the date range you want to check (for example \"next week\")"]))
        outcome = T.get_team_leave(ctx, dates.start, dates.end)
        if outcome.status == "error":
            return ChatResult(outcome.error or "Team leave could not be read.", "tool", tool_calls=[outcome])
        return ChatResult(R.team_answer(outcome.data["entries"], dates.start, dates.end), "tool", tool_calls=[outcome])

    # --- write tools: propose ------------------------------------------------------------

    def _apply(self, ctx: T.ToolContext, session: Session, plan: Plan, defaults: dict | None = None) -> ChatResult:
        defaults = defaults or {}
        leave_type = plan.leave_type or defaults.get("leave_type", "")
        start, end, problem = defaults.get("start"), defaults.get("end"), ""
        start_given = bool(plan.start_date_text or plan.start_date)
        end_given = bool(plan.end_date_text or plan.end_date)
        if start_given:
            dates = resolve_dates(
                plan.start_date_text, plan.end_date_text, plan.start_date, plan.end_date, ctx.today, ctx.leave_year, allow_past=False
            )
            start, end, problem = dates.start, dates.end, dates.problem
        elif end_given:
            # Only the end changed, e.g. "make it until Friday instead": keep the proposed start.
            dates = resolve_dates(plan.end_date_text, "", plan.end_date, "", ctx.today, ctx.leave_year, allow_past=False)
            end, problem = dates.start, dates.problem

        if leave_type == "other":
            label = plan.leave_type_text or "that type of"
            outcome = T.ToolOutcome(
                "apply_leave",
                {"leave_type": label, "start_date": start.isoformat() if start else "", "end_date": end.isoformat() if end else ""},
                "error",
                error=f"Only annual or sick leave can be applied for here, not {label} leave.",
            )
            return ChatResult(R.rejected("submit this request", outcome.error or ""), "tool", tool_calls=[outcome])

        missing: list[str] = []
        if not leave_type:
            missing.append("whether this is annual or sick leave")
        if problem:
            missing.append("the dates you want (they must be today or later)")
        elif not start or not end:
            missing.append("the start and end dates")
        if missing:
            raise Clarify(R.ask_for(missing, f"I couldn't use that date: {problem}." if problem else ""))

        assert start and end
        outcome = T.apply_leave(ctx, leave_type, start, end, execute=False)
        if outcome.status == "error":
            return ChatResult(R.rejected("submit this leave request", outcome.error or ""), "tool", tool_calls=[outcome])
        days = outcome.data["working_days"]
        session.pending = PendingProposal(
            "apply_leave",
            {"leave_type": leave_type, "start": start, "end": end},
            describe_apply(leave_type, start, end, days),
        )
        answer = R.apply_proposal(leave_type, start, end, days, outcome.data["balance"])
        return ChatResult(answer, "tool", tool_calls=[outcome], pending_confirmation=True)

    def _cancel(self, ctx: T.ToolContext, session: Session, plan: Plan) -> ChatResult:
        request_id = plan.request_id
        if not request_id:
            with self.db.read() as conn:
                active = repo.list_requests(conn, ctx.requester.employee_id, T.ACTIVE)
            candidates = active
            if plan.start_date_text or plan.start_date:
                dates = resolve_dates(
                    plan.start_date_text, plan.end_date_text, plan.start_date, plan.end_date, ctx.today, ctx.leave_year, allow_past=True
                )
                if dates.start:
                    end = dates.end or dates.start
                    candidates = [r for r in candidates if r.overlaps(dates.start, end)]
            if plan.leave_type in T.LEAVE_TYPES:
                candidates = [r for r in candidates if r.leave_type == plan.leave_type]
            if len(candidates) > 1:
                # Prefer requests that can still be cancelled when that narrows it down.
                future = [r for r in candidates if r.start_date > ctx.today]
                if len(future) == 1:
                    candidates = future
            if not candidates:
                outcome = T.list_leave_requests(ctx, ctx.requester.employee_id, None)
                listing = R.requests_answer(ctx.requester.employee_id, ctx.requester, None, outcome.data.get("requests", []), None)
                return ChatResult(
                    "I couldn't find a pending or approved request of yours matching that.\n" + listing, "tool", tool_calls=[outcome]
                )
            if len(candidates) > 1:
                lines = "\n".join(R.request_line(r) for r in candidates)
                raise Clarify(f"You have several leave requests that could match. Which one should I cancel?\n{lines}")
            request_id = candidates[0].request_id

        outcome = T.cancel_leave(ctx, request_id, execute=False)
        if outcome.status == "error":
            return ChatResult(R.rejected("cancel that request", outcome.error or ""), "tool", tool_calls=[outcome])
        request = outcome.data["request"]
        session.pending = PendingProposal(
            "cancel_leave",
            {"request_id": request_id},
            f"cancel_leave request {request_id} ({request.leave_type} leave {request.start_date} to {request.end_date})",
        )
        return ChatResult(R.cancel_proposal(request), "tool", tool_calls=[outcome], pending_confirmation=True)

    # --- write tools: execute ------------------------------------------------------------

    def _execute(self, ctx: T.ToolContext, pending: PendingProposal) -> ChatResult:
        if pending.tool == "apply_leave":
            a = pending.args
            outcome = T.apply_leave(ctx, a["leave_type"], a["start"], a["end"], execute=True)
            if outcome.status == "error":
                return ChatResult(R.rejected("submit this leave request", outcome.error or ""), "tool", tool_calls=[outcome])
            d = outcome.data
            return ChatResult(
                R.apply_done(a["leave_type"], a["start"], a["end"], d["working_days"], d["request_id"], d["new_balance"]),
                "tool",
                tool_calls=[outcome],
            )
        outcome = T.cancel_leave(ctx, pending.args["request_id"], execute=True)
        if outcome.status == "error":
            return ChatResult(R.rejected("cancel that request", outcome.error or ""), "tool", tool_calls=[outcome])
        return ChatResult(R.cancel_done(outcome.data["request"], outcome.data["new_balance"]), "tool", tool_calls=[outcome])
