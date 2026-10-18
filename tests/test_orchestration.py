"""Conversation behaviour through POST /chat: target resolution, proposals, failures, determinism."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import date

import pytest

from app.llm.base import CorpusBudgetExceeded, LLMError
from app.orchestration.orchestrator import ChatResult, enforce_route_contract
from app.tools.leave_tools import ToolOutcome
from tests.conftest import make_client
from tests.fakes import FakeLLM

REFUSAL = "I can't share information about that employee."


def scripted(plans: dict[str, dict], default: dict | None = None) -> FakeLLM:
    def planner(message: str) -> dict:
        for key, plan in plans.items():
            if key in message:
                return plan
        return default or {"intent": "policy_question", "search_query": message}

    return FakeLLM(planner)


def chat(client, employee, message, session=None):
    session = session or str(uuid.uuid4())
    r = client.post("/chat", headers={"X-Employee-Id": employee}, json={"session_id": session, "message": message})
    assert r.status_code == 200, r.text
    return r.json(), session


def apply_plan(start: str, end: str, leave_type: str = "annual") -> dict:
    return {"intent": "apply_leave", "leave_type": leave_type, "start_date_text": start, "start_date": start,
            "end_date_text": end, "end_date": end}


# --- who is being asked about ------------------------------------------------------------------

def test_ambiguous_name_asks_which_employee(settings):
    llm = scripted({"Kumar": {"intent": "leave_balance", "target_employee": "Kumar"}})
    with make_client(settings, llm) as client:
        body, _ = chat(client, "X200", "what is Kumar's balance")
    assert body["route"] == "clarify" and "Ravi Kumar (X201)" in body["answer"] and "Asha Kumar (X202)" in body["answer"]


def test_hr_can_look_up_any_employee_by_name(settings):
    llm = scripted({"Lena": {"intent": "leave_balance", "target_employee": "Lena"}})
    with make_client(settings, llm) as client:
        body, _ = chat(client, "X100", "Lena's balance please")
    assert body["route"] == "tool" and "Lena Berg (X301)" in body["answer"]
    assert body["tool_calls"][0]["args"] == {"employee_id": "X301"}


def test_manager_cannot_look_up_other_team_by_name(settings):
    llm = scripted({"Lena": {"intent": "leave_balance", "target_employee": "Lena"}})
    with make_client(settings, llm) as client:
        body, _ = chat(client, "X200", "Lena's balance please")
    assert body["route"] == "refuse" and body["answer"] == REFUSAL and body["tool_calls"] == []


def test_own_name_means_self(settings):
    llm = scripted({"Ravi": {"intent": "leave_balance", "target_employee": "Ravi Kumar"}})
    with make_client(settings, llm) as client:
        body, _ = chat(client, "X201", "balance for Ravi Kumar")
    assert body["route"] == "tool" and body["answer"].startswith("You have")


def test_list_requests_with_status_filter(settings):
    llm = scripted({"pending": {"intent": "leave_requests", "status_filter": "pending"}})
    with make_client(settings, llm) as client:
        body, _ = chat(client, "X201", "show my pending requests")
    assert body["tool_calls"] == [{"name": "list_leave_requests", "args": {"employee_id": "X201", "status": "pending"},
                                   "status": "success"}]
    assert "req-future" in body["answer"] and "req-past" not in body["answer"]


def test_manager_lists_report_requests_but_not_others(settings):
    llm = scripted({"X202": {"intent": "leave_requests", "target_employee": "X202"},
                    "X301": {"intent": "leave_requests", "target_employee": "X301"}})
    with make_client(settings, llm) as client:
        ok, _ = chat(client, "X200", "requests of X202")
        denied, _ = chat(client, "X200", "requests of X301")
    assert ok["route"] == "tool" and "req-other" in ok["answer"]
    assert denied["route"] == "refuse" and denied["answer"] == REFUSAL


# --- proposals ---------------------------------------------------------------------------------

def _count_requests(settings, employee="X201") -> int:
    conn = sqlite3.connect(settings.db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM leave_requests WHERE employee_id=?", (employee,)).fetchone()[0]
    finally:
        conn.close()


def test_decline_discards_the_proposal(settings):
    llm = scripted({"book": apply_plan("2026-09-16", "2026-09-17"), "no thanks": {"intent": "decline"}})
    with make_client(settings, llm) as client:
        before = _count_requests(settings)
        _, session = chat(client, "X201", "book leave")
        declined, _ = chat(client, "X201", "no thanks", session)
        late, _ = chat(client, "X201", "yes", session)
    assert declined["route"] == "clarify" and "haven't made any changes" in declined["answer"]
    assert late["tool_calls"] == [] and _count_requests(settings) == before


def test_only_the_latest_proposal_is_executed(settings):
    llm = scripted({"first": apply_plan("2026-09-16", "2026-09-16"), "second": apply_plan("2026-09-24", "2026-09-24")})
    with make_client(settings, llm) as client:
        _, session = chat(client, "X201", "first booking")
        chat(client, "X201", "second booking", session)
        done, _ = chat(client, "X201", "yes", session)
    assert done["tool_calls"][0]["args"]["start_date"] == "2026-09-24"
    conn = sqlite3.connect(settings.db_path)
    starts = {r[0] for r in conn.execute("SELECT start_date FROM leave_requests WHERE employee_id='X201'")}
    conn.close()
    assert "2026-09-24" in starts and "2026-09-16" not in starts


def test_confirmation_revalidates_changed_data(settings):
    llm = scripted({"book": apply_plan("2026-09-16", "2026-09-17")})
    with make_client(settings, llm) as client:
        proposal, session = chat(client, "X201", "book leave")
        assert proposal["pending_confirmation"]
        conn = sqlite3.connect(settings.db_path)
        conn.execute("UPDATE leave_balances SET balance_days = 1 WHERE employee_id='X201' AND leave_type='annual'")
        conn.commit()
        conn.close()
        done, _ = chat(client, "X201", "yes", session)
    assert done["route"] == "tool" and done["tool_calls"][0]["status"] == "error" and "balance is 1" in done["answer"]


def test_other_leave_type_is_a_tool_error(settings):
    plan = {**apply_plan("2026-09-16", "2026-09-17"), "leave_type": "other", "leave_type_text": "parental"}
    llm = scripted({"parental": plan})
    with make_client(settings, llm) as client:
        body, _ = chat(client, "X201", "book parental leave")
    assert body["route"] == "tool" and body["tool_calls"][0]["status"] == "error"
    assert body["tool_calls"][0]["args"]["leave_type"] == "parental" and "annual or sick" in body["answer"]


def test_sick_leave_proposal_uses_sick_balance(settings):
    llm = scripted({"sick": apply_plan("2026-09-16", "2026-09-18", "sick")})
    with make_client(settings, llm) as client:
        body, _ = chat(client, "X201", "sick leave")
    assert body["tool_calls"][0]["status"] == "error" and "balance is 2" in body["answer"]  # 3 days > 2 sick days


def test_cancel_with_several_matches_asks_which(settings):
    llm = scripted({"book": apply_plan("2026-10-20", "2026-10-20"), "cancel": {"intent": "cancel_leave"}})
    with make_client(settings, llm) as client:
        _, session = chat(client, "X201", "book")
        chat(client, "X201", "yes", session)
        body, _ = chat(client, "X201", "cancel my leave")
    assert body["route"] == "clarify" and "req-future" in body["answer"] and "Which one" in body["answer"]


def test_cancel_with_no_match_lists_requests(settings):
    llm = scripted({"cancel": {"intent": "cancel_leave", "start_date_text": "1 December", "start_date": "2026-12-01"}})
    with make_client(settings, llm) as client:
        body, _ = chat(client, "X201", "cancel my leave on 1 December")
    assert body["route"] == "tool" and body["tool_calls"][0]["name"] == "list_leave_requests"
    assert "couldn't find" in body["answer"]


def test_confirm_without_pending(settings):
    llm = scripted({"yes": {"intent": "confirm"}})
    with make_client(settings, llm) as client:
        body, _ = chat(client, "X201", "yes")
    assert body["route"] == "clarify" and body["tool_calls"] == []


# --- team leave dates ----------------------------------------------------------------------------

@pytest.mark.parametrize("plan", [
    {"intent": "team_leave"},
    {"intent": "team_leave", "start_date_text": "03/04"},
    {"intent": "team_leave", "start_date_text": "next month"},
])
def test_team_leave_needs_a_clear_range(settings, plan):
    with make_client(settings, scripted({}, default=plan)) as client:
        body, _ = chat(client, "X200", "who is off")
    assert body["route"] == "clarify"


def test_team_leave_single_day(settings):
    plan = {"intent": "team_leave", "start_date_text": "2026-09-22", "start_date": "2026-09-22"}
    with make_client(settings, scripted({}, default=plan)) as client:
        body, _ = chat(client, "X200", "who is off on the 22nd")
    assert body["tool_calls"][0]["args"] == {"start_date": "2026-09-22", "end_date": "2026-09-22"}
    assert "Asha Kumar" in body["answer"] and "Ravi Kumar" not in body["answer"]


# --- failures and fallbacks ------------------------------------------------------------------------

class FailingLLM(FakeLLM):
    def __init__(self, fail_on: str, **kw):
        super().__init__(**kw)
        self.fail_on = fail_on

    def _generate_json(self, *, system, user, schema):
        if self._purpose == self.fail_on:
            raise LLMError("simulated outage")
        return super()._generate_json(system=system, user=user, schema=schema)


def test_planner_outage_is_a_graceful_no_answer(settings):
    with make_client(settings, FailingLLM("plan")) as client:
        body, _ = chat(client, "X201", "How many volunteer days do I get?")
    assert body["route"] == "no_answer" and "unavailable" in body["answer"]


def test_compose_outage_falls_back_to_extracted_facts(settings):
    with make_client(settings, FailingLLM("compose")) as client:
        body, _ = chat(client, "X201", "How many volunteer days do I get?")
    assert body["route"] == "rag" and body["citations"] and "3 volunteer days" in body["answer"]


def test_hybrid_without_document_support_falls_back_to_tool(settings):
    plan = {"intent": "leave_balance", "needs_policy_documents": True, "search_query": "bicycle parking"}
    with make_client(settings, scripted({}, default=plan)) as client:
        body, _ = chat(client, "X201", "what will my balance be after parking changes")
    assert body["route"] == "tool" and "don't cover" in body["answer"]


def test_owner_keeps_working_after_another_employee_gets_403(settings):
    with make_client(settings, FakeLLM()) as client:
        _, session = chat(client, "X201", "How many volunteer days do I get?")
        intruder = client.post("/chat", headers={"X-Employee-Id": "X202"}, json={"session_id": session, "message": "hi"})
        again, _ = chat(client, "X201", "How many volunteer days do I get?", session)
    assert intruder.status_code == 403 and again["route"] == "rag"


def test_smalltalk_gets_capabilities(settings):
    with make_client(settings, scripted({"hello": {"intent": "smalltalk"}})) as client:
        body, _ = chat(client, "X201", "hello")
    assert body["route"] == "clarify" and "leave" in body["answer"]


def test_answers_are_deterministic(settings):
    with make_client(settings, FakeLLM()) as client:
        runs = [chat(client, "X201", "How many volunteer days do I get?")[0] for _ in range(3)]
    stripped = [{k: v for k, v in r.items() if k != "request_id"} for r in runs]
    assert stripped[0] == stripped[1] == stripped[2]
    assert len({r["request_id"] for r in runs}) == 3


def test_planner_history_has_no_names_from_answers(settings):
    llm = scripted({"Ravi": {"intent": "leave_balance", "target_employee": "Ravi"}})
    with make_client(settings, llm) as client:
        first, session = chat(client, "X200", "show Ravi balance")
        chat(client, "X200", "thanks, and volunteer days?", session)
    assert "Ravi Kumar" in first["answer"]
    last_plan = [c for c in llm.calls if c["purpose"] == "plan"][-1]["user"]
    assert "Ravi Kumar" not in last_plan and "ravi.kumar@" not in last_plan


# --- corpus budget with a large corpus -----------------------------------------------------------

def test_large_corpus_stays_within_budget(settings, data_dir):
    import json

    entries = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    for i in range(25):
        name = f"long_{i}.md"
        body = " ".join(["Employees receive volunteer days for community work and charity events."] * 60)
        (data_dir / "corpus" / name).write_text(f"# Long Policy {i}\n\n## 1. Volunteer days\n\n{body}\n", encoding="utf-8")
        entries.append({"doc_id": f"LONG-{i}", "file": name, "title": f"Long Policy {i}", "doc_type": "policy",
                        "regions": ["GLOBAL"], "effective_date": "2026-01-01", "access_level": "all"})
    (data_dir / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    llm = FakeLLM()
    with make_client(settings, llm) as client:
        chat(client, "X201", "How many volunteer days do I get?")
    extract = [c for c in llm.calls if c["purpose"] == "extract"]
    assert extract and all(0 < c["corpus_tokens"] <= 3000 for c in extract)
    from app.llm.tokens import count_tokens

    sources = extract[0]["user"].split("Sources:\n\n", 1)[1]
    blocks = sources.split("\n\n[S")
    assert sum(count_tokens(b if i == 0 else "[S" + b) for i, b in enumerate(blocks)) <= 3000


def test_llm_client_refuses_calls_over_budget():
    llm = FakeLLM()
    with pytest.raises(CorpusBudgetExceeded):
        llm.generate_json(purpose="extract", system="s", user="u", schema={}, corpus_tokens=3001)


# --- route contract ------------------------------------------------------------------------------

def _call(status="success") -> ToolOutcome:
    return ToolOutcome("get_leave_balance", {"employee_id": "X1"}, status)


@pytest.mark.parametrize(
    "route,citations,calls,expected",
    [
        ("rag", [], [], "no_answer"),
        ("rag", [("D", "s")], [_call()], "rag"),
        ("tool", [("D", "s")], [_call()], "tool"),
        ("tool", [], [], "no_answer"),
        ("hybrid", [], [_call()], "tool"),
        ("hybrid", [("D", "s")], [], "rag"),
        ("refuse", [("D", "s")], [_call()], "refuse"),
        ("clarify", [("D", "s")], [_call()], "clarify"),
        ("bogus", [], [], "no_answer"),
    ],
)
def test_route_contract(route, citations, calls, expected):
    result = enforce_route_contract(ChatResult("x", route, list(citations), list(calls), pending_confirmation=True))
    assert result.route == expected
    if expected in ("clarify", "no_answer", "refuse"):
        assert result.citations == [] and result.tool_calls == []
    if expected == "rag":
        assert result.tool_calls == [] and result.citations
    if expected == "tool":
        assert result.citations == [] and result.tool_calls
    assert result.pending_confirmation == (expected == "tool")
