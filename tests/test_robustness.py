"""Concurrency, extra date phrases, and the Gemini adapter's error handling (no network)."""
from __future__ import annotations

import threading
from datetime import date

import pytest
from google.genai import errors

from app.data import repositories as repo
from app.llm.base import LLMError
from app.llm.gemini import GeminiClient
from app.orchestration.dates import resolve_phrase
from app.tools import leave_tools as T

TODAY = date(2026, 9, 15)  # Tuesday


# --- concurrency -----------------------------------------------------------------------------------

def test_concurrent_overlapping_applies_only_one_succeeds(db, corpus):
    with db.read() as conn:
        requester = repo.get_employee(conn, "X202")
    ctx = T.ToolContext(db, corpus.docs, requester, TODAY, 2026)
    results: list[str] = []
    barrier = threading.Barrier(4)

    def worker():
        barrier.wait()
        results.append(T.apply_leave(ctx, "annual", date(2026, 10, 13), date(2026, 10, 14), execute=True).status)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["error", "error", "error", "success"]
    with db.read() as conn:
        assert repo.get_balances(conn, "X202", "annual")["annual"] == 10.5  # 12.5 - 2, subtracted once


# --- date phrases ---------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phrase,start,end",
    [
        ("today", date(2026, 9, 15), date(2026, 9, 15)),
        ("tomorrow", date(2026, 9, 16), date(2026, 9, 16)),
        ("day after tomorrow", date(2026, 9, 17), date(2026, 9, 17)),
        ("this week", date(2026, 9, 15), date(2026, 9, 18)),
        ("next Monday", date(2026, 9, 21), date(2026, 9, 21)),
        ("this Friday", date(2026, 9, 18), date(2026, 9, 18)),
        ("Thursday", date(2026, 9, 17), date(2026, 9, 17)),
        ("2026-12-24", date(2026, 12, 24), date(2026, 12, 24)),
        ("24th of December", date(2026, 12, 24), date(2026, 12, 24)),
        ("Dec 24", date(2026, 12, 24), date(2026, 12, 24)),
        ("on 5 Oct", date(2026, 10, 5), date(2026, 10, 5)),
    ],
)
def test_date_phrases(phrase, start, end):
    r = resolve_phrase(phrase, TODAY, 2026)
    assert r.status == "ok" and (r.start, r.end) == (start, end)


@pytest.mark.parametrize("phrase", ["31 February", "yesterday", "last week", "next month", "05/06"])
def test_ambiguous_or_invalid_dates(phrase):
    assert resolve_phrase(phrase, TODAY, 2026).status == "ambiguous"


def test_this_week_on_saturday_is_ambiguous():
    assert resolve_phrase("this week", date(2026, 9, 19), 2026).status == "ambiguous"


# --- Gemini adapter (client mocked, no network) ---------------------------------------------------

class _Response:
    def __init__(self, text):
        self.text = text


def _client(monkeypatch, behaviours):
    client = GeminiClient("test-key", "gemini-test", 5, 3000)
    calls = []

    def fake_generate(model, contents, config):
        calls.append(config)
        outcome = behaviours.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)

    monkeypatch.setattr(client._client.models, "generate_content", fake_generate)
    return client, calls


def _gen(client):
    return client.generate_json(purpose="plan", system="s", user="u", schema={"type": "object"}, corpus_tokens=0)


def test_timeout_is_at_least_ten_seconds():
    assert GeminiClient("k", "m", 5, 3000)._timeout_ms == 10_000
    assert GeminiClient("k", "m", 20, 3000)._timeout_ms == 20_000


def test_unsupported_thinking_level_is_retried_without_it(monkeypatch):
    err = errors.ClientError(400, {"error": {"code": 400, "message": "thinking_level is not supported", "status": "INVALID_ARGUMENT"}})
    client, calls = _client(monkeypatch, [err, '{"intent": "smalltalk"}'])
    assert _gen(client) == {"intent": "smalltalk"}
    assert calls[0].thinking_config is not None and calls[1].thinking_config is None


def test_server_error_is_retried_once(monkeypatch):
    err = errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})
    client, calls = _client(monkeypatch, [err, '{"ok": true}'])
    assert _gen(client) == {"ok": True} and len(calls) == 2


def test_rate_limit_is_not_retried(monkeypatch):
    err = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    client, calls = _client(monkeypatch, [err, '{"ok": true}'])
    with pytest.raises(LLMError, match="429"):
        _gen(client)
    assert len(calls) == 1


@pytest.mark.parametrize("bad", ["not json", "[1, 2]"])
def test_malformed_output_raises_after_retry(monkeypatch, bad):
    client, calls = _client(monkeypatch, [bad, bad])
    with pytest.raises(LLMError):
        _gen(client)
    assert len(calls) == 2


def test_request_uses_temperature_zero_and_json_schema(monkeypatch):
    client, calls = _client(monkeypatch, ['{"a": 1}'])
    _gen(client)
    config = calls[0]
    assert config.temperature == 0 and config.response_mime_type == "application/json"
    assert config.response_json_schema == {"type": "object"} and config.seed is not None


# --- reviewer configuration variants ------------------------------------------------------------

@pytest.mark.parametrize("name", ["GOOGLE_API_KEY", "GEMINI_API_KEY"])
def test_either_api_key_name_is_accepted(monkeypatch, name):
    from app.config import load_settings

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv(name, "key-123")
    assert load_settings().google_api_key == "key-123"


@pytest.mark.parametrize("provider", ["gemini", "GEMINI", "google", "Google-Gemini"])
def test_provider_aliases_build_the_gemini_client(monkeypatch, provider):
    from dataclasses import replace

    from app.config import load_settings
    from app.llm.gemini import build_llm

    monkeypatch.setenv("LLM_PROVIDER", provider)
    settings = replace(load_settings(), google_api_key="key-123")
    assert isinstance(build_llm(settings), GeminiClient)


# --- deadline-aware retries ------------------------------------------------------------------------

class _SlowClock:
    """Each call to monotonic() advances 6 s, so every attempt looks slow."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 6.0
        return self.t


def _overloaded():
    return errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})


def test_slow_failure_is_retried_when_the_request_has_time(monkeypatch):
    from app.llm import gemini

    client, calls = _client(monkeypatch, [_overloaded(), '{"ok": true}'])
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)
    monkeypatch.setattr(gemini, "remaining_seconds", lambda: 20.0)  # plenty left in this request
    monkeypatch.setattr(gemini.time, "monotonic", _SlowClock())
    assert _gen(client) == {"ok": True} and len(calls) == 2


def test_slow_failure_is_not_retried_near_the_deadline(monkeypatch):
    from app.llm import gemini

    client, calls = _client(monkeypatch, [_overloaded(), '{"ok": true}'])
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)
    monkeypatch.setattr(gemini, "remaining_seconds", lambda: 8.0)  # a second 10 s attempt would not fit
    monkeypatch.setattr(gemini.time, "monotonic", _SlowClock())
    with pytest.raises(LLMError):
        _gen(client)
    assert len(calls) == 1


def test_request_deadline_is_per_thread():
    from app.llm.base import remaining_seconds, request_deadline

    assert remaining_seconds() is None
    with request_deadline(27):
        assert 26 < remaining_seconds() <= 27
        seen = []
        t = threading.Thread(target=lambda: seen.append(remaining_seconds()))
        t.start(); t.join()
        assert seen == [None]  # another request's thread is not affected
    assert remaining_seconds() is None
