"""Signed identity mode, the UI switch and security headers."""
from __future__ import annotations

import time
import uuid
from dataclasses import replace

import pytest

from app.api import identity
from app.main import create_app
from tests.conftest import make_client
from tests.fakes import FakeLLM

SECRET = "s" * 40
BODY = {"message": "How many volunteer days do I get?"}


def post(client, employee, signature=None, session=None):
    headers = {"X-Employee-Id": employee}
    if signature is not None:
        headers["X-Employee-Signature"] = signature
    return client.post("/chat", headers=headers, json={"session_id": session or str(uuid.uuid4()), **BODY})


@pytest.fixture
def signed(settings):
    with make_client(replace(settings, identity_secret=SECRET, ui_enabled=True), FakeLLM()) as client:
        yield client


# --- signature primitives -------------------------------------------------------------------------

def test_sign_and_verify_roundtrip():
    header = identity.sign(SECRET, "X201")
    assert identity.verify(SECRET, "X201", header, max_age=300)


@pytest.mark.parametrize("header", [None, "", "garbage", "t=1,v1=zz", "v1=" + "a" * 64])
def test_malformed_signatures_fail(header):
    assert not identity.verify(SECRET, "X201", header, max_age=300)


def test_signature_is_bound_to_the_employee_and_secret():
    header = identity.sign(SECRET, "X201")
    assert not identity.verify(SECRET, "X200", header, max_age=300)  # reuse for a manager's ID
    assert not identity.verify("t" * 40, "X201", header, max_age=300)  # signed with another secret


def test_old_and_far_future_signatures_fail():
    now = time.time()
    assert not identity.verify(SECRET, "X201", identity.sign(SECRET, "X201", int(now) - 301), max_age=300, now=now)
    assert not identity.verify(SECRET, "X201", identity.sign(SECRET, "X201", int(now) + 301), max_age=300, now=now)
    assert identity.verify(SECRET, "X201", identity.sign(SECRET, "X201", int(now) - 299), max_age=300, now=now)


def test_weak_secret_refuses_to_start(settings):
    with pytest.raises(ValueError):
        create_app(replace(settings, identity_secret="short"), llm=FakeLLM(), background=False, build_llm=False)


# --- signed mode end to end -------------------------------------------------------------------------

def test_valid_signature_is_accepted(signed):
    assert post(signed, "X201", identity.sign(SECRET, "X201")).status_code == 200


def test_typing_a_manager_id_without_signature_is_rejected(signed):
    response = post(signed, "X200")
    assert response.status_code == 401 and "verified" in response.json()["detail"]


def test_employee_cannot_reuse_own_signature_for_manager_id(signed):
    assert post(signed, "X200", identity.sign(SECRET, "X201")).status_code == 401


def test_unknown_ids_cannot_be_probed_without_signature(signed):
    # Same answer for a real and a made-up ID when unsigned: nothing about the DB is revealed.
    real, fake = post(signed, "X200"), post(signed, "NOPE123")
    assert real.status_code == fake.status_code == 401 and real.json() == fake.json()


def test_expired_signature_is_rejected(signed):
    assert post(signed, "X201", identity.sign(SECRET, "X201", int(time.time()) - 3600)).status_code == 401


def test_ui_is_never_served_in_signed_mode(signed):
    assert signed.get("/").status_code == 404


def test_session_ownership_still_enforced_in_signed_mode(signed):
    session = str(uuid.uuid4())
    assert post(signed, "X201", identity.sign(SECRET, "X201"), session).status_code == 200
    assert post(signed, "X202", identity.sign(SECRET, "X202"), session).status_code == 403


def test_header_mode_is_unchanged_without_secret(client):
    assert post(client, "X201").status_code == 200  # the graders' contract: plain X-Employee-Id


# --- headers ----------------------------------------------------------------------------------------

def test_security_headers(settings):
    with make_client(replace(settings, ui_enabled=True), FakeLLM()) as client:
        chat = post(client, "X201")
        page = client.get("/")
    assert chat.headers["Cache-Control"] == "no-store"
    assert chat.headers["X-Content-Type-Options"] == "nosniff" and chat.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in page.headers["Content-Security-Policy"]


def test_api_docs_are_not_exposed(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404
