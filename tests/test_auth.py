"""Application-level login (AUTH_MODE=login): typed employee IDs are never trusted."""
from __future__ import annotations

import time
import uuid
from dataclasses import replace

import pytest

from app.auth import manage, store
from app.auth.passwords import WeakPassword, check_policy, hash_password, verify_password
from tests.conftest import make_client
from tests.fakes import FakeLLM

EMPLOYEE_PW = "Employee-pass-2026"
MANAGER_PW = "Manager-pass-2026"
MSG = {"message": "How many volunteer days do I get?"}


@pytest.fixture
def login_settings(settings, db):
    db.ensure_schema()
    with db.transaction() as conn:
        store.set_password(conn, "X201", EMPLOYEE_PW)  # employee
        store.set_password(conn, "X200", MANAGER_PW)   # X201's manager
    return replace(settings, auth_mode="login", ui_enabled=True, login_max_failures=3)


@pytest.fixture
def app_client(login_settings):
    with make_client(login_settings, FakeLLM()) as c:
        yield c


def login(client, employee_id, password):
    return client.post("/auth/login", json={"employee_id": employee_id, "password": password})


def chat(client, headers=None, session=None):
    return client.post("/chat", headers=headers or {}, json={"session_id": session or str(uuid.uuid4()), **MSG})


# --- passwords ---------------------------------------------------------------------------------

def test_hash_is_salted_and_verifies():
    a, b = hash_password(EMPLOYEE_PW), hash_password(EMPLOYEE_PW)
    assert a != b and a.startswith("scrypt$")
    assert verify_password(EMPLOYEE_PW, a) and not verify_password("wrong-Pass-1", a)
    assert not verify_password(EMPLOYEE_PW, "garbage")


@pytest.mark.parametrize("weak", ["short1A", "alllowercase123", "ALLUPPERCASE123", "NoDigitsHereAtAll"])
def test_password_policy(weak):
    with pytest.raises(WeakPassword):
        check_policy(weak)


# --- the core protection ---------------------------------------------------------------------------

def test_typing_a_manager_id_without_signing_in_is_rejected(app_client):
    response = chat(app_client, {"X-Employee-Id": "X200"})
    assert response.status_code == 401 and response.json()["detail"] == "sign in required"


def test_signed_in_employee_cannot_claim_the_manager_id(app_client):
    assert login(app_client, "X201", EMPLOYEE_PW).status_code == 200
    response = chat(app_client, {"X-Employee-Id": "X200"})
    assert response.status_code == 401 and "does not match" in response.json()["detail"]


def test_identity_comes_from_the_session(app_client):
    assert login(app_client, "x201", EMPLOYEE_PW).json() == {"employee_id": "X201"}  # IDs are case-insensitive
    assert chat(app_client).status_code == 200  # no header needed
    assert chat(app_client, {"X-Employee-Id": "X201"}).status_code == 200  # a matching header is fine


def test_cookie_is_httponly_and_samesite_strict(app_client):
    cookie = login(app_client, "X201", EMPLOYEE_PW).headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "path=/" in cookie


def test_wrong_password_and_unknown_id_look_the_same(app_client):
    wrong = login(app_client, "X201", "Wrong-pass-2026")
    unknown = login(app_client, "NOBODY1", "Wrong-pass-2026")
    no_credential = login(app_client, "X202", "Wrong-pass-2026")  # real employee without a password
    assert wrong.status_code == unknown.status_code == no_credential.status_code == 401
    assert wrong.json() == unknown.json() == no_credential.json()


def test_repeated_failures_lock_the_account(app_client):
    for _ in range(3):
        assert login(app_client, "X200", "Wrong-pass-2026").status_code == 401
    locked = login(app_client, "X200", MANAGER_PW)  # even the right password is refused while locked
    assert locked.status_code == 429


def test_failures_across_accounts_lock_the_client(app_client):
    for who in ("A1", "B2", "C3"):
        login(app_client, who, "Wrong-pass-2026")
    assert login(app_client, "X201", EMPLOYEE_PW).status_code == 429  # same client address


def test_logout_ends_the_session(app_client):
    login(app_client, "X201", EMPLOYEE_PW)
    assert app_client.post("/auth/logout").status_code == 200
    assert chat(app_client).status_code == 401
    assert app_client.get("/auth/me").status_code == 401


def test_password_reset_ends_existing_sessions(app_client, db):
    login(app_client, "X201", EMPLOYEE_PW)
    assert chat(app_client).status_code == 200
    with db.transaction() as conn:
        store.set_password(conn, "X201", "Reset-pass-2026")  # e.g. by HR with the admin CLI
    assert chat(app_client).status_code == 401
    assert login(app_client, "X201", EMPLOYEE_PW).status_code == 401
    assert login(app_client, "X201", "Reset-pass-2026").status_code == 200


def test_removed_credential_ends_sessions(app_client, db):
    login(app_client, "X201", EMPLOYEE_PW)
    with db.transaction() as conn:
        store.remove_password(conn, "X201")
    assert chat(app_client).status_code == 401


def test_new_sign_in_replaces_the_old_session():
    sessions = store.LoginSessions(600, 3600)
    old = sessions.create("X201", "v1")
    new = sessions.create("X201", "v1")
    assert sessions.resolve(old) is None and sessions.resolve(new).employee_id == "X201"


def test_idle_and_absolute_expiry(monkeypatch):
    sessions = store.LoginSessions(idle_seconds=60, absolute_seconds=120)
    clock = [1000.0]
    monkeypatch.setattr(store.time, "time", lambda: clock[0])
    token = sessions.create("X201", "v1")
    clock[0] += 50
    assert sessions.resolve(token) is not None  # activity resets the idle timer
    clock[0] += 61
    assert sessions.resolve(token) is None
    token = sessions.create("X201", "v1")
    for _ in range(3):
        clock[0] += 50
        sessions.resolve(token)
    assert sessions.resolve(token) is None  # absolute lifetime (120 s) exceeded despite activity


def test_cross_site_requests_are_refused(app_client):
    evil = {"Origin": "https://evil.example"}
    assert app_client.post("/auth/login", headers=evil, json={"employee_id": "X201", "password": EMPLOYEE_PW}).status_code == 403
    login(app_client, "X201", EMPLOYEE_PW)
    assert chat(app_client, evil).status_code == 403
    assert chat(app_client, {"Origin": "http://testserver"}).status_code == 200


def test_chat_sessions_still_belong_to_one_employee(login_settings):
    session = str(uuid.uuid4())
    with make_client(login_settings, FakeLLM()) as c:
        login(c, "X201", EMPLOYEE_PW)
        assert chat(c, session=session).status_code == 200
        login(c, "X200", MANAGER_PW)
        assert chat(c, session=session).status_code == 403


def test_bad_login_body_is_400(app_client):
    assert app_client.post("/auth/login", json={"employee_id": "X201"}).status_code == 400
    assert app_client.post("/auth/login", content=b"not json", headers={"Content-Type": "application/json"}).status_code == 400


def test_ui_and_config_in_login_mode(app_client):
    assert app_client.get("/auth/config").json() == {"mode": "login"}
    assert app_client.get("/").status_code == 200 and "loginForm" in app_client.get("/").text


def test_login_endpoint_absent_in_header_mode(client):
    assert client.get("/auth/config").json() == {"mode": "header"}
    assert client.post("/auth/login", json={"employee_id": "X201", "password": EMPLOYEE_PW}).status_code == 404


def test_invalid_auth_mode_refuses_to_start(settings):
    from app.main import create_app

    with pytest.raises(ValueError):
        create_app(replace(settings, auth_mode="open"), llm=FakeLLM(), background=False, build_llm=False)
    with pytest.raises(ValueError):
        create_app(replace(settings, auth_mode="signed"), llm=FakeLLM(), background=False, build_llm=False)


def test_passwords_never_reach_the_logs(app_client, caplog):
    import logging

    with caplog.at_level(logging.INFO):
        login(app_client, "X201", "Wrong-pass-2026")
        login(app_client, "X200", MANAGER_PW)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "Wrong-pass-2026" not in text and MANAGER_PW not in text


# --- admin command -----------------------------------------------------------------------------------

def test_manage_cli(settings, db, monkeypatch, capsys):
    monkeypatch.setenv("DB_PATH", settings.db_path)
    assert manage.main(["set-password", "X202", "--generate"]) == 0
    generated = capsys.readouterr().out.split(": ", 1)[1].split()[0]
    with db.read() as conn:
        assert store.check_credentials(conn, "X202", generated)
    assert manage.main(["list"]) == 0 and "X202" in capsys.readouterr().out
    assert manage.main(["set-password", "NOPE1", "--generate"]) == 1
    monkeypatch.setattr(manage.getpass, "getpass", lambda prompt="": "weak")
    assert manage.main(["set-password", "X202"]) == 1  # policy enforced
    assert manage.main(["remove", "X202"]) == 0
    with db.read() as conn:
        assert not store.check_credentials(conn, "X202", generated)
