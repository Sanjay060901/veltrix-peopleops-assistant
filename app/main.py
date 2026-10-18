"""FastAPI app: POST /chat, GET /health. Ingestion runs in the background at startup."""
from __future__ import annotations

import json
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from urllib.parse import urlparse

from app.api import identity
from app.auth import store as auth_store
from app.api.schemas import ChatRequest, ChatResponse, Citation, ToolCall
from app.config import Settings, load_settings
from app.data import repositories as repo
from app.data.db import Database
from app.ingestion.cache import IngestCache
from app.ingestion.pipeline import ingest
from app.ingestion.watcher import corpus_fingerprint
from app.llm.base import LLMClient
from app.orchestration.orchestrator import Orchestrator
from app.orchestration.sessions import SessionOwnershipError, SessionStore
from app.privacy.logging import setup_logging
from app.privacy.pii import registry, scrub_answer
from app.retrieval.embeddings import load_embedder

log = logging.getLogger("app")

UI_PATH = Path(__file__).parent / "ui" / "index.html"
SESSION_COOKIE = "veltrix_session"


@dataclass
class AppState:
    settings: Settings
    db: Database
    sessions: SessionStore = field(default_factory=SessionStore)
    orchestrator: Orchestrator | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    error: str | None = None
    embedder: object | None = None
    fingerprint: tuple = ()
    reload_lock: threading.Lock = field(default_factory=threading.Lock)


def initialise(state: AppState, llm: LLMClient | None, embedder=None, load_embeddings: bool = True) -> None:
    """Load personal-data registry, ingest the corpus and build the orchestrator."""
    try:
        if embedder is None and load_embeddings:
            embedder = load_embedder(
                state.settings.embedding_provider, state.settings.embedding_model, state.settings.embedding_cache_dir
            )
        state.embedder = embedder
        state.orchestrator = Orchestrator(state.settings, state.db, _ingest(state), llm)
        state.ready.set()
    except Exception as exc:  # keep the process alive; /health stays 503 and the log says why
        state.error = type(exc).__name__
        log.exception("startup failed")


def _ingest(state: AppState):
    s = state.settings
    state.fingerprint = corpus_fingerprint(s.corpus_dir, s.manifest_path)
    with state.db.read() as conn:
        registry.load((e.full_name, e.email, e.phone) for e in repo.all_employees(conn))
    cache = IngestCache(s.cache_dir) if s.cache_dir else None
    return ingest(s.corpus_dir, s.manifest_path, state.db, state.embedder, cache=cache, embedding_model=s.embedding_model)


def reload_if_changed(state: AppState) -> bool:
    """Re-ingest when the manifest or a corpus file was added, removed, renamed or edited.

    The current corpus keeps serving requests until the new one is built, then it is swapped
    in one assignment, so /health stays 200 and no request sees a half-built index."""
    if state.orchestrator is None or not state.reload_lock.acquire(blocking=False):
        return False
    try:
        s = state.settings
        if corpus_fingerprint(s.corpus_dir, s.manifest_path) == state.fingerprint:
            return False
        log.info("corpus change detected, re-ingesting")
        state.orchestrator.corpus = _ingest(state)
        return True
    except Exception:
        log.exception("re-ingestion failed; keeping the previous corpus")
        return False
    finally:
        state.reload_lock.release()


def refresh_personal_data(state: AppState) -> None:
    """Re-read names/emails/phones used for redaction, in case employee rows changed."""
    try:
        with state.db.read() as conn:
            registry.load((e.full_name, e.email, e.phone) for e in repo.all_employees(conn))
    except Exception:
        log.exception("could not refresh personal-data registry")


def _watch(state: AppState, stop: threading.Event) -> None:
    interval = state.settings.corpus_reload_seconds
    while not stop.wait(interval):
        reload_if_changed(state)
        refresh_personal_data(state)


def create_app(
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    embedder=None,
    background: bool = True,
    build_llm: bool = True,
) -> FastAPI:
    settings = settings or load_settings()
    setup_logging(settings.log_level)
    identity.check_secret(settings.identity_secret)
    auth_mode = settings.effective_auth_mode()
    signed_mode = auth_mode == "signed"
    login_mode = auth_mode == "login"
    # In signed mode identities come from a gateway the browser UI cannot sign for, so it is
    # never served. In login mode the UI shows a sign-in form instead of a free ID field.
    ui_enabled = settings.ui_enabled and not signed_mode
    log.info("identity_mode=%s ui_enabled=%s", auth_mode, ui_enabled)
    state = AppState(settings=settings, db=Database(settings.db_path))
    logins = auth_store.LoginSessions(settings.login_idle_minutes * 60, settings.login_absolute_hours * 3600)
    throttle = auth_store.LoginThrottle(settings.login_max_failures, 15 * 60, settings.login_lockout_minutes * 60)
    if login_mode:
        state.db.ensure_schema()
    if llm is None and build_llm:
        from app.llm.gemini import build_llm as _build

        llm = _build(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = threading.Event()
        if background:
            threading.Thread(target=initialise, args=(state, llm, embedder), daemon=True, name="ingestion").start()
        else:
            initialise(state, llm, embedder, load_embeddings=embedder is not None)
        if settings.corpus_reload_seconds > 0:
            threading.Thread(target=_watch, args=(state, stop), daemon=True, name="corpus-watcher").start()
        yield
        stop.set()

    app = FastAPI(title="Veltrix PeopleOps Assistant", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.veltrix = state

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path == "/chat" or request.url.path.startswith("/auth/"):
            response.headers["Cache-Control"] = "no-store"  # answers can contain personal data
        if request.url.path == "/":
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
            )
        return response

    @app.get("/", include_in_schema=False)
    def ui():
        """Development chat page (UI_ENABLED=true). It only calls /chat and /health."""
        if not ui_enabled:
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        return FileResponse(UI_PATH, media_type="text/html")

    # --- application-level login (AUTH_MODE=login) ---------------------------------------------

    def _same_origin(request: Request) -> bool:
        """Cookie-authenticated writes must come from this site (defence against CSRF on top of
        SameSite=Strict). Browsers always send Origin on cross-site POSTs."""
        origin = request.headers.get("origin")
        return not origin or urlparse(origin).netloc == request.headers.get("host", "")

    def _session_employee(request: Request) -> str | None:
        token = request.cookies.get(SESSION_COOKIE)
        session = logins.resolve(token)
        if session is None:
            return None
        with state.db.read() as conn:
            current = auth_store.credential_version(conn, session.employee_id)
        if current != session.credential_version:  # password reset or removed since sign-in
            logins.end(token)
            return None
        return session.employee_id

    @app.get("/auth/config", include_in_schema=False)
    def auth_config():
        return {"mode": auth_mode}

    @app.post("/auth/login", include_in_schema=False)
    async def login(request: Request):
        if not login_mode:
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        if not _same_origin(request):
            return JSONResponse(status_code=403, content={"detail": "cross-site request refused"})
        try:
            payload = json.loads(await request.body() or b"null")
            employee_id = str(payload["employee_id"]).strip().upper()[:64]
            password = str(payload["password"])[:256]
            if not employee_id or not password:
                raise ValueError("empty")
        except (ValueError, KeyError, TypeError):
            return JSONResponse(status_code=400, content={"detail": "employee_id and password are required"})
        host = request.client.host if request.client else "unknown"
        client_key, account_key = f"ip:{host}", f"id:{employee_id}"
        if throttle.locked(account_key, client_key):
            return JSONResponse(status_code=429, content={"detail": "Too many failed sign-in attempts. Try again later."})

        def check() -> str | None:
            with state.db.read() as conn:
                if not auth_store.check_credentials(conn, employee_id, password):
                    return None
                return auth_store.credential_version(conn, employee_id)

        version = await run_in_threadpool(check)  # scrypt is deliberately slow; keep the loop free
        if version is None:
            throttle.failure(account_key, client_key)
            log.warning("failed sign-in attempt")
            return JSONResponse(status_code=401, content={"detail": "Invalid employee ID or password."})
        throttle.success(account_key, client_key)
        token = logins.create(employee_id, version)
        log.info("sign-in employee=%s", employee_id)
        response = JSONResponse(content={"employee_id": employee_id})
        response.set_cookie(
            SESSION_COOKIE, token, max_age=settings.login_absolute_hours * 3600, httponly=True,
            secure=settings.cookie_secure, samesite="strict", path="/",
        )
        return response

    @app.post("/auth/logout", include_in_schema=False)
    def logout(request: Request):
        logins.end(request.cookies.get(SESSION_COOKIE))
        response = JSONResponse(content={"signed_out": True})
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    @app.get("/auth/me", include_in_schema=False)
    def me(request: Request):
        employee_id = _session_employee(request) if login_mode else None
        if employee_id is None:
            return JSONResponse(status_code=401, content={"detail": "not signed in"})
        return {"employee_id": employee_id}

    @app.get("/health")
    def health():
        if state.ready.is_set():
            return {"status": "ok"}
        return JSONResponse(status_code=503, content={"status": "starting" if not state.error else "error"})

    @app.post("/chat")
    async def chat(request: Request):
        employee_id = (request.headers.get("x-employee-id") or "").strip()
        if login_mode:
            # Identity comes only from the signed-in session; a typed header is never trusted.
            if not _same_origin(request):
                return JSONResponse(status_code=403, content={"detail": "cross-site request refused"})
            session_employee = _session_employee(request)
            if session_employee is None:
                return JSONResponse(status_code=401, content={"detail": "sign in required"})
            if employee_id and employee_id != session_employee:
                log.warning("X-Employee-Id did not match the signed-in employee")
                return JSONResponse(status_code=401, content={"detail": "X-Employee-Id does not match the signed-in user"})
            employee_id = session_employee
        if not employee_id:
            return JSONResponse(status_code=401, content={"detail": "X-Employee-Id header is required"})
        if signed_mode and not identity.verify(
            settings.identity_secret, employee_id, request.headers.get("x-employee-signature"), settings.identity_max_age_seconds
        ):
            # Checked before any database lookup, so unsigned callers cannot probe which IDs exist.
            log.warning("rejected request with missing or invalid identity signature")
            return JSONResponse(status_code=401, content={"detail": "identity could not be verified"})
        if not state.ready.is_set() or state.orchestrator is None:
            return JSONResponse(status_code=503, content={"detail": "service is starting"})
        with state.db.read() as conn:
            requester = repo.get_employee(conn, employee_id)
        if requester is None:
            return JSONResponse(status_code=401, content={"detail": "unknown employee"})

        raw = await request.body()
        try:
            payload = json.loads(raw) if raw else None
            if not isinstance(payload, dict):
                raise ValueError("body must be a JSON object")
            body = ChatRequest.model_validate(payload)
        except (ValueError, ValidationError):
            return JSONResponse(status_code=400, content={"detail": "body must be JSON with string fields session_id and message"})

        try:
            session = state.sessions.get_or_create(body.session_id, requester.employee_id)
        except SessionOwnershipError:
            return JSONResponse(status_code=403, content={"detail": "session belongs to another employee"})

        # Sync work (LLM calls, SQLite) runs in the threadpool so the event loop stays free.
        result = await run_in_threadpool(state.orchestrator.handle, requester, session, body.message)
        response = ChatResponse(
            request_id=str(uuid.uuid4()),
            answer=result.answer,
            route=result.route,
            citations=[Citation(doc_id=d, locator=loc) for d, loc in result.citations],
            tool_calls=[
                ToolCall(name=t.name, args={k: scrub_answer(v) if isinstance(v, str) else v for k, v in t.args.items()}, status=t.status)
                for t in result.tool_calls
            ],
            pending_confirmation=result.pending_confirmation,
        )
        return JSONResponse(content=response.model_dump())

    return app


def app_factory() -> FastAPI:
    return create_app()
