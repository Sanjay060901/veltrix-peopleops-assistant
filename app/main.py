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

from app.api import identity
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
    signed_mode = bool(settings.identity_secret)
    # The UI lets a person type any employee ID, so it is never served when identities are signed.
    ui_enabled = settings.ui_enabled and not signed_mode
    log.info("identity_mode=%s ui_enabled=%s", "signed" if signed_mode else "header", ui_enabled)
    state = AppState(settings=settings, db=Database(settings.db_path))
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
        if request.url.path == "/chat":
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

    @app.get("/health")
    def health():
        if state.ready.is_set():
            return {"status": "ok"}
        return JSONResponse(status_code=503, content={"status": "starting" if not state.error else "error"})

    @app.post("/chat")
    async def chat(request: Request):
        employee_id = (request.headers.get("x-employee-id") or "").strip()
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
