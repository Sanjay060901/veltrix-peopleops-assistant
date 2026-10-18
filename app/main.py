"""FastAPI app: POST /chat, GET /health. Ingestion runs in the background at startup."""
from __future__ import annotations

import json
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.api.schemas import ChatRequest, ChatResponse, Citation, ToolCall
from app.config import Settings, load_settings
from app.data import repositories as repo
from app.data.db import Database
from app.ingestion.pipeline import ingest
from app.llm.base import LLMClient
from app.orchestration.orchestrator import Orchestrator
from app.orchestration.sessions import SessionOwnershipError, SessionStore
from app.privacy.logging import setup_logging
from app.privacy.pii import registry, scrub_answer
from app.retrieval.embeddings import load_embedder

log = logging.getLogger("app")


@dataclass
class AppState:
    settings: Settings
    db: Database
    sessions: SessionStore = field(default_factory=SessionStore)
    orchestrator: Orchestrator | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    error: str | None = None


def initialise(state: AppState, llm: LLMClient | None, embedder=None, load_embeddings: bool = True) -> None:
    """Load personal-data registry, ingest the corpus and build the orchestrator."""
    try:
        with state.db.read() as conn:
            registry.load((e.full_name, e.email, e.phone) for e in repo.all_employees(conn))
        if embedder is None and load_embeddings:
            embedder = load_embedder(
                state.settings.embedding_provider, state.settings.embedding_model, state.settings.embedding_cache_dir
            )
        corpus = ingest(state.settings.corpus_dir, state.settings.manifest_path, state.db, embedder)
        state.orchestrator = Orchestrator(state.settings, state.db, corpus, llm)
        state.ready.set()
    except Exception as exc:  # keep the process alive; /health stays 503 and the log says why
        state.error = type(exc).__name__
        log.exception("startup failed")


def create_app(
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    embedder=None,
    background: bool = True,
    build_llm: bool = True,
) -> FastAPI:
    settings = settings or load_settings()
    setup_logging(settings.log_level)
    state = AppState(settings=settings, db=Database(settings.db_path))
    if llm is None and build_llm:
        from app.llm.gemini import build_llm as _build

        llm = _build(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if background:
            threading.Thread(target=initialise, args=(state, llm, embedder), daemon=True, name="ingestion").start()
        else:
            initialise(state, llm, embedder, load_embeddings=embedder is not None)
        yield

    app = FastAPI(title="Veltrix PeopleOps Assistant", lifespan=lifespan)
    app.state.veltrix = state

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
