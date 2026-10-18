from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.data.db import Database
from app.ingestion.pipeline import ingest
from app.main import create_app
from tests.fakes import FakeLLM
from tests.fixtures import TODAY, build_corpus, build_db


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    build_corpus(tmp_path)
    build_db(tmp_path / "veltrix.db")
    return tmp_path


@pytest.fixture
def settings(data_dir: Path) -> Settings:
    return Settings(
        corpus_dir=str(data_dir / "corpus"),
        manifest_path=str(data_dir / "manifest.json"),
        db_path=str(data_dir / "veltrix.db"),
        app_today=TODAY,
        embedding_provider="none",
        log_level="WARNING",
    )


@pytest.fixture
def db(settings: Settings) -> Database:
    return Database(settings.db_path)


@pytest.fixture
def corpus(settings: Settings, db: Database):
    return ingest(settings.corpus_dir, settings.manifest_path, db, None)


def make_client(settings: Settings, llm: FakeLLM) -> TestClient:
    app = create_app(settings, llm=llm, background=False, build_llm=False)
    return TestClient(app)


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(settings: Settings, llm: FakeLLM):
    with make_client(settings, llm) as c:
        yield c
