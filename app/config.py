"""Runtime configuration, read once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


@dataclass(frozen=True)
class Settings:
    corpus_dir: str = "./data/corpus"
    manifest_path: str = "./data/manifest.json"
    db_path: str = "./data/veltrix.db"
    app_today: str = ""  # YYYY-MM-DD; empty means the system date
    llm_provider: str = "gemini"
    google_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    llm_timeout_seconds: float = 10.0
    embedding_provider: str = "local"  # local | none
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: str = ""
    corpus_token_budget: int = 3000
    leave_year: int = 2026
    log_level: str = "INFO"
    history_turns: int = 6

    def today(self) -> date:
        """The service's notion of today (APP_TODAY, or the system date)."""
        if self.app_today:
            return date.fromisoformat(self.app_today)
        return date.today()


def load_settings() -> Settings:
    return Settings(
        corpus_dir=_env("CORPUS_DIR", "./data/corpus"),
        manifest_path=_env("MANIFEST_PATH", "./data/manifest.json"),
        db_path=_env("DB_PATH", "./data/veltrix.db"),
        app_today=_env("APP_TODAY"),
        llm_provider=_env("LLM_PROVIDER", "gemini").lower(),
        google_api_key=_env("GOOGLE_API_KEY", _env("GEMINI_API_KEY")),
        gemini_model=_env("GEMINI_MODEL", "gemini-3.6-flash"),
        llm_timeout_seconds=float(_env("LLM_TIMEOUT_SECONDS", "10")),
        embedding_provider=_env("EMBEDDING_PROVIDER", "local").lower(),
        embedding_model=_env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        embedding_cache_dir=_env("EMBEDDING_CACHE_DIR"),
        corpus_token_budget=int(_env("CORPUS_TOKEN_BUDGET", "3000")),
        leave_year=int(_env("LEAVE_YEAR", "2026")),
        log_level=_env("LOG_LEVEL", "INFO").upper(),
        history_turns=int(_env("HISTORY_TURNS", "6")),
    )
