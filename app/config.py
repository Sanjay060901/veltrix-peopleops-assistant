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
    corpus_reload_seconds: float = 30.0  # 0 disables watching for corpus changes
    cache_dir: str = ""  # ingestion cache keyed by file SHA-256; empty disables
    ui_enabled: bool = False  # development chat page at /; off unless explicitly enabled
    identity_secret: str = ""  # set => X-Employee-Id must carry a valid HMAC signature
    identity_max_age_seconds: int = 300
    auth_mode: str = ""  # header | signed | login; empty = signed if identity_secret else header
    cookie_secure: bool = False  # set true when served over HTTPS
    login_idle_minutes: int = 30
    login_absolute_hours: int = 8
    login_max_failures: int = 5
    login_lockout_minutes: int = 15

    def effective_auth_mode(self) -> str:
        mode = (self.auth_mode or ("signed" if self.identity_secret else "header")).lower()
        if mode not in ("header", "signed", "login"):
            raise ValueError(f"AUTH_MODE must be header, signed or login, not {mode!r}")
        if mode == "signed" and not self.identity_secret:
            raise ValueError("AUTH_MODE=signed needs IDENTITY_SECRET")
        return mode

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
        corpus_reload_seconds=float(_env("CORPUS_RELOAD_SECONDS", "30")),
        cache_dir=_env("CACHE_DIR", "./.cache/ingest"),
        ui_enabled=_env("UI_ENABLED", "false").lower() in ("1", "true", "yes"),
        identity_secret=_env("IDENTITY_SECRET"),
        identity_max_age_seconds=int(_env("IDENTITY_MAX_AGE_SECONDS", "300")),
        auth_mode=_env("AUTH_MODE"),
        cookie_secure=_env("COOKIE_SECURE", "false").lower() in ("1", "true", "yes"),
        login_idle_minutes=int(_env("LOGIN_IDLE_MINUTES", "30")),
        login_absolute_hours=int(_env("LOGIN_ABSOLUTE_HOURS", "8")),
        login_max_failures=int(_env("LOGIN_MAX_FAILURES", "5")),
        login_lockout_minutes=int(_env("LOGIN_LOCKOUT_MINUTES", "15")),
    )
