"""Google Gemini adapter (google-genai SDK). Temperature 0, fixed seed, JSON-schema output."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.llm.base import LLMClient, LLMError, remaining_seconds

log = logging.getLogger(__name__)

RETRY_WINDOW_SECONDS = 3.0
RETRY_MARGIN_SECONDS = 2.0
RETRY_BACKOFF_SECONDS = 0.5
MIN_TIMEOUT_SECONDS = 10.0


class GeminiClient(LLMClient):
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout_seconds: float, corpus_token_budget: int) -> None:
        super().__init__(corpus_token_budget)
        if not api_key:
            raise LLMError("GOOGLE_API_KEY is not set")
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self._model = model
        # The API rejects deadlines below 10 s.
        self._timeout_ms = int(max(timeout_seconds, MIN_TIMEOUT_SECONDS) * 1000)
        self._use_thinking_config = True

    def _config(self, system: str, schema: dict[str, Any]):
        from google.genai import types

        kwargs: dict[str, Any] = dict(
            system_instruction=system,
            temperature=0,
            seed=7,
            candidate_count=1,
            response_mime_type="application/json",
            response_json_schema=schema,
            http_options=types.HttpOptions(timeout=self._timeout_ms),
        )
        if self._use_thinking_config:
            # Minimal thinking keeps latency low and output stable; models that reject the
            # setting are retried without it.
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="MINIMAL")
        return types.GenerateContentConfig(**kwargs)

    def _generate_json(self, *, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        from google.genai import errors

        started = time.monotonic()
        last_exc: Exception | None = None
        for _attempt in range(2):
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=user, config=self._config(system, schema)
                )
                data = json.loads(response.text or "")
                if not isinstance(data, dict):
                    raise ValueError("expected a JSON object")
                return data
            except errors.ClientError as exc:
                if self._use_thinking_config and "thinking" in str(exc).lower():
                    log.info("model does not accept thinking_level, retrying without it")
                    self._use_thinking_config = False
                    last_exc = exc
                    continue
                raise LLMError(f"gemini client error {getattr(exc, 'code', '')} {getattr(exc, 'status', '')}") from exc
            except Exception as exc:  # server errors, timeouts, malformed JSON
                last_exc = exc
                # Retry if the failure was quick, or if the request still has time for a full
                # second attempt; every request must finish within 30 s.
                remaining = remaining_seconds()
                quick = time.monotonic() - started <= RETRY_WINDOW_SECONDS
                if not quick and (remaining is None or remaining < self._timeout_ms / 1000 + RETRY_MARGIN_SECONDS):
                    break
                time.sleep(RETRY_BACKOFF_SECONDS)
        raise LLMError(f"gemini call failed: {type(last_exc).__name__}") from last_exc


def build_llm(settings) -> LLMClient | None:
    provider = settings.llm_provider
    if provider in ("gemini", "google", "google-gemini", "googleai", "google_genai"):
        try:
            return GeminiClient(
                settings.google_api_key, settings.gemini_model, settings.llm_timeout_seconds, settings.corpus_token_budget
            )
        except LLMError as exc:
            log.error("LLM unavailable: %s", exc)
            return None
    log.error("unsupported LLM_PROVIDER=%s (supported: gemini)", provider)
    return None
