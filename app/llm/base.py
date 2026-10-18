"""Provider-neutral LLM interface. Every call declares and logs its corpus-token count."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

log = logging.getLogger("app.llm")


class LLMError(Exception):
    pass


class CorpusBudgetExceeded(LLMError):
    pass


class LLMClient(ABC):
    def __init__(self, corpus_token_budget: int = 3000) -> None:
        self.corpus_token_budget = corpus_token_budget

    def generate_json(self, *, purpose: str, system: str, user: str, schema: dict[str, Any], corpus_tokens: int) -> dict[str, Any]:
        """Run one structured-output call. `corpus_tokens` is counted by the caller (the packer)
        over every piece of corpus-derived text placed in `user`; the system prompt never holds any."""
        if corpus_tokens > self.corpus_token_budget:
            # Hard stop: this must never happen, the packer enforces the budget first.
            log.error("llm_call_blocked purpose=%s corpus_tokens=%s budget=%s", purpose, corpus_tokens, self.corpus_token_budget)
            raise CorpusBudgetExceeded(f"{corpus_tokens} corpus tokens exceeds budget")
        started = time.monotonic()
        try:
            result = self._generate_json(system=system, user=user, schema=schema)
            log.info(
                "llm_call purpose=%s provider=%s corpus_tokens=%s seconds=%.2f status=ok",
                purpose, self.name, corpus_tokens, time.monotonic() - started,
            )
            return result
        except Exception as exc:
            log.warning(
                "llm_call purpose=%s provider=%s corpus_tokens=%s seconds=%.2f status=error error=%s",
                purpose, self.name, corpus_tokens, time.monotonic() - started, type(exc).__name__,
            )
            raise

    name: str = "base"

    @abstractmethod
    def _generate_json(self, *, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        ...
