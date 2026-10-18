"""A scripted stand-in for the LLM. Deterministic code paths are tested without network calls."""
from __future__ import annotations

import re
from typing import Any, Callable

from app.llm.base import LLMClient
from app.orchestration.planner import PLAN_SCHEMA

PLAN_DEFAULTS: dict[str, Any] = {
    k: (False if v.get("type") == "boolean" else "") for k, v in PLAN_SCHEMA["properties"].items()
}


class FakeLLM(LLMClient):
    name = "fake"

    def __init__(self, planner: Callable[[str], dict] | None = None, budget: int = 3000) -> None:
        super().__init__(budget)
        self.planner = planner or (lambda msg: {"intent": "policy_question", "search_query": msg})
        self.calls: list[dict[str, Any]] = []
        self._purpose = ""

    def generate_json(self, *, purpose, system, user, schema, corpus_tokens):
        self._purpose = purpose
        self.calls.append({"purpose": purpose, "system": system, "user": user, "corpus_tokens": corpus_tokens})
        return super().generate_json(purpose=purpose, system=system, user=user, schema=schema, corpus_tokens=corpus_tokens)

    def _generate_json(self, *, system: str, user: str, schema: dict) -> dict:
        if self._purpose == "plan":
            message = user.rsplit("Latest user message: ", 1)[-1]
            return {**PLAN_DEFAULTS, **self.planner(message)}
        if self._purpose == "extract":
            return self._extract(user)
        if self._purpose == "compose":
            facts = re.findall(r"^(F\d+) \[([^\]]+)\] rule: [^|]+\| value: ([^|]+)\|", user, re.M)
            answer = "; ".join(f"{label}: {value.strip()}" for _, label, value in facts) or "nothing"
            return {"answer": answer, "used_fact_ids": [f for f, _, _ in facts]}
        raise AssertionError(self._purpose)

    @staticmethod
    def _extract(user: str) -> dict:
        """Like a real extractor, only reports rules the question asks about."""
        question = user.splitlines()[0].lower()
        claims = []
        for label, body in re.findall(r"^\[(S\d+)\] (.*?)(?=^\[S\d+\] |\Z)", user, re.M | re.S):
            if "volunteer" in question:
                for m in re.finditer(r"(\d+) volunteer days", body):
                    claims.append(_claim("R1", label, f"{m.group(1)} days", m.group(0)))
            if "expense" in question:
                for m in re.finditer(r"within (\d+) days", body):
                    claims.append(_claim("R2", label, f"{m.group(1)} days", m.group(0)))
        rules = [{"rule_id": "R1", "description": "volunteer days per year"}, {"rule_id": "R2", "description": "expense claim deadline"}]
        return {"rules": rules, "claims": claims, "coverage": "full" if claims else "none", "uncovered": ""}


def _claim(rule_id: str, source: str, value: str, statement: str) -> dict:
    return {
        "rule_id": rule_id, "source": source, "value": value, "statement": statement,
        "defers_to_other_document": False, "describes_other_version": False,
    }
