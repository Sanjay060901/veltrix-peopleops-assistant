"""Precedence between documents that state values for the same rule (TASK §5.1 rules 2-3).

Input: claims extracted from eligible (and upcoming) documents, grouped by rule. Output, per rule:
the winning claim(s), whether an unresolved conflict remains, and any upcoming change that would
win on its effective date. Pure functions; no LLM involved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from app.ingestion.models import Chunk, DocMeta

# Authority test: these doc types outrank faq and wiki. Unknown types rank lowest.
HIGH_AUTHORITY = frozenset({"policy", "addendum", "memo", "handbook", "calendar"})


def authority(doc: DocMeta) -> int:
    return 1 if doc.doc_type in HIGH_AUTHORITY else 0


def scope(doc: DocMeta) -> int:
    """Region-specific documents outrank GLOBAL ones."""
    return 0 if doc.is_global else 1


def recency(doc: DocMeta) -> date:
    return doc.effective_date


TESTS: list[tuple[str, Callable[[DocMeta], object]]] = [
    ("authority", authority),
    ("scope", scope),
    ("recency", recency),
]


@dataclass(frozen=True)
class Claim:
    claim_id: str
    rule_id: str
    rule: str
    value: str
    statement: str
    chunk: Chunk
    doc: DocMeta


@dataclass
class Resolution:
    rule_id: str
    rule: str
    winners: list[Claim]
    conflict: bool
    decided_by: str | None
    upcoming: list[Claim] = field(default_factory=list)

    @property
    def cited_claims(self) -> list[Claim]:
        return self.winners + self.upcoming


# --- value comparison ------------------------------------------------------------------------

_UNITS = {
    "day", "week", "month", "year", "hour", "minute", "%", "percent", "time",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}
_FILLER = {"up", "to", "a", "an", "the", "of", "per", "each", "every", "for", "in", "and", "or", "paid",
           "working", "calendar", "consecutive", "at", "most", "maximum", "max", "only"}


def _singular(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _signature(value: str) -> tuple[frozenset, frozenset]:
    tokens = re.findall(r"\d+(?:[.,]\d+)?|[a-z%]+", value.lower().replace(",", ""))
    numbers: set[tuple[float, str | None]] = set()
    for i, tok in enumerate(tokens):
        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
            unit = None
            for nxt in tokens[i + 1 : i + 4]:
                word = _singular(nxt)
                if word in _UNITS:
                    unit = word
                    break
                if re.fullmatch(r"\d+(?:\.\d+)?", nxt):
                    break
            numbers.add((float(tok), unit))
    words = frozenset(_singular(t) for t in tokens if t.isalpha() and t not in _FILLER)
    return frozenset(numbers), words


def values_equal(a: str, b: str) -> bool:
    if a.strip().lower() == b.strip().lower():
        return True
    nums_a, words_a = _signature(a)
    nums_b, words_b = _signature(b)
    if nums_a or nums_b:
        if {n for n, _ in nums_a} != {n for n, _ in nums_b}:
            return False
        units_a = {(n, u) for n, u in nums_a if u}
        units_b = {(n, u) for n, u in nums_b if u}
        nums_with_both = {n for n, _ in units_a} & {n for n, _ in units_b}
        return all(
            {u for m, u in units_a if m == n} & {u for m, u in units_b if m == n} for n in nums_with_both
        )
    # No numbers or dates on either side: these are procedural statements, which documents
    # paraphrase freely. Only a negation on one side ("deducted" vs "not deducted") is treated
    # as a different value; word-level differences are not.
    return _negated(a) == _negated(b)


_NEGATION = re.compile(r"\b(not|no|never|cannot|can't|isn't|aren't|doesn't|don't|without|none)\b", re.I)


def _negated(value: str) -> bool:
    return bool(_NEGATION.search(value))


def all_agree(claims: list[Claim]) -> bool:
    return all(values_equal(claims[0].value, c.value) for c in claims[1:])


# --- resolution ------------------------------------------------------------------------------

def narrow(claims: list[Claim]) -> tuple[list[Claim], bool, str | None]:
    """Apply authority, then scope, then recency. Returns (remaining, conflict, decided_by).

    The value is decided by the first test after which the remaining claims agree; the
    remaining tests still run so that only the highest-precedence agreeing documents are cited.
    """
    remaining = list(claims)
    decided_by: str | None = "agreement" if all_agree(remaining) else None
    for name, key in TESTS:
        best = max(key(c.doc) for c in remaining)
        narrowed = [c for c in remaining if key(c.doc) == best]
        if decided_by is None and all_agree(narrowed) and len(narrowed) < len(remaining):
            decided_by = name
        remaining = narrowed
    conflict = not all_agree(remaining)
    return remaining, conflict, (None if conflict else decided_by)


def resolve_rule(current: list[Claim], upcoming: list[Claim]) -> Resolution:
    reference = current[0] if current else upcoming[0]
    if current:
        winners, conflict, decided_by = narrow(current)
    else:
        winners, conflict, decided_by = [], False, None
    mentions: list[Claim] = []
    for claim in sorted(upcoming, key=lambda c: (c.doc.effective_date, c.claim_id)):
        on_date = claim.doc.effective_date
        pool = current + [u for u in upcoming if u.doc.effective_date <= on_date]
        future_winners, _, _ = narrow(pool)
        if claim not in future_winners:
            continue
        if winners and not conflict and values_equal(winners[0].value, claim.value):
            continue  # no change to report
        mentions.append(claim)
    return Resolution(reference.rule_id, reference.rule, winners, conflict, decided_by, mentions)


def resolve(claims: list[Claim], reference_date: date, include_upcoming: bool = True) -> list[Resolution]:
    """Group claims by rule and resolve each group. Claims from documents that are not yet
    effective on the reference date are treated as upcoming changes."""
    groups: dict[str, list[Claim]] = {}
    for claim in claims:
        groups.setdefault(claim.rule_id, []).append(claim)
    resolutions: list[Resolution] = []
    for rule_id in groups:
        group = groups[rule_id]
        current = [c for c in group if c.doc.effective_date <= reference_date]
        upcoming = [c for c in group if c.doc.effective_date > reference_date] if include_upcoming else []
        if not current and not upcoming:
            continue
        resolutions.append(resolve_rule(current, upcoming))
    return resolutions
