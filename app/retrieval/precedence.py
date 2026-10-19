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


# --- rule grouping safety net ------------------------------------------------------------------

_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"}
_DESC_STOP = {"the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "per", "by", "with", "at",
              "employee", "employees", "staff", "rule", "policy", "allowance", "entitlement", "number"}


def value_kinds(value: str) -> frozenset[str]:
    """What kind of quantity a value is: its units (months collapse to 'date'), or 'text'."""
    numbers, _ = _signature(value)
    kinds = {("date" if u in _MONTHS else u) for _, u in numbers if u}
    if numbers and not kinds:
        kinds = {"number"}
    return frozenset(kinds or {"text"})


def description_overlap(a: str, b: str) -> float:
    """Overlap coefficient of the content words of two rule descriptions."""
    def words(text: str) -> set[str]:
        return {_singular(w) for w in re.findall(r"[a-z]+", text.lower()) if w not in _DESC_STOP and len(w) > 2}

    wa, wb = words(a), words(b)
    if len(wa) < 2 or len(wb) < 2:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


# Words that make two otherwise similar rules different ("primary" vs "secondary caregiver").
_CONTRASTS = [
    {"primary", "secondary"}, {"minimum", "maximum"}, {"before", "after"}, {"first", "last"},
    {"paid", "unpaid"}, {"birth", "adoptive"}, {"full", "part"}, {"less", "more"}, {"start", "end"},
    {"annual", "sick"}, {"maternity", "paternity"}, {"employee", "manager"},
]


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _contrasting(a: str, b: str) -> bool:
    wa, wb = set(re.findall(r"[a-z]+", a.lower())), set(re.findall(r"[a-z]+", b.lower()))
    return any((wa & pair) and (wb & pair) and (wa & pair) != (wb & pair) for pair in _CONTRASTS)


def descriptions_compatible(a: str, b: str) -> bool:
    """Hard guards: numbers named in the descriptions must match (accrual bands '<2 years' vs
    '2-5 years'), and no contrasting qualifier may differ."""
    return _numbers(a) == _numbers(b) and not _contrasting(a, b)


def merge_similar_rules(
    claims: list[Claim],
    similarity: Callable[[str, str], float] | None = None,
    similarity_threshold: float = 0.93,
    overlap_threshold: float = 1.0,
) -> list[Claim]:
    """Give one rule_id to groups the extractor split but that state the same rule.

    Precedence is only correct when competing claims share a rule_id, so this does not rely on the
    LLM alone. It is conservative: a wrong merge would be worse than a missed one. Two groups are
    merged only when all hold: one description's content words contain the other's, or their
    embeddings are near-identical; numbers in the descriptions match and no contrasting qualifier
    differs; the values are the same kind of quantity; and no document appears in both groups
    (a document does not contradict itself, so two values from one document are two rules).
    Deterministic: groups are processed in sorted order.
    """
    groups: dict[str, list[Claim]] = {}
    for claim in claims:
        groups.setdefault(claim.rule_id, []).append(claim)
    ids = sorted(groups)
    parent = {rid: rid for rid in ids}

    def find(rid: str) -> str:
        while parent[rid] != rid:
            parent[rid] = parent[parent[rid]]
            rid = parent[rid]
        return rid

    members = {rid: list(groups[rid]) for rid in ids}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            ga, gb = members[ra], members[rb]
            if {c.doc.doc_id for c in ga} & {c.doc.doc_id for c in gb}:
                continue
            kinds_a = frozenset().union(*(value_kinds(c.value) for c in ga))
            kinds_b = frozenset().union(*(value_kinds(c.value) for c in gb))
            if not kinds_a & kinds_b:
                continue
            desc_a, desc_b = ga[0].rule, gb[0].rule
            if not descriptions_compatible(desc_a, desc_b):
                continue
            # One description containing all the other's content words ("work from home days per
            # week" vs the same "for the Austin office") or near-identical meaning.
            same = description_overlap(desc_a, desc_b) >= overlap_threshold
            if not same and similarity is not None:
                same = similarity(desc_a, desc_b) >= similarity_threshold
            if same:
                keep, drop = sorted((ra, rb))
                parent[drop] = keep
                members[keep] = members[keep] + members.pop(drop)
    return [
        claim if find(claim.rule_id) == claim.rule_id
        else Claim(claim.claim_id, find(claim.rule_id), groups[find(claim.rule_id)][0].rule, claim.value,
                   claim.statement, claim.chunk, claim.doc)
        for claim in claims
    ]


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
