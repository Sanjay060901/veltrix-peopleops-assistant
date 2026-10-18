"""Working-day calculation (TASK §6.2).

Working days are Monday-Friday dates that are not office-closure holidays for the requester's
region. Holidays come from the calendar document that applies under the §5 rules, read from the
holiday_calendar table filled at ingestion.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

from app.data import repositories as repo
from app.ingestion.models import DocMeta
from app.retrieval.eligibility import eligible_docs
from app.retrieval.precedence import TESTS


def count_working_days(start: date, end: date, holidays: Iterable[date]) -> int:
    return len(working_dates(start, end, holidays))


def working_dates(start: date, end: date, holidays: Iterable[date]) -> list[date]:
    off = set(holidays)
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in off:
            days.append(current)
        current += timedelta(days=1)
    return days


@dataclass(frozen=True)
class CalendarChoice:
    doc_ids: tuple[str, ...]
    holidays: frozenset[date]


def choose_calendar(
    conn: sqlite3.Connection, docs: Iterable[DocMeta], role: str, region: str, year: int, reference_date: date
) -> CalendarChoice:
    """Pick the holiday calendar(s) that apply for `year`.

    Candidates are calendar documents eligible on the reference date (the leave start date)
    that list holidays in that year. Among them the §5 precedence tests apply (scope, then
    recency). Calendars still tied after all tests are combined.
    """
    calendars = [d for d in eligible_docs(docs, role, region, reference_date) if d.doc_type == "calendar"]
    records = repo.holidays_for_docs(conn, [d.doc_id for d in calendars])
    years: dict[str, set[int]] = {}
    for doc_id, day, _name in records:
        years.setdefault(doc_id, set()).add(day.year)
    candidates = [d for d in calendars if year in years.get(d.doc_id, set())]
    if not candidates:
        return CalendarChoice((), frozenset())
    for _name, key in TESTS:
        best = max(key(d) for d in candidates)
        candidates = [d for d in candidates if key(d) == best]
    chosen = tuple(sorted(d.doc_id for d in candidates))
    days = frozenset(day for doc_id, day, _ in records if doc_id in chosen and day.year == year)
    return CalendarChoice(chosen, days)
