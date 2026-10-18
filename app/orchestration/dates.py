"""Date phrase resolution (TASK §8.4), deterministic.

Rules: dates are calendar dates; today is APP_TODAY; a date without a year is in the leave year;
weeks start on Monday; "next week" is Monday-Friday of the following week; "next Friday" is the
first Friday after today. Ambiguous phrases and past days are reported so the caller can ask.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from app.ingestion.holidays import MONTH_RE, MONTHS

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WD_RE = r"(mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:r|rs|rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)"


def _weekday_index(word: str) -> int:
    word = word.lower()
    for i, name in enumerate(WEEKDAYS):
        if name.startswith(word[:3]):
            return i
    raise ValueError(word)


@dataclass(frozen=True)
class DateRange:
    start: date | None
    end: date | None
    status: str  # ok | ambiguous | unknown
    reason: str = ""


def _ok(start: date, end: date | None = None) -> DateRange:
    return DateRange(start, end or start, "ok")


def resolve_phrase(text: str, today: date, default_year: int) -> DateRange:
    """Resolve one date phrase to a date or a range. Returns status 'unknown' for phrases this
    resolver does not handle (the caller may then use the LLM's reading)."""
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = re.sub(r"^(on|from|starting|start|until|till|to|through|for|the)\s+", "", t)
    if not t:
        return DateRange(None, None, "unknown")

    if t in ("today", "this day"):
        return _ok(today)
    if t == "tomorrow":
        return _ok(today + timedelta(days=1))
    if t in ("day after tomorrow", "the day after tomorrow"):
        return _ok(today + timedelta(days=2))
    if t in ("yesterday", "day before yesterday") or t.startswith("last "):
        return DateRange(None, None, "ambiguous", "that date has already passed")

    monday = today - timedelta(days=today.weekday())
    if t == "next week":
        start = monday + timedelta(days=7)
        return _ok(start, start + timedelta(days=4))
    if t == "this week":
        start = max(today, monday)
        end = monday + timedelta(days=4)
        if start > end:
            return DateRange(None, None, "ambiguous", "the working days of this week have passed")
        return _ok(start, end)
    if t in ("next month", "this month", "later this month", "sometime next week", "soon", "next few days"):
        return DateRange(None, None, "ambiguous", "that period is not specific enough")

    m = re.fullmatch(rf"(next|this|coming|this coming)?\s*{_WD_RE}", t)
    if m:
        target = _weekday_index(m.group(2))
        qualifier = m.group(1) or ""
        delta = (target - today.weekday()) % 7
        if qualifier == "next":
            delta = delta or 7  # first such weekday after today
        elif delta == 0:
            if qualifier:
                return _ok(today)
            return DateRange(None, None, "ambiguous", f"it is not clear whether you mean today or next {WEEKDAYS[target].title()}")
        return _ok(today + timedelta(days=delta))

    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        return _checked(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = re.fullmatch(rf"(\d{{1,2}})(?:st|nd|rd|th)?(?: of)? ({MONTH_RE})\.?,?(?: (\d{{4}}))?", t)
    if m:
        return _checked(int(m.group(3) or default_year), MONTHS[m.group(2)], int(m.group(1)))
    m = re.fullmatch(rf"({MONTH_RE})\.? (\d{{1,2}})(?:st|nd|rd|th)?,?(?: (\d{{4}}))?", t)
    if m:
        return _checked(int(m.group(3) or default_year), MONTHS[m.group(1)], int(m.group(2)))

    m = re.fullmatch(r"(\d{1,2})[/.](\d{1,2})(?:[/.](\d{2,4}))?", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else default_year
        if year < 100:
            year += 2000
        if a > 12 and b <= 12:
            return _checked(year, b, a)
        if b > 12 and a <= 12:
            return _checked(year, a, b)
        if a == b:
            return _checked(year, a, b)
        return DateRange(None, None, "ambiguous", f"'{text.strip()}' could be day/month or month/day")
    return DateRange(None, None, "unknown")


def _checked(year: int, month: int, day: int) -> DateRange:
    try:
        return _ok(date(year, month, day))
    except ValueError:
        return DateRange(None, None, "ambiguous", "that is not a valid calendar date")


def parse_iso(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


@dataclass(frozen=True)
class ResolvedDates:
    start: date | None
    end: date | None
    problem: str = ""  # non-empty: ask the user


def resolve_dates(
    start_text: str, end_text: str, start_iso: str, end_iso: str, today: date, default_year: int, allow_past: bool
) -> ResolvedDates:
    """Combine the deterministic resolver with the LLM's ISO reading.

    A phrase the resolver understands always wins over the LLM. Unknown phrases fall back to the
    LLM's ISO date. Ambiguous phrases and (unless allow_past) past dates produce a problem."""
    start_r = resolve_phrase(start_text, today, default_year) if start_text else DateRange(None, None, "unknown")
    end_r = resolve_phrase(end_text, today, default_year) if end_text else DateRange(None, None, "unknown")
    for r in (start_r, end_r):
        if r.status == "ambiguous":
            return ResolvedDates(None, None, r.reason)
    start = start_r.start if start_r.status == "ok" else parse_iso(start_iso)
    if end_text:
        end = end_r.end if end_r.status == "ok" else parse_iso(end_iso)
    else:
        # A range phrase ("next week") gives its own end; a single day ends where it starts.
        end = start_r.end if start_r.status == "ok" else (parse_iso(end_iso) or start)
    if start and end and end < start and start_r.status == "ok" and not end_text:
        end = start
    if not allow_past and start and start < today:
        return ResolvedDates(None, None, f"{start.strftime('%a')} {start.day} {start.strftime('%B %Y')} has already passed")
    return ResolvedDates(start, end)
