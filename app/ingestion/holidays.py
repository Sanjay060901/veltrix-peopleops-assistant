"""Holiday extraction from calendar documents into structured records.

Works on any calendar document regardless of source format: it looks for tables (or lines)
whose cells parse as dates, and classifies each table as office-closure holidays or as
optional/restricted holidays (which are not office closures and do not reduce working days).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from app.ingestion.models import ParsedDoc

MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
            ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
            ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"), ("december", "dec"),
        ],
        start=1,
    )
    for m in names
}
MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_RE})\.?,?\s+(\d{{4}})\b", re.I)
_MDY = re.compile(rf"\b({MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_OPTIONAL_WORDS = re.compile(r"\b(restricted|optional|floating|elective)\b", re.I)


@dataclass(frozen=True)
class Holiday:
    doc_id: str
    day: date
    name: str
    kind: str  # "closure" | "optional"


def parse_document_date(text: str) -> date | None:
    """Parse a written date such as '15 January 2026', 'January 15, 2026' or '2026-01-15'."""
    for pattern, order in ((_ISO, "ymd"), (_DMY, "dmy"), (_MDY, "mdy")):
        m = pattern.search(text)
        if not m:
            continue
        try:
            if order == "ymd":
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if order == "dmy":
                return date(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
            return date(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)))
        except ValueError:
            return None
    return None


_WEEKDAYS = re.compile(r"^(mon|tues?|wed(nes)?|thu(rs)?|fri|sat(ur)?|sun)(day)?\.?$", re.I)


def _name_from_cells(cells: list[str], date_col: int, header: list[str] | None) -> str:
    if header:
        for i, h in enumerate(header):
            if i != date_col and re.search(r"holiday|name|occasion|festival|event", h, re.I) and i < len(cells):
                return cells[i]
    others = [c for i, c in enumerate(cells) if i != date_col and c and not _WEEKDAYS.match(c.strip())]
    return others[-1] if others else ""


def extract_holidays(doc: ParsedDoc) -> list[Holiday]:
    holidays: list[Holiday] = []
    heading = ""
    for block in doc.blocks:
        if block.kind == "heading":
            heading = block.text
            continue
        if block.kind == "table" and block.rows:
            rows = block.rows
            first_dates = [parse_document_date(c) for c in rows[0]]
            header = None if any(first_dates) else rows[0]
            body = rows if header is None else rows[1:]
            # The date column is the one where most cells parse as dates.
            width = max(len(r) for r in body) if body else 0
            scores = [sum(1 for r in body if i < len(r) and parse_document_date(r[i])) for i in range(width)]
            if not scores or max(scores) == 0:
                continue
            date_col = scores.index(max(scores))
            context = " ".join([heading] + (header or []))
            kind = "optional" if _OPTIONAL_WORDS.search(context) else "closure"
            for row in body:
                if date_col >= len(row):
                    continue
                day = parse_document_date(row[date_col])
                if day:
                    holidays.append(Holiday(doc.meta.doc_id, day, _name_from_cells(row, date_col, header), kind))
        elif block.kind == "paragraph":
            # Calendars written as lines: "1 January 2026 - New Year's Day".
            text = block.text.lstrip("- ").strip()
            day = parse_document_date(text[:30])
            if day and len(text) < 120 and parse_document_date(text) == day and text.find(str(day.year)) < 30:
                name = re.sub(r"^.*?\d{4}\s*[-:–,|]?\s*", "", text).strip()
                kind = "optional" if _OPTIONAL_WORDS.search(heading + " " + text) else "closure"
                holidays.append(Holiday(doc.meta.doc_id, day, name, kind))
    # De-duplicate (same date and name listed twice).
    unique: dict[tuple[date, str], Holiday] = {}
    for h in holidays:
        unique.setdefault((h.day, h.name), h)
    return sorted(unique.values(), key=lambda h: (h.day, h.name))
