"""Document eligibility (TASK §5.1 rule 1). Applied before any search runs.

A document the requester cannot read is removed here, so it can never be retrieved, packed into
a prompt or cited.
"""
from __future__ import annotations

from datetime import date
from typing import Iterable

from app.ingestion.models import DocMeta

ROLE_ACCESS: dict[str, frozenset[str]] = {
    "employee": frozenset({"all"}),
    "manager": frozenset({"all", "managers"}),
    "hr": frozenset({"all", "managers", "hr"}),
}


def can_read_doc(role: str, access_level: str) -> bool:
    """Access-control decision for documents. Unknown roles read only 'all' documents."""
    return access_level in ROLE_ACCESS.get(role, ROLE_ACCESS["employee"])


def region_matches(meta: DocMeta, region: str) -> bool:
    return meta.is_global or region.upper() in meta.regions


def readable_for(docs: Iterable[DocMeta], role: str, region: str) -> list[DocMeta]:
    """Documents the requester may read that are for their region (any effective date)."""
    return [d for d in docs if can_read_doc(role, d.access_level) and region_matches(d, region)]


def eligible_docs(docs: Iterable[DocMeta], role: str, region: str, reference_date: date) -> list[DocMeta]:
    return [d for d in readable_for(docs, role, region) if d.effective_date <= reference_date]


def upcoming_docs(docs: Iterable[DocMeta], role: str, region: str, reference_date: date) -> list[DocMeta]:
    """Readable, for the requester's region, but not yet effective on the reference date."""
    return [d for d in readable_for(docs, role, region) if d.effective_date > reference_date]
