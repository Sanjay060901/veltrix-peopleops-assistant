"""Manifest loading and validation."""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from app.ingestion.models import DocMeta

log = logging.getLogger(__name__)

VALID_DOC_TYPES = {"policy", "addendum", "memo", "handbook", "calendar", "faq", "wiki"}
VALID_ACCESS = {"all", "managers", "hr"}


def load_manifest(path: str | Path) -> list[DocMeta]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    docs: list[DocMeta] = []
    seen: set[str] = set()
    for entry in raw:
        try:
            meta = DocMeta(
                doc_id=str(entry["doc_id"]),
                file=str(entry["file"]),
                title=str(entry.get("title") or entry["doc_id"]),
                doc_type=str(entry["doc_type"]).lower(),
                regions=tuple(str(r).upper() for r in entry.get("regions") or ["GLOBAL"]),
                effective_date=date.fromisoformat(str(entry["effective_date"])),
                access_level=str(entry.get("access_level", "all")).lower(),
            )
        except (KeyError, ValueError) as exc:
            log.warning("skipping malformed manifest entry: %s", exc)
            continue
        if meta.doc_id in seen:
            log.warning("duplicate doc_id in manifest, keeping first: %s", meta.doc_id)
            continue
        if meta.doc_type not in VALID_DOC_TYPES:
            log.warning("unknown doc_type %s for %s", meta.doc_type, meta.doc_id)
        if meta.access_level not in VALID_ACCESS:
            # Unknown access levels are treated as the most restrictive (hr) rather than open.
            log.warning("unknown access_level %s for %s, treating as hr", meta.access_level, meta.doc_id)
            meta = DocMeta(**{**meta.__dict__, "access_level": "hr"})
        seen.add(meta.doc_id)
        docs.append(meta)
    return docs
