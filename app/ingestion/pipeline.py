"""Ingestion: manifest + corpus -> parsed documents, chunks, holiday records and search index.

Runs automatically at startup. Nothing here knows any corpus value; everything is derived from
CORPUS_DIR and MANIFEST_PATH, so a modified corpus is picked up on the next start.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.data import repositories as repo
from app.data.db import Database
from app.ingestion.chunking import chunk_document
from app.ingestion.holidays import Holiday, extract_holidays
from app.ingestion.loaders import load_blocks
from app.ingestion.manifest import load_manifest
from app.ingestion.models import Chunk, DocMeta, ParsedDoc
from app.retrieval.embeddings import Embedder
from app.retrieval.index import CorpusIndex

log = logging.getLogger(__name__)


@dataclass
class Corpus:
    docs: dict[str, DocMeta]
    chunks: list[Chunk]
    index: CorpusIndex
    holidays: list[Holiday] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def chunk(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def __post_init__(self) -> None:
        self._by_id = {c.chunk_id: c for c in self.chunks}


def parse_document(corpus_dir: Path, meta: DocMeta) -> ParsedDoc:
    blocks, ocr_used = load_blocks(corpus_dir / meta.file)
    return ParsedDoc(meta=meta, blocks=blocks, ocr_used=ocr_used)


def ingest(corpus_dir: str, manifest_path: str, db: Database | None, embedder: Embedder | None) -> Corpus:
    started = time.monotonic()
    metas = load_manifest(manifest_path)
    root = Path(corpus_dir)
    docs: dict[str, DocMeta] = {}
    chunks: list[Chunk] = []
    holidays: list[Holiday] = []
    failed: list[str] = []
    for meta in metas:
        path = root / meta.file
        if not path.is_file():
            log.warning("manifest file missing doc=%s", meta.doc_id)
            failed.append(meta.doc_id)
            continue
        try:
            parsed = parse_document(root, meta)
        except Exception as exc:
            log.exception("failed to parse doc=%s: %s", meta.doc_id, exc)
            failed.append(meta.doc_id)
            continue
        doc_chunks = chunk_document(parsed)
        if not doc_chunks:
            log.warning("no text extracted doc=%s ocr=%s", meta.doc_id, parsed.ocr_used)
        docs[meta.doc_id] = meta
        chunks.extend(doc_chunks)
        if meta.doc_type == "calendar":
            found = extract_holidays(parsed)
            holidays.extend(found)
            log.info("holidays extracted doc=%s count=%s", meta.doc_id, len(found))
        log.info("ingested doc=%s blocks=%s chunks=%s ocr=%s", meta.doc_id, len(parsed.blocks), len(doc_chunks), parsed.ocr_used)

    if db is not None:
        db.ensure_schema()
        with db.transaction() as conn:
            repo.replace_holidays(conn, [(h.doc_id, h.day.isoformat(), h.name, h.kind) for h in holidays])

    index = CorpusIndex(docs, chunks, embedder)
    log.info(
        "ingestion finished docs=%s chunks=%s holidays=%s failed=%s seconds=%.1f",
        len(docs), len(chunks), len(holidays), len(failed), time.monotonic() - started,
    )
    return Corpus(docs=docs, chunks=chunks, index=index, holidays=holidays, failed=failed)
