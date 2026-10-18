"""Ingestion: manifest + corpus -> parsed documents, chunks, holiday records and search index.

Runs automatically at startup and again whenever a file's content changes. Nothing here knows
any corpus value; everything is derived from CORPUS_DIR and MANIFEST_PATH. With a cache,
documents whose SHA-256 is unchanged are taken from the cache instead of being re-parsed,
re-OCR'd and re-embedded.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.data import repositories as repo
from app.data.db import Database
from app.ingestion.cache import IngestCache, entry_key, file_sha256, vector_key
from app.ingestion.chunking import chunk_document
from app.ingestion.holidays import Holiday, extract_holidays
from app.ingestion.loaders import load_blocks
from app.ingestion.manifest import load_manifest
from app.ingestion.models import Chunk, DocMeta, ParsedDoc
from app.retrieval.embeddings import Embedder
from app.retrieval.index import CorpusIndex

log = logging.getLogger(__name__)


@dataclass
class DocStatus:
    doc_id: str
    file: str
    sha256: str
    chunks: int
    source: str  # parsed | cache | missing | failed


@dataclass
class Corpus:
    docs: dict[str, DocMeta]
    chunks: list[Chunk]
    index: CorpusIndex
    holidays: list[Holiday] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    status: list[DocStatus] = field(default_factory=list)
    version: str = ""  # hash over every document's content hash and manifest entry

    def chunk(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def __post_init__(self) -> None:
        self._by_id = {c.chunk_id: c for c in self.chunks}


def parse_document(corpus_dir: Path, meta: DocMeta) -> ParsedDoc:
    blocks, ocr_used = load_blocks(corpus_dir / meta.file)
    return ParsedDoc(meta=meta, blocks=blocks, ocr_used=ocr_used)


def _parse(root: Path, meta: DocMeta) -> tuple[list[Chunk], list[Holiday], bool]:
    parsed = parse_document(root, meta)
    doc_chunks = chunk_document(parsed)
    found = extract_holidays(parsed) if meta.doc_type == "calendar" else []
    if not doc_chunks:
        log.warning("no text extracted doc=%s ocr=%s", meta.doc_id, parsed.ocr_used)
    return doc_chunks, found, parsed.ocr_used


def ingest(
    corpus_dir: str,
    manifest_path: str,
    db: Database | None,
    embedder: Embedder | None,
    cache: IngestCache | None = None,
    embedding_model: str = "",
) -> Corpus:
    started = time.monotonic()
    metas = load_manifest(manifest_path)
    root = Path(corpus_dir)
    docs: dict[str, DocMeta] = {}
    chunks: list[Chunk] = []
    holidays: list[Holiday] = []
    failed: list[str] = []
    status: list[DocStatus] = []
    vectors: list[np.ndarray] = []
    keep: set[str] = set()

    for meta in metas:
        path = root / meta.file
        if not path.is_file():
            log.warning("manifest file missing doc=%s", meta.doc_id)
            failed.append(meta.doc_id)
            status.append(DocStatus(meta.doc_id, meta.file, "", 0, "missing"))
            continue
        sha = file_sha256(path)
        key = entry_key(f"{meta.doc_id}|{meta.doc_type}", sha)
        cached = cache.load(key) if cache else None
        if cached is not None:
            doc_chunks, found, ocr_used = cached
            source = "cache"
        else:
            try:
                doc_chunks, found, ocr_used = _parse(root, meta)
            except Exception as exc:
                log.exception("failed to parse doc=%s: %s", meta.doc_id, exc)
                failed.append(meta.doc_id)
                status.append(DocStatus(meta.doc_id, meta.file, sha, 0, "failed"))
                continue
            if cache:
                cache.save(key, doc_chunks, found, ocr_used)
            source = "parsed"
        keep.add(key)
        docs[meta.doc_id] = meta
        chunks.extend(doc_chunks)
        holidays.extend(found)
        status.append(DocStatus(meta.doc_id, meta.file, sha, len(doc_chunks), source))
        log.info("ingested doc=%s sha256=%s chunks=%s ocr=%s source=%s", meta.doc_id, sha[:12], len(doc_chunks), ocr_used, source)

        if embedder is not None and doc_chunks:
            vkey = vector_key(key, meta.title, embedding_model)
            doc_vectors = cache.load_vectors(vkey, len(doc_chunks)) if cache else None
            if doc_vectors is None:
                try:
                    doc_vectors = embedder.embed_documents([CorpusIndex.text_for(meta, c) for c in doc_chunks])
                except Exception as exc:  # degrade to keyword retrieval for the whole corpus
                    log.warning("embedding failed, keyword retrieval only: %s", exc)
                    embedder, vectors = None, []
                    continue
                if cache:
                    cache.save_vectors(vkey, doc_vectors)
            keep.add(vkey)
            vectors.append(doc_vectors)

    if db is not None:
        db.ensure_schema()
        with db.transaction() as conn:
            repo.replace_holidays(conn, [(h.doc_id, h.day.isoformat(), h.name, h.kind) for h in holidays])

    matrix = np.vstack(vectors) if embedder is not None and vectors else None
    index = CorpusIndex(docs, chunks, embedder, vectors=matrix)
    if cache:
        cache.prune(keep)
    version = hashlib.sha256(
        "\n".join(f"{s.doc_id}|{s.file}|{s.sha256}|{docs[s.doc_id].__dict__ if s.doc_id in docs else ''}" for s in status).encode()
    ).hexdigest()
    parsed = sum(1 for s in status if s.source == "parsed")
    log.info(
        "ingestion finished docs=%s parsed=%s cached=%s chunks=%s holidays=%s failed=%s corpus_version=%s seconds=%.1f",
        len(docs), parsed, len(docs) - parsed, len(chunks), len(holidays), len(failed), version[:12], time.monotonic() - started,
    )
    return Corpus(docs=docs, chunks=chunks, index=index, holidays=holidays, failed=failed, status=status, version=version)
