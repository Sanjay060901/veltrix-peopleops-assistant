"""Content-addressed ingestion cache.

Each document's parsed chunks, extracted holidays and embedding vectors are stored under a key
derived from the SHA-256 of the file's bytes. Unchanged files are never re-parsed, re-OCR'd or
re-embedded, across reloads and restarts; a changed byte gives a new key, so stale entries are
never used. Bump PARSER_VERSION whenever loaders or chunking change their output.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date
from pathlib import Path

import numpy as np

from app.ingestion.holidays import Holiday
from app.ingestion.models import Chunk

log = logging.getLogger(__name__)

PARSER_VERSION = "1"

# (size, mtime_ns) -> sha256, so unchanged files are not re-read just to be hashed.
_hash_memo: dict[str, tuple[int, int, str]] = {}


def file_sha256(path: str | Path) -> str:
    path = str(path)
    st = os.stat(path)
    memo = _hash_memo.get(path)
    if memo and memo[0] == st.st_size and memo[1] == st.st_mtime_ns:
        return memo[2]
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    sha = digest.hexdigest()
    _hash_memo[path] = (st.st_size, st.st_mtime_ns, sha)
    return sha


def entry_key(doc_id: str, sha: str) -> str:
    return hashlib.sha256(f"{PARSER_VERSION}|{doc_id}|{sha}".encode()).hexdigest()[:32]


def vector_key(key: str, title: str, model: str) -> str:
    return hashlib.sha256(f"{key}|{title}|{model}".encode()).hexdigest()[:32]


class IngestCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str, suffix: str) -> Path:
        return self.root / f"{key}{suffix}"

    def load(self, key: str) -> tuple[list[Chunk], list[Holiday], bool] | None:
        path = self._path(key, ".json")
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            chunks = [Chunk(**c) for c in data["chunks"]]
            holidays = [Holiday(h["doc_id"], date.fromisoformat(h["day"]), h["name"], h["kind"]) for h in data["holidays"]]
            return chunks, holidays, bool(data.get("ocr_used"))
        except Exception as exc:  # a corrupt entry is treated as a miss
            log.warning("ignoring unreadable cache entry %s: %s", path.name, exc)
            return None

    def save(self, key: str, chunks: list[Chunk], holidays: list[Holiday], ocr_used: bool) -> None:
        data = {
            "chunks": [c.__dict__ for c in chunks],
            "holidays": [{"doc_id": h.doc_id, "day": h.day.isoformat(), "name": h.name, "kind": h.kind} for h in holidays],
            "ocr_used": ocr_used,
        }
        self._atomic_write(self._path(key, ".json"), json.dumps(data).encode("utf-8"))

    def load_vectors(self, key: str, rows: int) -> np.ndarray | None:
        path = self._path(key, ".npy")
        if not path.is_file():
            return None
        try:
            vectors = np.load(path)
            return vectors if vectors.shape[0] == rows else None
        except Exception:
            return None

    def save_vectors(self, key: str, vectors: np.ndarray) -> None:
        tmp = self._path(key, ".npy.tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, vectors)
        os.replace(tmp, self._path(key, ".npy"))

    def prune(self, keep: set[str]) -> int:
        """Delete entries no current document uses."""
        removed = 0
        for path in self.root.glob("*"):
            if path.suffix in (".json", ".npy") and path.stem not in keep:
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
