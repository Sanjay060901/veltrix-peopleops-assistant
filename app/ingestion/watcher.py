"""Detects content changes to the manifest or corpus files so the service can re-ingest.

The fingerprint is the SHA-256 of every file's bytes. Files whose size and modification time
are unchanged are not re-read (the hash is memoised on those), so the periodic check is cheap.
A file that is only touched, or copied over with identical bytes, does not trigger a reload.
Added, removed, renamed or edited files do.
"""
from __future__ import annotations

from pathlib import Path

from app.ingestion.cache import file_sha256


def corpus_fingerprint(corpus_dir: str, manifest_path: str) -> tuple:
    entries: list[tuple[str, str]] = []
    for path in [Path(manifest_path)] + sorted(Path(corpus_dir).glob("*")):
        try:
            if path.is_file():
                entries.append((path.name, file_sha256(path)))
        except OSError:
            entries.append((path.name, "unreadable"))
    return tuple(entries)
