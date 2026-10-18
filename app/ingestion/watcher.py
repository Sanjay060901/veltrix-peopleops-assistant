"""Detects changes to the manifest or corpus files so the service can re-ingest without a restart.

A fingerprint of (name, size, mtime) for the manifest and every corpus file is compared on a
timer. Added, removed, renamed or edited files all change the fingerprint.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def corpus_fingerprint(corpus_dir: str, manifest_path: str) -> tuple:
    entries: list[tuple[str, int, int]] = []
    for path in [Path(manifest_path)] + sorted(Path(corpus_dir).glob("*")):
        try:
            if path.is_file():
                st = os.stat(path)
                entries.append((path.name, st.st_size, st.st_mtime_ns))
        except OSError:
            entries.append((path.name, -1, -1))
    return tuple(entries)
