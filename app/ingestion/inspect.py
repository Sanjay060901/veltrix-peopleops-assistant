"""Local admin view of the corpus: SHA-256 per document and whether it is already cached.

    python -m app.ingestion.inspect            # uses CORPUS_DIR / MANIFEST_PATH / CACHE_DIR
    docker compose exec assistant python -m app.ingestion.inspect

This is deliberately a command-line tool and not an HTTP endpoint: listing documents over the
API would reveal that restricted documents exist.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from app.config import load_settings
from app.ingestion.cache import IngestCache, entry_key, file_sha256
from app.ingestion.manifest import load_manifest
from app.ingestion.watcher import corpus_fingerprint


def main() -> int:
    s = load_settings()
    cache = IngestCache(s.cache_dir) if s.cache_dir else None
    root = Path(s.corpus_dir)
    metas = load_manifest(s.manifest_path)
    listed = {m.file for m in metas}
    print(f"manifest  {s.manifest_path}  sha256={file_sha256(s.manifest_path)}")
    print(f"{'doc_id':28} {'file':42} {'sha256':16} {'bytes':>8}  cached")
    for m in metas:
        path = root / m.file
        if not path.is_file():
            print(f"{m.doc_id:28} {m.file:42} {'-':16} {'-':>8}  MISSING FILE")
            continue
        sha = file_sha256(path)
        hit = bool(cache and (cache.root / f"{entry_key(f'{m.doc_id}|{m.doc_type}', sha)}.json").is_file())
        print(f"{m.doc_id:28} {m.file:42} {sha[:16]} {path.stat().st_size:>8}  {'yes' if hit else 'no'}")
    extra = sorted(p.name for p in root.glob("*") if p.is_file() and p.name not in listed)
    if extra:
        print("\nfiles not in the manifest (ignored):", ", ".join(extra))
    fingerprint = hashlib.sha256(repr(corpus_fingerprint(s.corpus_dir, s.manifest_path)).encode()).hexdigest()
    print(f"\ncorpus fingerprint  {fingerprint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
