"""Content-hash change detection and the ingestion cache."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from app.ingestion import pipeline
from app.ingestion.cache import IngestCache, file_sha256
from app.ingestion.watcher import corpus_fingerprint
from app.main import reload_if_changed
from tests.conftest import make_client
from tests.fakes import FakeLLM


class CountingEmbedder:
    """Deterministic stand-in for the embedding model that counts what it embeds."""

    def __init__(self):
        self.embedded = 0

    def embed_documents(self, texts):
        self.embedded += len(texts)
        return np.array([[len(t) % 7 + 1.0, 1.0, 0.5] for t in texts], dtype=np.float32)

    def embed_query(self, text):
        return np.array([1.0, 1.0, 0.5], dtype=np.float32)


@pytest.fixture
def parse_counter(monkeypatch):
    seen: list[str] = []
    real = pipeline._parse

    def counting(root, meta):
        seen.append(meta.doc_id)
        return real(root, meta)

    monkeypatch.setattr(pipeline, "_parse", counting)
    return seen


def _ingest(settings, db, cache_dir, embedder=None):
    return pipeline.ingest(settings.corpus_dir, settings.manifest_path, db, embedder,
                           cache=IngestCache(cache_dir), embedding_model="test-model")


def _touch(path: Path) -> None:
    later = time.time() + 5
    os.utime(path, (later, later))


def test_sha256_matches_hashlib(data_dir):
    path = data_dir / "corpus" / "faq.html"
    assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_second_ingest_parses_nothing(settings, db, tmp_path, parse_counter):
    first = _ingest(settings, db, tmp_path / "c")
    assert len(parse_counter) == len(first.docs)
    parse_counter.clear()
    second = _ingest(settings, db, tmp_path / "c")
    assert parse_counter == []
    assert all(s.source == "cache" for s in second.status)
    assert [c.__dict__ for c in second.chunks] == [c.__dict__ for c in first.chunks]
    assert second.holidays == first.holidays and second.version == first.version


def test_only_the_changed_file_is_parsed(settings, db, tmp_path, data_dir, parse_counter):
    _ingest(settings, db, tmp_path / "c")
    parse_counter.clear()
    md = data_dir / "corpus" / "addendum_de.md"
    md.write_text(md.read_text(encoding="utf-8").replace("5 volunteer days", "7 volunteer days"), encoding="utf-8")
    corpus = _ingest(settings, db, tmp_path / "c")
    assert parse_counter == ["VOL-ADD-DE"]
    assert any("7 volunteer days" in c.text for c in corpus.chunks)


def test_cached_result_equals_uncached(settings, db, tmp_path):
    cached_run = _ingest(settings, db, tmp_path / "c")
    cached_run = _ingest(settings, db, tmp_path / "c")
    plain = pipeline.ingest(settings.corpus_dir, settings.manifest_path, db, None)
    assert [c.__dict__ for c in cached_run.chunks] == [c.__dict__ for c in plain.chunks]
    assert cached_run.holidays == plain.holidays


def test_embeddings_are_cached_per_document(settings, db, tmp_path, data_dir):
    emb = CountingEmbedder()
    first = _ingest(settings, db, tmp_path / "c", emb)
    total = emb.embedded
    assert total == len(first.chunks)
    _ingest(settings, db, tmp_path / "c", emb)
    assert emb.embedded == total  # nothing re-embedded
    md = data_dir / "corpus" / "expenses_b.md"
    md.write_text(md.read_text(encoding="utf-8").replace("45 days", "50 days"), encoding="utf-8")
    third = _ingest(settings, db, tmp_path / "c", emb)
    changed = [c for c in third.chunks if c.doc_id == "EXP-HB"]
    assert emb.embedded == total + len(changed)
    assert third.index._vectors is not None and third.index._vectors.shape[0] == len(third.chunks)


def test_title_change_re_embeds_but_does_not_re_parse(settings, db, tmp_path, data_dir, parse_counter):
    emb = CountingEmbedder()
    _ingest(settings, db, tmp_path / "c", emb)
    before = emb.embedded
    parse_counter.clear()
    entries = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    for e in entries:
        if e["doc_id"] == "EXP-POL":
            e["title"] = "Expense Policy (renamed)"
    (data_dir / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    _ingest(settings, db, tmp_path / "c", emb)
    assert parse_counter == [] and emb.embedded == before + 1


def test_doc_type_change_re_parses(settings, db, tmp_path, data_dir, parse_counter):
    _ingest(settings, db, tmp_path / "c")
    parse_counter.clear()
    entries = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    for e in entries:
        if e["doc_id"] == "CAL-DE-2026":
            e["doc_type"] = "policy"  # no longer a calendar: no holidays should be extracted
    (data_dir / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    corpus = _ingest(settings, db, tmp_path / "c")
    assert parse_counter == ["CAL-DE-2026"]
    assert not any(h.doc_id == "CAL-DE-2026" for h in corpus.holidays)


def test_corrupt_cache_entry_is_a_miss(settings, db, tmp_path, parse_counter):
    _ingest(settings, db, tmp_path / "c")
    for entry in (tmp_path / "c").glob("*.json"):
        entry.write_text("{not json", encoding="utf-8")
    parse_counter.clear()
    corpus = _ingest(settings, db, tmp_path / "c")
    assert len(parse_counter) == len(corpus.docs)


def test_prune_removes_entries_of_removed_documents(settings, db, tmp_path, data_dir):
    _ingest(settings, db, tmp_path / "c")
    before = len(list((tmp_path / "c").glob("*.json")))
    entries = [e for e in json.loads((data_dir / "manifest.json").read_text(encoding="utf-8")) if e["doc_id"] != "VOL-V3"]
    (data_dir / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    _ingest(settings, db, tmp_path / "c")
    assert len(list((tmp_path / "c").glob("*.json"))) == before - 1


def test_version_changes_only_with_content(settings, db, tmp_path, data_dir):
    v1 = _ingest(settings, db, tmp_path / "c").version
    _touch(data_dir / "corpus" / "faq.html")
    assert _ingest(settings, db, tmp_path / "c").version == v1
    (data_dir / "corpus" / "faq.html").write_text("<h1>Changed</h1><p>New text.</p>", encoding="utf-8")
    assert _ingest(settings, db, tmp_path / "c").version != v1


# --- watcher: content, not timestamps ----------------------------------------------------------

def test_touching_a_file_does_not_trigger_reload(settings, data_dir):
    with make_client(settings, FakeLLM()) as client:
        _touch(data_dir / "corpus" / "faq.html")
        _touch(data_dir / "manifest.json")
        assert reload_if_changed(client.app.state.veltrix) is False


def test_copying_identical_bytes_does_not_trigger_reload(settings, data_dir, tmp_path):
    with make_client(settings, FakeLLM()) as client:
        src = data_dir / "corpus" / "vol_v1.pdf"
        shutil.copy(src, tmp_path / "copy.pdf")
        shutil.copy(tmp_path / "copy.pdf", src)
        _touch(src)
        assert reload_if_changed(client.app.state.veltrix) is False


def test_fingerprint_is_content_based(data_dir):
    corpus, manifest = str(data_dir / "corpus"), str(data_dir / "manifest.json")
    base = corpus_fingerprint(corpus, manifest)
    _touch(data_dir / "corpus" / "faq.html")
    assert corpus_fingerprint(corpus, manifest) == base
    path = data_dir / "corpus" / "faq.html"
    path.write_bytes(path.read_bytes() + b" ")
    assert corpus_fingerprint(corpus, manifest) != base


def test_app_restart_uses_cache(settings, data_dir, tmp_path, parse_counter):
    from dataclasses import replace

    cached = replace(settings, cache_dir=str(tmp_path / "c"))
    with make_client(cached, FakeLLM()):
        pass
    parse_counter.clear()
    with make_client(cached, FakeLLM()) as client:
        assert parse_counter == []
        assert client.post("/chat", headers={"X-Employee-Id": "X201"},
                           json={"session_id": "s", "message": "How many volunteer days do I get?"}).json()["route"] == "rag"


def test_inspect_command_prints_hashes(settings, data_dir, tmp_path, monkeypatch, capsys):
    from app.ingestion import inspect

    for key, value in {"CORPUS_DIR": settings.corpus_dir, "MANIFEST_PATH": settings.manifest_path,
                       "CACHE_DIR": str(tmp_path / "c")}.items():
        monkeypatch.setenv(key, value)
    (data_dir / "corpus" / "stray.txt").write_text("not listed", encoding="utf-8")
    assert inspect.main() == 0
    out = capsys.readouterr().out
    assert file_sha256(data_dir / "corpus" / "faq.html")[:16] in out
    assert "VOL-ADD-DE" in out and "stray.txt" in out and "corpus fingerprint" in out
