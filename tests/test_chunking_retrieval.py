"""Chunking, locators, retrieval isolation, holiday formats, manifest validation, extra precedence cases."""
from __future__ import annotations

import itertools
import json
from datetime import date
from pathlib import Path

import pymupdf

from app.ingestion.chunking import MAX_CHUNK_TOKENS, chunk_document
from app.ingestion.holidays import extract_holidays
from app.ingestion.manifest import load_manifest
from app.ingestion.models import Block, Chunk, DocMeta, ParsedDoc
from app.llm.tokens import count_tokens
from app.retrieval.index import CorpusIndex, tokenize
from app.retrieval.precedence import Claim, resolve, values_equal


def meta(doc_id="D", doc_type="policy", regions=("GLOBAL",), eff="2026-01-01", access="all"):
    return DocMeta(doc_id, f"{doc_id}.md", f"Title {doc_id}", doc_type, tuple(regions), date.fromisoformat(eff), access)


# --- chunking ------------------------------------------------------------------------------------

def test_long_section_is_split_under_the_limit():
    sentence = "Employees must follow the rule described in this sentence carefully. "
    blocks = [Block("heading", text="1. Long rule", level=1)] + [Block("paragraph", text=sentence * 20) for _ in range(6)]
    chunks = chunk_document(ParsedDoc(meta(), blocks))
    assert len(chunks) > 1
    assert all(count_tokens(c.text) <= MAX_CHUNK_TOKENS for c in chunks)
    assert {c.locator for c in chunks} == {"1. Long rule"}


def test_long_table_is_split_with_header_repeated():
    rows = [["Date", "Holiday"]] + [[f"{d} January 2026", f"Holiday number {d} with a longer name"] for d in range(1, 32)] * 3
    blocks = [Block("heading", text="1. Holidays", level=1), Block("table", rows=rows)]
    chunks = chunk_document(ParsedDoc(meta(), blocks))
    assert len(chunks) > 1
    assert all(c.text.splitlines()[0] == "Date | Holiday" for c in chunks)
    assert all(count_tokens(c.text) <= MAX_CHUNK_TOKENS for c in chunks)


def test_document_without_headings_uses_page_locators(tmp_path: Path):
    from app.ingestion.loaders import load_blocks

    path = tmp_path / "memo.pdf"
    doc = pymupdf.open()
    for text in ("First page says employees may work remotely.", "Second page says core hours apply."):
        doc.new_page().insert_text((50, 100), text, fontsize=10, fontname="helv")
    doc.save(str(path))
    blocks, _ = load_blocks(path)
    chunks = chunk_document(ParsedDoc(meta("MEMO", "memo"), blocks))
    assert [c.locator for c in chunks] == ["p. 1", "p. 2"]


def test_intro_text_before_first_heading_uses_title():
    blocks = [Block("heading", text="Policy Title", level=0), Block("paragraph", text="Intro text."),
              Block("heading", text="1. Rule", level=1), Block("paragraph", text="Rule text.")]
    chunks = chunk_document(ParsedDoc(meta(), blocks))
    assert [c.locator for c in chunks] == ["Policy Title", "1. Rule"]


# --- retrieval isolation and determinism -------------------------------------------------------------

def _index() -> CorpusIndex:
    docs = {f"D{i}": meta(f"D{i}") for i in range(6)}
    chunks = [Chunk(f"D{i}#1", f"D{i}", "1. Leave", "1. Leave", f"Employees receive {i} days of annual leave and sick leave.")
              for i in range(6)]
    return CorpusIndex(docs, chunks, None)


def test_search_only_returns_allowed_documents():
    index = _index()
    ids = [f"D{i}" for i in range(6)]
    for size in range(0, 4):
        for allowed in itertools.combinations(ids, size):
            hits = index.search("annual leave days", set(allowed), k=10)
            assert {h.chunk.doc_id for h in hits} <= set(allowed)


def test_search_is_deterministic():
    index = _index()
    first = [h.chunk.chunk_id for h in index.search("annual leave", {f"D{i}" for i in range(6)})]
    assert all(first == [h.chunk.chunk_id for h in index.search("annual leave", {f"D{i}" for i in range(6)})] for _ in range(5))


def test_tokenizer_normalises_plurals_and_stopwords():
    assert tokenize("How many holidays are there?") == ["holiday"]
    assert tokenize("Policies") == ["policy"]


# --- holiday extraction formats ----------------------------------------------------------------------

def _doc(blocks):
    return ParsedDoc(meta("CAL", "calendar", ("IN",)), blocks)


def test_holidays_from_lines_and_month_first_dates():
    blocks = [Block("heading", text="Holidays 2026", level=1),
              Block("paragraph", text="- 1 January 2026 - New Year"),
              Block("table", rows=[["Date", "Occasion"], ["March 3, 2026", "Spring Day"], ["March 3, 2026", "Spring Day"]])]
    found = {(h.day, h.name) for h in extract_holidays(_doc(blocks))}
    assert found == {(date(2026, 1, 1), "New Year"), (date(2026, 3, 3), "Spring Day")}  # duplicate row removed


def test_optional_tables_by_header_word():
    blocks = [Block("heading", text="2026", level=1),
              Block("table", rows=[["Date", "Optional holiday"], ["5 May 2026", "Choice Day"]]),
              Block("table", rows=[["Date", "Holiday"], ["6 May 2026", "Closed Day"]])]
    kinds = {h.name: h.kind for h in extract_holidays(_doc(blocks))}
    assert kinds == {"Choice Day": "optional", "Closed Day": "closure"}


def test_tables_without_dates_are_ignored():
    blocks = [Block("table", rows=[["Level", "Notice"], ["L1", "30 days"]])]
    assert extract_holidays(_doc(blocks)) == []


# --- manifest validation ------------------------------------------------------------------------------

def test_manifest_skips_malformed_and_duplicate_entries(tmp_path: Path):
    good = {"doc_id": "A", "file": "a.md", "title": "A", "doc_type": "policy", "regions": ["global"],
            "effective_date": "2026-01-01", "access_level": "ALL"}
    entries = [good, {**good, "title": "duplicate"}, {"doc_id": "B", "file": "b.md"}, {**good, "doc_id": "C", "effective_date": "soon"}]
    path = tmp_path / "m.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    docs = load_manifest(path)
    assert [d.doc_id for d in docs] == ["A"]
    assert docs[0].title == "A" and docs[0].regions == ("GLOBAL",) and docs[0].access_level == "all"


# --- more precedence cases ------------------------------------------------------------------------------

def _claims(*specs):
    out = []
    for i, (d, value) in enumerate(specs):
        out.append(Claim(f"F{i}", "R1", "rule", value, value, Chunk(f"{d.doc_id}#1", d.doc_id, "s", "s", "t"), d))
    return out


def test_three_way_tie_conflict_keeps_all_three():
    a, b, c = meta("A"), meta("B", "handbook"), meta("C", "memo")
    (res,) = resolve(_claims((a, "10 days"), (b, "12 days"), (c, "14 days")), date(2026, 9, 1))
    assert res.conflict and len(res.winners) == 3


def test_two_agree_against_third_at_same_level_is_still_a_conflict():
    a, b, c = meta("A"), meta("B"), meta("C")
    (res,) = resolve(_claims((a, "10 days"), (b, "10 days"), (c, "12 days")), date(2026, 9, 1))
    assert res.conflict  # majority does not decide; the spec says give both values


def test_upcoming_change_mentioned_alongside_conflict():
    a, b = meta("A"), meta("B", "handbook")
    future = meta("F", eff="2027-01-01")
    (res,) = resolve(_claims((a, "10 days"), (b, "12 days"), (future, "15 days")), date(2026, 9, 1))
    assert res.conflict and [c.doc.doc_id for c in res.upcoming] == ["F"]


def test_faq_only_answer_is_used_when_nothing_outranks_it():
    faq = meta("FAQ", "faq")
    (res,) = resolve(_claims((faq, "2 days")), date(2026, 9, 1))
    assert [c.doc.doc_id for c in res.winners] == ["FAQ"]


def test_region_specific_faq_still_loses_to_global_policy_on_authority():
    faq_de = meta("FAQDE", "faq", ("DE",))
    policy = meta("P")
    (res,) = resolve(_claims((faq_de, "9 days"), (policy, "5 days")), date(2026, 9, 1))
    assert [c.doc.doc_id for c in res.winners] == ["P"] and res.decided_by == "authority"


def test_date_value_formats_compare_equal():
    assert values_equal("31 March", "March 31")
    assert not values_equal("31 March", "30 June")
