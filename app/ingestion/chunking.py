"""Section-based chunking.

Each chunk is one section (heading + its paragraphs and tables). HR documents are short and
organised by numbered sections, and one section usually states one rule, so a section is the
natural unit both for retrieval and for the citation locator. Long sections are split on block
boundaries; long tables are split by rows with the header row repeated.
"""
from __future__ import annotations

import logging
import re

from app.ingestion.models import Block, Chunk, ParsedDoc
from app.ingestion.sanitize import is_injection
from app.llm.tokens import count_tokens

log = logging.getLogger(__name__)

MAX_CHUNK_TOKENS = 350


def render_table(rows: list[list[str]]) -> list[str]:
    return [" | ".join(cell for cell in row) for row in rows if any(row)]


def _block_lines(block: Block) -> list[str]:
    if block.kind == "table":
        return render_table(block.rows)
    return [block.text]


def _split_table(rows: list[list[str]], budget: int) -> list[str]:
    """Split a long table into row groups, repeating the header row in each piece."""
    lines = render_table(rows)
    if not lines:
        return []
    header, body = lines[0], lines[1:]
    pieces: list[str] = []
    current: list[str] = [header]
    for line in body:
        if count_tokens("\n".join(current + [line])) > budget and len(current) > 1:
            pieces.append("\n".join(current))
            current = [header]
        current.append(line)
    pieces.append("\n".join(current))
    return pieces


def chunk_document(doc: ParsedDoc) -> list[Chunk]:
    meta = doc.meta
    use_pages = not any(b.kind == "heading" and b.level >= 1 for b in doc.blocks)
    sections: list[tuple[str, int | None, list[Block]]] = []
    heading = ""
    current: list[Block] = []
    current_page: int | None = None

    for block in doc.blocks:
        if block.kind == "heading":
            if current:
                sections.append((heading, current_page, current))
            heading = block.text
            current, current_page = [], block.page
            continue
        if block.kind == "paragraph" and is_injection(block.text):
            log.warning("dropped instruction-like paragraph doc=%s", meta.doc_id)
            continue
        if use_pages and current and block.page is not None and block.page != current_page:
            # No headings: one section per page so the locator "p. N" is exact.
            sections.append((heading, current_page, current))
            current = []
        if not current:
            current_page = block.page if block.page is not None else current_page
        current.append(block)
    if current:
        sections.append((heading, current_page, current))

    chunks: list[Chunk] = []
    for heading, page, blocks in sections:
        locator = f"p. {page or 1}" if use_pages else (heading or meta.title)
        label = "" if use_pages else heading
        pieces: list[str] = []
        buf: list[str] = []
        for block in blocks:
            lines = _block_lines(block)
            text = "\n".join(lines)
            if count_tokens(text) > MAX_CHUNK_TOKENS:
                if buf:
                    pieces.append("\n".join(buf))
                    buf = []
                if block.kind == "table":
                    pieces.extend(_split_table(block.rows, MAX_CHUNK_TOKENS))
                else:
                    pieces.extend(_split_text(text, MAX_CHUNK_TOKENS))
                continue
            if buf and count_tokens("\n".join(buf + [text])) > MAX_CHUNK_TOKENS:
                pieces.append("\n".join(buf))
                buf = []
            buf.append(text)
        if buf:
            pieces.append("\n".join(buf))
        for piece in pieces:
            if piece.strip():
                chunks.append(
                    Chunk(
                        chunk_id=f"{meta.doc_id}#{len(chunks) + 1}",
                        doc_id=meta.doc_id,
                        locator=locator,
                        heading=label,
                        text=piece.strip(),
                    )
                )
    return chunks


def _split_text(text: str, budget: int) -> list[str]:
    """Split an over-long paragraph on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and count_tokens(candidate) > budget:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces
