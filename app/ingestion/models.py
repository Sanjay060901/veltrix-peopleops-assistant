"""Data model shared by ingestion and retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

BlockKind = Literal["heading", "paragraph", "table"]

# Field labels of the document-control table at the top of each document. They describe the
# document format, not corpus facts; the manifest is the source of truth for these values.
METADATA_LABELS = {"document id", "version", "effective date", "applies to", "owner", "classification"}


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    file: str
    title: str
    doc_type: str
    regions: tuple[str, ...]
    effective_date: date
    access_level: str

    @property
    def is_global(self) -> bool:
        return "GLOBAL" in self.regions


@dataclass
class Block:
    kind: BlockKind
    text: str = ""
    rows: list[list[str]] = field(default_factory=list)  # tables only
    page: int | None = None
    level: int = 1  # headings only


@dataclass
class ParsedDoc:
    meta: DocMeta
    blocks: list[Block]
    ocr_used: bool = False

    @property
    def has_headings(self) -> bool:
        return any(b.kind == "heading" for b in self.blocks)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    locator: str  # nearest section heading, or "p. N" for documents without headings
    heading: str
    text: str  # body text of the chunk (corpus text)

    def render(self, title: str) -> str:
        """Text block sent to the LLM. All of it is counted as corpus text."""
        head = f"{title} - {self.heading}" if self.heading else title
        return f"[{head}]\n{self.text}"


def is_metadata_table(rows: list[list[str]]) -> bool:
    """True for the key/value document-control table (Document ID, Version, ...)."""
    if not rows:
        return False
    labels: list[str] = []
    for row in rows:
        if row:
            labels.extend(part.strip().lower() for part in row[0].split("\n"))
    hits = sum(1 for lab in labels if lab in METADATA_LABELS)
    return hits >= 3
