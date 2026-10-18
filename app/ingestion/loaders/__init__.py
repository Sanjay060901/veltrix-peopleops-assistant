"""Format loaders. Each returns an ordered list of Blocks (headings, paragraphs, tables)."""
from __future__ import annotations

from pathlib import Path

from app.ingestion.models import Block


def load_blocks(path: Path) -> tuple[list[Block], bool]:
    """Dispatch on file extension. Returns (blocks, ocr_used)."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        from app.ingestion.loaders.pdf import load_pdf

        return load_pdf(path)
    if ext == ".docx":
        from app.ingestion.loaders.docx_loader import load_docx

        return load_docx(path), False
    if ext in (".html", ".htm"):
        from app.ingestion.loaders.html import load_html

        return load_html(path), False
    if ext in (".md", ".markdown", ".txt"):
        from app.ingestion.loaders.markdown import load_markdown

        return load_markdown(path), False
    raise ValueError(f"unsupported file type: {path.name}")
