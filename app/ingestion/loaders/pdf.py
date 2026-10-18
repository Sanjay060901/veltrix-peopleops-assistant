"""PDF loader: text layer with font-based heading detection, table extraction, OCR fallback."""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.ingestion.models import Block, is_metadata_table
from app.ingestion.ocr import OCR_DPI, ocr_png_bytes, ocr_text_to_paragraphs

log = logging.getLogger(__name__)

BULLETS = ("•", "•", "●", "▪", "-", "–")
MIN_TEXT_CHARS = 40  # a page with less text than this is treated as scanned


@dataclass
class _Line:
    page: int
    y0: float
    y1: float
    x0: float
    text: str
    size: float
    bold: bool


def _page_lines(page: pymupdf.Page, page_no: int, skip_boxes: list[pymupdf.Rect]) -> list[_Line]:
    raw: list[_Line] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            bbox = pymupdf.Rect(line["bbox"])
            if any(box.intersects(bbox) and _overlap(box, bbox) > 0.5 for box in skip_boxes):
                continue
            text = "".join(s["text"] for s in line["spans"]).strip()
            main = max(spans, key=lambda s: len(s["text"].strip()))
            bold = bool(main["flags"] & 16) or "bold" in main["font"].lower()
            raw.append(_Line(page_no, bbox.y0, bbox.y1, bbox.x0, text, round(main["size"], 1), bold))
    # Merge fragments printed on the same baseline (e.g. a bullet glyph and its text).
    raw.sort(key=lambda ln: (round(ln.y0), ln.x0))
    merged: list[_Line] = []
    for ln in raw:
        if merged and abs(merged[-1].y0 - ln.y0) < 2.0 and merged[-1].page == ln.page:
            prev = merged[-1]
            prev.text = f"{prev.text} {ln.text}"
            if len(ln.text) > len(prev.text) / 2:
                prev.size = max(prev.size, ln.size)
                prev.bold = prev.bold or ln.bold
            prev.y1 = max(prev.y1, ln.y1)
        else:
            merged.append(ln)
    return merged


def _overlap(box: pymupdf.Rect, other: pymupdf.Rect) -> float:
    inter = pymupdf.Rect(box) & other
    area = other.get_area()
    return inter.get_area() / area if area else 0.0


def _clean_cell(cell: str | None) -> str:
    return re.sub(r"\s+", " ", (cell or "").replace("\n", " ")).strip()


def load_pdf(path: Path) -> tuple[list[Block], bool]:
    doc = pymupdf.open(str(path))
    items: list[tuple[int, float, object]] = []  # (page, y, _Line | Block)
    all_lines: list[_Line] = []
    page_heights: dict[int, float] = {}
    ocr_used = False

    for index, page in enumerate(doc):
        page_no = index + 1
        page_heights[page_no] = page.rect.height
        if len(page.get_text().strip()) < MIN_TEXT_CHARS:
            pix = page.get_pixmap(dpi=OCR_DPI)
            text = ocr_png_bytes(pix.tobytes("png"))
            ocr_used = True
            log.info("ocr page=%s file=%s chars=%s", page_no, path.name, len(text))
            for i, para in enumerate(ocr_text_to_paragraphs(text)):
                items.append((page_no, float(i), Block("paragraph", text=para, page=page_no)))
            continue

        tables = page.find_tables().tables
        boxes = [pymupdf.Rect(t.bbox) for t in tables]
        for table, box in zip(tables, boxes):
            raw_rows = [[c or "" for c in row] for row in table.extract()]
            if is_metadata_table(raw_rows):
                continue  # document-control table; the manifest holds these values
            rows = [[_clean_cell(c) for c in row] for row in raw_rows]
            items.append((page_no, box.y0, Block("table", rows=rows, page=page_no)))
        lines = _page_lines(page, page_no, boxes)
        all_lines.extend(lines)
        items.extend((page_no, ln.y0, ln) for ln in lines)

    if not all_lines:
        return _finish(items, None, None, page_heights), ocr_used

    sizes = Counter()
    for ln in all_lines:
        sizes[ln.size] += len(ln.text)
    body_size = sizes.most_common(1)[0][0]
    # Repeated running headers/footers: identical text (digits normalised) on several pages,
    # or small text in the top/bottom margin.
    repeats = Counter(re.sub(r"\d+", "#", ln.text) for ln in all_lines)
    pages = len(page_heights)

    def is_furniture(ln: _Line) -> bool:
        height = page_heights[ln.page]
        in_margin = ln.y0 < height * 0.07 or ln.y1 > height * 0.93
        if in_margin and ln.size < body_size:
            return True
        if pages > 1 and repeats[re.sub(r"\d+", "#", ln.text)] >= pages and ln.size < body_size:
            return True
        return bool(re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", ln.text.strip(), re.I))

    return _finish(items, body_size, is_furniture, page_heights), ocr_used


def _finish(items, body_size, is_furniture, page_heights) -> list[Block]:
    items.sort(key=lambda it: (it[0], it[1]))
    blocks: list[Block] = []
    title_seen = False
    para: list[str] = []
    para_page: int | None = None
    last_line: _Line | None = None

    def flush() -> None:
        nonlocal para, para_page
        if para:
            blocks.append(Block("paragraph", text=" ".join(para), page=para_page))
        para, para_page = [], None

    for _page, _y, item in items:
        if isinstance(item, Block):
            flush()
            blocks.append(item)
            last_line = None
            continue
        ln: _Line = item
        if is_furniture and is_furniture(ln):
            continue
        is_title = body_size is not None and ln.size >= body_size * 1.5 and not title_seen
        is_heading = body_size is not None and ln.bold and ln.size >= body_size * 1.15 and len(ln.text) < 120
        if is_title:
            flush()
            title_seen = True
            blocks.append(Block("heading", text=ln.text, page=ln.page, level=0))
            last_line = None
            continue
        if is_heading:
            flush()
            blocks.append(Block("heading", text=ln.text, page=ln.page, level=1))
            last_line = None
            continue
        starts_bullet = ln.text.startswith(BULLETS[:4])
        # A page break inside a running paragraph is not a paragraph break, so gaps only
        # count between lines on the same page.
        same_page = last_line is not None and ln.page == last_line.page
        gap_break = same_page and ln.y0 - last_line.y1 > (last_line.y1 - last_line.y0) * 0.9
        if starts_bullet or gap_break:
            flush()
        text = ln.text
        if starts_bullet:
            text = "- " + text.lstrip("".join(BULLETS[:4])).strip()
        if para_page is None:
            para_page = ln.page
        para.append(text)
        last_line = ln
    flush()
    return blocks
