"""DOCX loader: walks the body in document order so tables stay next to their headings."""
from __future__ import annotations

import re
from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.table import Table

from app.ingestion.models import Block, is_metadata_table

_HEADING_STYLE = re.compile(r"^heading\s*(\d)$", re.I)


def _style_id(p_el) -> str:
    ppr = p_el.find(qn("w:pPr"))
    if ppr is None:
        return ""
    style = ppr.find(qn("w:pStyle"))
    return (style.get(qn("w:val")) or "") if style is not None else ""


def _is_list_item(p_el) -> bool:
    ppr = p_el.find(qn("w:pPr"))
    return ppr is not None and ppr.find(qn("w:numPr")) is not None


def _para_text(p_el) -> str:
    parts: list[str] = []
    for node in p_el.iter():
        tag = node.tag
        if tag == qn("w:t"):
            parts.append(node.text or "")
        elif tag in (qn("w:tab"),):
            parts.append(" ")
        elif tag in (qn("w:br"), qn("w:cr")):
            parts.append("\n")
    return "".join(parts).strip()


def _all_bold(p_el) -> bool:
    runs = [r for r in p_el.iter(qn("w:r")) if "".join(t.text or "" for t in r.iter(qn("w:t"))).strip()]
    if not runs:
        return False
    for run in runs:
        rpr = run.find(qn("w:rPr"))
        bold = rpr.find(qn("w:b")) if rpr is not None else None
        if bold is None or bold.get(qn("w:val")) in ("0", "false"):
            return False
    return True


def _table_rows(tbl: Table) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in tbl.rows:
        cells: list[str] = []
        previous = None
        for cell in row.cells:
            # Horizontally merged cells are returned once per grid column; keep one copy.
            if previous is not None and cell._tc is previous:
                continue
            previous = cell._tc
            cells.append(cell.text)
        rows.append(cells)
    return rows


def load_docx(path: Path) -> list[Block]:
    document = docx.Document(str(path))
    blocks: list[Block] = []
    for el in document.element.body.iterchildren():
        if el.tag == qn("w:p"):
            text = _para_text(el)
            if not text:
                continue
            style = _style_id(el).replace(" ", "").lower()
            match = _HEADING_STYLE.match(style)
            if style == "title":
                blocks.append(Block("heading", text=text, level=0))
            elif match or style.startswith("heading"):
                level = int(match.group(1)) if match else 1
                blocks.append(Block("heading", text=text, level=max(level, 1)))
            elif _all_bold(el) and len(text) < 100 and not text.endswith((".", ":")):
                blocks.append(Block("heading", text=text, level=1))
            elif _is_list_item(el) or style.startswith("list"):
                blocks.append(Block("paragraph", text=f"- {text}"))
            else:
                blocks.append(Block("paragraph", text=text))
        elif el.tag == qn("w:tbl"):
            raw_rows = _table_rows(Table(el, document))
            if is_metadata_table(raw_rows):
                continue
            rows = [[re.sub(r"\s+", " ", c).strip() for c in row] for row in raw_rows]
            if any(any(c for c in row) for row in rows):
                blocks.append(Block("table", rows=rows))
    return blocks
