"""Markdown loader: ATX headings, pipe tables, lists and paragraphs."""
from __future__ import annotations

import re
from pathlib import Path

from app.ingestion.models import Block, is_metadata_table

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_TABLE_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_LIST = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")


def _strip_inline(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links
    text = re.sub(r"(\*\*|__|\*|_|`)(.+?)\1", r"\2", text)  # emphasis, code
    return text.strip()


def _cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [_strip_inline(c) for c in line.split("|")]


def load_markdown(path: Path) -> list[Block]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    blocks: list[Block] = []
    para: list[str] = []
    title_seen = False
    i = 0

    def flush() -> None:
        if para:
            blocks.append(Block("paragraph", text=" ".join(para)))
            para.clear()

    while i < len(lines):
        line = lines[i].rstrip()
        heading = _HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            text = _strip_inline(heading.group(2))
            if level == 1 and not title_seen:
                title_seen = True
                blocks.append(Block("heading", text=text, level=0))
            else:
                blocks.append(Block("heading", text=text, level=max(level - 1, 1)))
            i += 1
            continue
        if "|" in line and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1].strip()):
            flush()
            rows = [_cells(line)]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(_cells(lines[i]))
                i += 1
            if not is_metadata_table(rows[1:] or rows):
                blocks.append(Block("table", rows=rows))
            continue
        item = _LIST.match(line)
        if item:
            flush()
            blocks.append(Block("paragraph", text=f"- {_strip_inline(item.group(1))}"))
            i += 1
            continue
        if not line.strip():
            flush()
        else:
            para.append(_strip_inline(line))
        i += 1
    flush()
    return blocks
