"""HTML loader: headings, paragraphs, list items and tables in document order."""
from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from app.ingestion.models import Block, is_metadata_table

_SKIP = {"nav", "footer", "header", "script", "style", "noscript", "aside", "form"}
_HEADINGS = {"h1": 0, "h2": 1, "h3": 2, "h4": 3, "h5": 4, "h6": 5}


def _text(node: Tag) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ")).strip()


def load_html(path: Path) -> list[Block]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    root = soup.body or soup
    blocks: list[Block] = []
    title_seen = False

    def walk(node: Tag) -> None:
        nonlocal title_seen
        for child in node.children:
            if not isinstance(child, Tag):
                continue
            name = child.name.lower()
            if name in _SKIP:
                continue
            if name in _HEADINGS:
                text = _text(child)
                if not text:
                    continue
                level = _HEADINGS[name]
                if level == 0 and not title_seen:
                    title_seen = True
                    blocks.append(Block("heading", text=text, level=0))
                else:
                    blocks.append(Block("heading", text=text, level=max(level, 1)))
            elif name == "table":
                rows = [
                    [_text(cell) for cell in tr.find_all(["td", "th"], recursive=False)]
                    for tr in child.find_all("tr")
                ]
                rows = [r for r in rows if any(r)]
                if rows and not is_metadata_table(rows):
                    blocks.append(Block("table", rows=rows))
            elif name in ("ul", "ol"):
                for li in child.find_all("li", recursive=False):
                    text = _text(li)
                    if text:
                        blocks.append(Block("paragraph", text=f"- {text}"))
            elif name in ("p", "blockquote", "pre", "dd", "dt"):
                text = _text(child)
                if text:
                    blocks.append(Block("paragraph", text=text))
            else:
                # Containers (div, section, main, article, ...): descend. Bare text in a
                # container without block children is kept as a paragraph.
                if any(isinstance(c, Tag) for c in child.children):
                    walk(child)
                else:
                    text = _text(child)
                    if text:
                        blocks.append(Block("paragraph", text=text))

    walk(root)
    return blocks
