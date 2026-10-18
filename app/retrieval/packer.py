"""Context packing under the corpus-token budget. The single place corpus tokens are counted."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.ingestion.models import Chunk, DocMeta
from app.llm.tokens import count_tokens


@dataclass
class PackedContext:
    chunks: list[Chunk] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)  # rendered text, one per chunk
    corpus_tokens: int = 0

    def add(self, chunk: Chunk, rendered: str, tokens: int) -> None:
        self.chunks.append(chunk)
        self.blocks.append(rendered)
        self.corpus_tokens += tokens


def pack(chunks: list[Chunk], docs: dict[str, DocMeta], budget: int, start_tokens: int = 0) -> PackedContext:
    """Greedy packing in retrieval order. A chunk that does not fit is skipped (a later, smaller
    one may still fit); the total never exceeds the budget."""
    ctx = PackedContext(corpus_tokens=start_tokens)
    for chunk in chunks:
        rendered = chunk.render(docs[chunk.doc_id].title)
        tokens = count_tokens(rendered)
        if ctx.corpus_tokens + tokens > budget:
            continue
        ctx.add(chunk, rendered, tokens)
    return ctx


def corpus_tokens_of(texts: list[str]) -> int:
    return sum(count_tokens(t) for t in texts)
