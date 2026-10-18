"""Hybrid retrieval: BM25 keyword scores fused with dense cosine scores (reciprocal rank fusion).

Search always takes the set of allowed doc_ids, computed by the eligibility filter; chunks of any
other document are never scored or returned. Ties are broken by chunk_id so results are stable.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from app.ingestion.models import Chunk, DocMeta
from app.llm.tokens import count_tokens
from app.retrieval.embeddings import Embedder

log = logging.getLogger(__name__)

_STOP = set(
    "a an the and or of to in on for by with at from is are was were be been do does did i me my "
    "we our you your it its this that these those what which who whom how when where can could "
    "should would will shall may might much many any some there their they them as if than then "
    "about into per please tell".split()
)
RRF_K = 60


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for w in words:
        if w in _STOP:
            continue
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class CorpusIndex:
    def __init__(
        self,
        docs: dict[str, DocMeta],
        chunks: list[Chunk],
        embedder: Embedder | None = None,
        vectors: np.ndarray | None = None,
    ) -> None:
        """`vectors` (one row per chunk, in order) can come from the ingestion cache; otherwise
        the chunks are embedded here when an embedder is available."""
        self.docs = docs
        self.chunks = chunks
        self.embedder = embedder
        self._tokens = [tokenize(self.search_text(c)) for c in chunks]
        self._tf = [Counter(t) for t in self._tokens]
        self._len = [len(t) for t in self._tokens]
        self._chunk_tokens = [count_tokens(c.text) for c in chunks]
        self._avg = (sum(self._len) / len(self._len)) if chunks else 0.0
        df: Counter = Counter()
        for toks in self._tokens:
            df.update(set(toks))
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._vectors: np.ndarray | None = None
        if vectors is not None and embedder is not None and len(vectors) == len(chunks):
            self._vectors = vectors
        elif embedder is not None and chunks:
            try:
                self._vectors = embedder.embed_documents([self.search_text(c) for c in chunks])
            except Exception as exc:
                log.warning("embedding chunks failed, keyword retrieval only: %s", exc)
                self._vectors = None

    @staticmethod
    def text_for(meta: DocMeta, chunk: Chunk) -> str:
        return f"{meta.title}\n{chunk.heading}\n{chunk.text}"

    def search_text(self, chunk: Chunk) -> str:
        return self.text_for(self.docs[chunk.doc_id], chunk)

    def _bm25(self, query_tokens: list[str], idx: int, k1: float = 1.4, b: float = 0.75) -> float:
        tf = self._tf[idx]
        score = 0.0
        for t in query_tokens:
            f = tf.get(t)
            if not f:
                continue
            denom = f + k1 * (1 - b + b * self._len[idx] / (self._avg or 1))
            score += self._idf.get(t, 0.0) * f * (k1 + 1) / denom
        return score

    def search(self, query: str, allowed_doc_ids: set[str], k: int = 12, per_doc_tokens: int = 450) -> list[Hit]:
        """Top-k chunks from the allowed documents. Each document may contribute at most
        `per_doc_tokens` tokens, so competing documents all get a share of the context while a
        document made of many small sections (a schedule, a calendar) is not cut short."""
        candidates = [i for i, c in enumerate(self.chunks) if c.doc_id in allowed_doc_ids]
        if not candidates or not query.strip():
            return []
        q_tokens = tokenize(query)
        bm25 = {i: self._bm25(q_tokens, i) for i in candidates}
        rankings: list[list[int]] = []
        keyword_ranked = sorted((i for i in candidates if bm25[i] > 0), key=lambda i: (-bm25[i], self.chunks[i].chunk_id))
        rankings.append(keyword_ranked)
        if self._vectors is not None and self.embedder is not None:
            try:
                qv = self.embedder.embed_query(query)
                sims = self._vectors[candidates] @ qv
                dense = {i: float(s) for i, s in zip(candidates, sims)}
                rankings.append(sorted(candidates, key=lambda i: (-round(dense[i], 6), self.chunks[i].chunk_id)))
            except Exception as exc:
                log.warning("query embedding failed: %s", exc)
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        ordered = sorted(fused, key=lambda i: (-round(fused[i], 9), self.chunks[i].chunk_id))
        hits: list[Hit] = []
        per_doc_used: Counter = Counter()
        for i in ordered:
            chunk = self.chunks[i]
            size = self._chunk_tokens[i]
            # The best chunk of a document is always allowed, whatever its size.
            if per_doc_used[chunk.doc_id] and per_doc_used[chunk.doc_id] + size > per_doc_tokens:
                continue
            per_doc_used[chunk.doc_id] += size
            hits.append(Hit(chunk, fused[i]))
            if len(hits) >= k:
                break
        return hits
