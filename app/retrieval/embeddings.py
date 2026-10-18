"""Local CPU embeddings (fastembed / ONNX). Optional: retrieval falls back to BM25 alone."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str, cache_dir: str = "") -> None:
        from fastembed import TextEmbedding

        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        self._model = TextEmbedding(model_name=model_name, **kwargs)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self._model.passage_embed(texts)), dtype=np.float32)
        return _normalise(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.array(list(self._model.query_embed([text]))[0], dtype=np.float32)
        return _normalise(vector[None, :])[0]


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def load_embedder(provider: str, model_name: str, cache_dir: str = "") -> Embedder | None:
    if provider != "local":
        return None
    try:
        return Embedder(model_name, cache_dir)
    except Exception as exc:  # model download or runtime failure: degrade to keyword search
        log.warning("embedding model unavailable, using keyword retrieval only: %s", exc)
        return None
