"""Retrieval: category filtering, a relevance threshold, and diverse (MMR) selection.

For a query, every chunk is scored by cosine similarity (exact search; the corpus is
about a thousand chunks). Then:

1. Keep chunks in the requested categories (glossary terms are included by default).
2. Drop chunks below ``rag.score_threshold``.
3. If fewer than ``k`` remain, widen to all categories and say so in the result.
4. From the best ``rag.fetch_k`` candidates, pick ``k`` with maximal marginal relevance
   (relevance balanced against similarity to chunks already picked), allowing at most
   ``rag.max_chunks_per_article`` chunks from any one article.

An empty result means the knowledge base doesn't confidently cover the question.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from functools import cache
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from src.core.config import RAGConfig, get_settings
from src.rag.chunking import GLOSSARY_CATEGORY, Chunk
from src.rag.embeddings import get_embeddings
from src.rag.index import VectorIndex, ensure_index
from src.rag.knowledge_base import CATEGORIES

logger = logging.getLogger(__name__)


class RetrievedChunk(BaseModel):
    """A chunk with its cosine similarity to the query."""

    chunk: Chunk
    score: float


class RetrievalResult(BaseModel):
    """The chunks chosen for a query.

    ``widened`` is true when the category filter left too few chunks, so all categories were
    searched.
    """

    query: str
    categories: list[str] | None
    chunks: list[RetrievedChunk]
    widened: bool = False
    latency_ms: float = 0.0

    @property
    def confident(self) -> bool:
        """False when nothing in the knowledge base cleared the relevance threshold."""
        return bool(self.chunks)


class Retriever:
    """Picks relevant, diverse chunks from a ``VectorIndex`` (steps in the module docstring)."""

    def __init__(self, index: VectorIndex, embeddings: Embeddings, config: RAGConfig) -> None:
        self.index = index
        self.embeddings = embeddings
        self.config = config
        self._categories = np.array([c.category for c in index.chunks])
        self._articles = [c.article_id for c in index.chunks]

    def retrieve(
        self,
        query: str,
        *,
        categories: Iterable[str] | None = None,
        k: int | None = None,
        include_glossary: bool = True,
    ) -> RetrievalResult:
        """Up to ``k`` (default ``rag.top_k``) relevant, diverse chunks for ``query``.

        Raises ``ValueError`` for an empty query or an unknown category.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query is empty")
        wanted = sorted(set(categories)) if categories else None
        unknown = [c for c in wanted or [] if c not in CATEGORIES and c != GLOSSARY_CATEGORY]
        if unknown:
            raise ValueError(f"Unknown categories: {', '.join(unknown)}")
        k = k or self.config.top_k
        started = time.perf_counter()

        scores = self.index.scores(self.embeddings.embed_query(query))
        relevant = scores >= self.config.score_threshold
        widened = False
        if wanted is None:
            candidates = relevant
        else:
            allowed = set(wanted) | ({GLOSSARY_CATEGORY} if include_glossary else set())
            candidates = relevant & np.isin(self._categories, list(allowed))
            if candidates.sum() < k and relevant.sum() > candidates.sum():
                widened = True
                candidates = relevant
                logger.info("Widened retrieval beyond %s", wanted)

        picked = self._mmr(scores, np.flatnonzero(candidates), k)
        return RetrievalResult(
            query=query,
            categories=wanted,
            chunks=[
                RetrievedChunk(chunk=self.index.chunks[i], score=round(float(scores[i]), 4))
                for i in picked
            ],
            widened=widened,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def _mmr(self, scores: np.ndarray, candidates: np.ndarray, k: int) -> list[int]:
        """Maximal marginal relevance over the top ``fetch_k`` candidates."""
        if candidates.size == 0:
            return []
        top = candidates[np.argsort(-scores[candidates])][: self.config.fetch_k]
        vectors = self.index.vectors
        lam = self.config.mmr_lambda
        per_article: dict[str, int] = {}
        picked: list[int] = []
        remaining = list(top)
        while remaining and len(picked) < k:
            best, best_value = None, -np.inf
            for i in remaining:
                if per_article.get(self._articles[i], 0) >= self.config.max_chunks_per_article:
                    continue
                redundancy = max((float(vectors[i] @ vectors[j]) for j in picked), default=0.0)
                value = lam * float(scores[i]) - (1 - lam) * redundancy
                if value > best_value:
                    best, best_value = i, value
            if best is None:
                break
            picked.append(int(best))
            remaining.remove(best)
            per_article[self._articles[best]] = per_article.get(self._articles[best], 0) + 1
        return picked


def build_retriever(
    config: RAGConfig | None = None,
    embeddings: Embeddings | None = None,
    *,
    index_dir: Path | None = None,
) -> Retriever:
    """Create a retriever from settings, using the saved index or rebuilding it.

    The index is rebuilt and saved to ``index_dir`` when it is missing, unreadable, or out of date.
    """
    settings = get_settings()
    config = config or settings.rag
    embeddings = embeddings or get_embeddings()
    index = ensure_index(
        config, embeddings, index_dir=index_dir or settings.resolve_path(config.index_dir)
    )
    return Retriever(index, embeddings, config)


@cache
def get_retriever() -> Retriever:
    """Process-wide retriever; builds or loads the index on first use."""
    return build_retriever()
