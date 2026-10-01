"""Deterministic bag-of-words embeddings: texts that share words get similar vectors."""

from __future__ import annotations

import hashlib
import re

import numpy as np
from langchain_core.embeddings import Embeddings

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "is",
        "are",
        "and",
        "or",
        "for",
        "on",
        "what",
        "how",
        "do",
        "does",
        "it",
        "its",
        "by",
        "when",
        "with",
    ]
)


class FakeEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions
        self.documents_embedded = 0
        self.queries_embedded = 0

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word in STOPWORDS:
                continue
            word = word.removesuffix("s") if len(word) > 3 else word
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimensions
            vector[bucket] += 1.0
        norm = np.linalg.norm(vector)
        return (vector / norm if norm else vector).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents_embedded += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries_embedded += 1
        return self._vector(text)
