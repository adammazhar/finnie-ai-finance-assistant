"""Embedding model factory (sentence-transformers all-MiniLM-L6-v2, run locally on CPU)."""

from __future__ import annotations

from functools import cache

from langchain_core.embeddings import Embeddings

from src.core.config import RAGConfig, get_settings


def build_embeddings(config: RAGConfig) -> Embeddings:
    """Vectors are L2-normalized, so inner product equals cosine similarity."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model=config.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


@cache
def get_embeddings() -> Embeddings:
    """Process-wide model (loading takes seconds). ``get_embeddings.cache_clear()`` resets it."""
    return build_embeddings(get_settings().rag)
