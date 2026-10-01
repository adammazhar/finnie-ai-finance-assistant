"""Embedding model factory (sentence-transformers all-MiniLM-L6-v2, run locally on CPU).

With ``HF_HUB_OFFLINE=1`` (see ``.env.example``) the model loads from the local Hugging
Face cache without contacting the Hub, so startup is quiet and works without internet.
The cache is filled once by running ``python scripts/build_index.py`` with
``HF_HUB_OFFLINE=0`` (see README "Setup").
"""

from __future__ import annotations

import os
import sys
from functools import cache

from langchain_core.embeddings import Embeddings

from src.core.config import RAGConfig, get_settings

DOWNLOAD_HINT = (
    "The embedding model '{model}' isn't in the local Hugging Face cache, and HF_HUB_OFFLINE=1 "
    "prevents downloading it. Run this once to download it (about 90 MB):\n"
    "    HF_HUB_OFFLINE=0 python scripts/build_index.py"
)


class EmbeddingModelUnavailableError(RuntimeError):
    """The model can't be loaded (typically: offline mode with an empty cache)."""


def _enable_offline_mode() -> None:
    """Must run before Hugging Face libraries read their settings at import time."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    hub = sys.modules.get("huggingface_hub")
    if hub is not None:  # already imported: update the live setting too
        hub.constants.HF_HUB_OFFLINE = True


def build_embeddings(config: RAGConfig, *, offline: bool = False) -> Embeddings:
    """Vectors are L2-normalized, so inner product equals cosine similarity."""
    if offline:
        _enable_offline_mode()
    from langchain_huggingface import HuggingFaceEmbeddings

    try:
        return HuggingFaceEmbeddings(
            model=config.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
    except OSError as exc:
        if offline:
            raise EmbeddingModelUnavailableError(
                DOWNLOAD_HINT.format(model=config.embedding_model)
            ) from exc
        raise


@cache
def get_embeddings() -> Embeddings:
    """Process-wide model (loading takes seconds). ``get_embeddings.cache_clear()`` resets it."""
    settings = get_settings()
    return build_embeddings(settings.rag, offline=settings.hf_hub_offline)
