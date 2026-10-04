"""FAISS vector index over knowledge base chunks, persisted without pickle.

Files in ``rag.index_dir``:

- ``index.faiss``: a flat inner-product index over L2-normalized embeddings (cosine similarity)
- ``chunks.json``: the chunk text and metadata, in index order
- ``meta.json``: embedding model, dimensions, chunk settings, and a hash of the knowledge
  base content, so a stale index is detected and rebuilt automatically

LangChain's FAISS wrapper stores its document store as a pickle, which can execute code
when loaded from an untrusted file. Plain JSON avoids that entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from langchain_core.embeddings import Embeddings

from src.core.config import RAGConfig
from src.rag.chunking import Chunk, build_chunks
from src.rag.knowledge_base import GLOSSARY_FILE, KB_ROOT, article_paths

logger = logging.getLogger(__name__)

INDEX_FILE, CHUNKS_FILE, META_FILE = "index.faiss", "chunks.json", "meta.json"
FORMAT_VERSION = 1


class VectorIndexError(RuntimeError):
    """The index on disk is missing, corrupt, or doesn't match the current settings."""


def content_hash(root: Path = KB_ROOT) -> str:
    """Hash of every knowledge base file, so edits to any article trigger a rebuild."""
    digest = hashlib.sha256()
    files = [*article_paths(root), root / GLOSSARY_FILE]
    for path in files:
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms == 0, 1, norms)).astype("float32")


class VectorIndex:
    """Chunks with their L2-normalized embeddings in an exact (flat) FAISS inner-product index."""

    def __init__(self, chunks: Sequence[Chunk], vectors: np.ndarray, meta: dict) -> None:
        if len(chunks) != vectors.shape[0]:
            raise VectorIndexError("chunk count doesn't match vector count")
        self.chunks = list(chunks)
        self.vectors = _normalize(vectors)
        self.meta = meta
        self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
        self._faiss.add(self.vectors)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def dimensions(self) -> int:
        """Length of the embedding vectors."""
        return int(self.vectors.shape[1])

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        embeddings: Embeddings,
        config: RAGConfig,
        *,
        kb_hash: str = "",
    ) -> VectorIndex:
        """Embed the chunks into a new index; its metadata records the settings and ``kb_hash``.

        Raises ``VectorIndexError`` when ``chunks`` is empty.
        """
        if not chunks:
            raise VectorIndexError("no chunks to index")
        vectors = np.asarray(embeddings.embed_documents([c.embed_text for c in chunks]))
        meta = {
            "format_version": FORMAT_VERSION,
            "embedding_model": config.embedding_model,
            "dimensions": int(vectors.shape[1]),
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "kb_hash": kb_hash,
            "chunks": len(chunks),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        return cls(chunks, vectors, meta)

    def scores(self, query_vector: Sequence[float]) -> np.ndarray:
        """Cosine similarity of the query against every chunk, in index order."""
        query = _normalize(np.asarray([query_vector], dtype="float32"))
        if query.shape[1] != self.dimensions:
            raise VectorIndexError(
                f"query has {query.shape[1]} dimensions; index has {self.dimensions}"
            )
        scores, ids = self._faiss.search(query, len(self))
        ordered = np.empty(len(self), dtype="float32")
        ordered[ids[0]] = scores[0]
        return ordered

    def save(self, directory: Path) -> None:
        """Write the index, chunks, and metadata files to ``directory``, creating it if needed."""
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss, str(directory / INDEX_FILE))
        (directory / CHUNKS_FILE).write_text(
            json.dumps([c.model_dump(mode="json") for c in self.chunks]), encoding="utf-8"
        )
        (directory / META_FILE).write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> VectorIndex:
        """Load an index written by ``save``.

        Raises ``VectorIndexError`` if a file is missing or unreadable, or the format version
        changed.
        """
        try:
            meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
            chunks = [
                Chunk.model_validate(c)
                for c in json.loads((directory / CHUNKS_FILE).read_text(encoding="utf-8"))
            ]
            stored = faiss.read_index(str(directory / INDEX_FILE))
            vectors = stored.reconstruct_n(0, stored.ntotal)
        except (OSError, ValueError, RuntimeError) as exc:
            raise VectorIndexError(f"can't load index from {directory}: {exc}") from exc
        if meta.get("format_version") != FORMAT_VERSION:
            raise VectorIndexError("index format version changed")
        return cls(chunks, vectors, meta)


def is_current(index: VectorIndex, config: RAGConfig, kb_hash: str) -> bool:
    """Whether the index matches this embedding model, these chunk settings, and ``kb_hash``."""
    meta = index.meta
    return (
        meta.get("embedding_model") == config.embedding_model
        and meta.get("chunk_size") == config.chunk_size
        and meta.get("chunk_overlap") == config.chunk_overlap
        and meta.get("kb_hash") == kb_hash
    )


def ensure_index(
    config: RAGConfig,
    embeddings: Embeddings,
    *,
    index_dir: Path,
    kb_root: Path = KB_ROOT,
    force: bool = False,
) -> VectorIndex:
    """Load the saved index if it matches the knowledge base and settings; otherwise rebuild."""
    kb_hash = content_hash(kb_root)
    if not force and (index_dir / META_FILE).is_file():
        try:
            index = VectorIndex.load(index_dir)
            if is_current(index, config, kb_hash):
                return index
            logger.info("Knowledge base or RAG settings changed; rebuilding the index")
        except VectorIndexError as exc:
            logger.warning("Rebuilding unusable index: %s", exc)
    chunks = build_chunks(config, root=kb_root)
    index = VectorIndex.build(chunks, embeddings, config, kb_hash=kb_hash)
    index.save(index_dir)
    logger.info("Built index with %d chunks in %s", len(index), index_dir)
    return index
