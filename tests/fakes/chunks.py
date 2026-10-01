"""Knowledge base chunks built directly, for tests that need a tiny search index.

``build_chunks`` runs LangChain's text splitters, whose package import loads
sentence-transformers, transformers, and torch (about 20 s). Agent and UI tests only need
a few searchable passages, so they use these helpers instead; for short sections the
result is the same as ``build_chunks`` (one chunk per section). The chunking itself is
tested in tests/unit/rag.
"""

from __future__ import annotations

from src.core.config import RAGConfig
from src.rag.chunking import Chunk
from src.rag.index import VectorIndex
from src.rag.knowledge_base import SourceRef
from src.rag.retriever import Retriever
from tests.fakes.embeddings import FakeEmbeddings

GLOSSARY_SOURCE = SourceRef(name="Investor.gov glossary", url="https://www.investor.gov/glossary")


def article(article_id: str, category: str, title: str, sections: dict[str, str]) -> list[Chunk]:
    source = SourceRef(name=f"{title} source", url=f"https://www.investor.gov/{article_id}")
    return [
        Chunk(
            id=f"{article_id}#{i}",
            kind="article",
            article_id=article_id,
            title=title,
            category=category,
            difficulty="beginner",
            section=section,
            text=text,
            sources=[source],
        )
        for i, (section, text) in enumerate(sections.items())
    ]


def glossary(term: str, definition: str) -> Chunk:
    slug = "-".join(term.lower().split())
    return Chunk(
        id=f"glossary:{slug}",
        kind="glossary",
        article_id=f"glossary:{slug}",
        title=term,
        category="glossary",
        section="Definition",
        text=definition,
        sources=[GLOSSARY_SOURCE],
    )


def retriever(chunks: list[Chunk], config: RAGConfig) -> Retriever:
    embeddings = FakeEmbeddings()
    return Retriever(VectorIndex.build(chunks, embeddings, config), embeddings, config)
