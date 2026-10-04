"""Split knowledge base articles and glossary terms into retrievable chunks.

Articles are split on ``##``/``###`` headings first, so a chunk never straddles two
sections, then long sections are split to ``rag.chunk_size`` characters with overlap.
MiniLM truncates input at 256 word-pieces (roughly 1,000 characters), so chunks stay
below that. Each chunk is embedded with its article title and section prepended, which
helps short sections match questions about their topic.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from src.core.config import RAGConfig

if TYPE_CHECKING:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from src.rag.knowledge_base import (
    CATEGORIES,
    KB_ROOT,
    Article,
    Glossary,
    SourceRef,
    load_articles,
    load_glossary,
)

GLOSSARY_CATEGORY = "glossary"
INTRO_SECTION = "Overview"


class Chunk(BaseModel):
    """One retrievable piece of an article section, or one glossary term, plus its metadata."""

    model_config = ConfigDict(frozen=True)

    id: str
    kind: Literal["article", "glossary"]
    article_id: str
    title: str
    category: str
    difficulty: str | None = None
    section: str
    text: str
    sources: list[SourceRef]

    @property
    def embed_text(self) -> str:
        """What gets embedded: title and section give the text its context."""
        return f"{self.title} > {self.section}: {self.text}"

    @property
    def category_label(self) -> str:
        """Display name of the chunk's category; ``"Glossary"`` for glossary terms."""
        return CATEGORIES.get(self.category, "Glossary")


def _splitters(
    config: RAGConfig,
) -> tuple[MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter]:
    # Imported here: the package's __init__ loads sentence-transformers, transformers, and
    # torch (about 20 s), which only chunking needs. Importing it lazily keeps the app,
    # the agents, and test collection fast.
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    headers = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "section"), ("###", "subsection")], strip_headers=True
    )
    sizes = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return headers, sizes


def chunk_article(article: Article, config: RAGConfig) -> list[Chunk]:
    """Split an article by its markdown headings, then by size; each chunk keeps its section."""
    headers, sizes = _splitters(config)
    meta = article.meta
    chunks: list[Chunk] = []
    for part in headers.split_text(article.body):
        section = (
            " / ".join(v for k, v in part.metadata.items() if k in ("section", "subsection") and v)
            or INTRO_SECTION
        )
        for piece in sizes.split_text(part.page_content):
            text = piece.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    id=f"{meta.id}#{len(chunks)}",
                    kind="article",
                    article_id=meta.id,
                    title=meta.title,
                    category=meta.category,
                    difficulty=meta.difficulty,
                    section=section,
                    text=text,
                    sources=meta.sources,
                )
            )
    return chunks


def _slug(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")


def chunk_glossary(glossary: Glossary) -> list[Chunk]:
    """One chunk per term, so a definition question retrieves exactly that definition."""
    chunks = []
    for term in glossary.terms:
        text = term.definition
        if term.related:
            text += f" Related terms: {', '.join(term.related)}."
        chunks.append(
            Chunk(
                id=f"glossary:{_slug(term.term)}",
                kind="glossary",
                article_id=f"glossary:{_slug(term.term)}",
                title=term.term,
                category=GLOSSARY_CATEGORY,
                section="Definition",
                text=text,
                sources=glossary.sources,
            )
        )
    return chunks


def build_chunks(
    config: RAGConfig,
    *,
    articles: Iterable[Article] | None = None,
    glossary: Glossary | None = None,
    root: Path = KB_ROOT,
) -> list[Chunk]:
    """All chunks for the knowledge base: every article, then every glossary term."""
    articles = list(articles) if articles is not None else load_articles(root)
    if glossary is None and (root / "glossary.yaml").is_file():
        glossary = load_glossary(root)
    chunks = [chunk for article in articles for chunk in chunk_article(article, config)]
    if glossary is not None:
        chunks.extend(chunk_glossary(glossary))
    return chunks
