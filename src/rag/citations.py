"""Turn retrieved chunks into numbered LLM context, and keep answers' citations honest.

The LLM sees context blocks numbered ``[1]..[k]`` and is asked to cite them inline.
``check_citations`` removes any ``[n]`` that doesn't match a block, so an invented
citation never reaches the user, and ``render_sources`` lists only the sources the answer
actually cited.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from src.core.models import Source
from src.rag.retriever import RetrievedChunk

CITATION = re.compile(r"\[(\d{1,2})\]")


class ContextBlock(BaseModel):
    number: int
    chunk: RetrievedChunk

    @property
    def source(self) -> Source:
        chunk = self.chunk.chunk
        first = chunk.sources[0] if chunk.sources else None
        return Source(
            title=chunk.title,
            kind="knowledge_base",
            category=chunk.category_label,
            url=first.url if first else None,
            article_id=chunk.article_id,
            score=self.chunk.score,
        )


def build_context(chunks: Sequence[RetrievedChunk]) -> tuple[str, list[ContextBlock]]:
    """Numbered context for the prompt, plus the blocks for citation checking."""
    blocks = [ContextBlock(number=i, chunk=c) for i, c in enumerate(chunks, start=1)]
    text = "\n\n".join(
        f"[{b.number}] {b.chunk.chunk.title} > {b.chunk.chunk.section}\n{b.chunk.chunk.text}"
        for b in blocks
    )
    return text, blocks


class CitationCheck(BaseModel):
    text: str
    cited: list[int]
    removed: list[int]


def check_citations(answer: str, block_count: int) -> CitationCheck:
    """Keep valid ``[n]`` citations; strip any number with no matching context block."""
    cited: list[int] = []
    removed: list[int] = []

    def keep_or_drop(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if 1 <= number <= block_count:
            if number not in cited:
                cited.append(number)
            return match.group(0)
        removed.append(number)
        return ""

    text = CITATION.sub(keep_or_drop, answer)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)  # tidy spaces left by removed markers
    text = re.sub(r"[ \t]{2,}", " ", text)
    return CitationCheck(text=text, cited=cited, removed=removed)


def cited_sources(blocks: Sequence[ContextBlock], cited: Sequence[int]) -> list[Source]:
    """Sources for the cited blocks, one per article, in citation order."""
    by_number = {b.number: b for b in blocks}
    seen: set[str] = set()
    sources = []
    for number in cited:
        source = by_number[number].source
        key = source.article_id or source.title
        if key not in seen:
            seen.add(key)
            sources.append(source)
    return sources


def render_sources(sources: Sequence[Source]) -> str:
    """Markdown 'Sources' list, e.g. ``1. What Is a Bond? (Bonds and Fixed Income), link``."""
    if not sources:
        return ""
    lines = ["**Sources**"]
    for i, source in enumerate(sources, start=1):
        label = f"{source.title} ({source.category})" if source.category else source.title
        lines.append(f"{i}. {label}" + (f" - [reference]({source.url})" if source.url else ""))
    return "\n".join(lines)
