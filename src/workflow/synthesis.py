"""Combine one or more agent results into a single answer with one consistent citation list.

Each agent numbers its own citations (``[1]`` for its first knowledge base passage,
``[N1]`` for its first news article). Before combining, every marker is rewritten to a
single global numbering over the merged sources list, so ``[3]`` in the final answer is
always the third entry under "Sources", whichever agent wrote it.

With one successful agent, its renumbered answer is used directly (no extra LLM call).
With several, the main model merges them into one coherent answer, keeping the markers.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.base import message_text, source_key
from src.core.models import AgentResult, Freshness, Source
from src.rag.citations import check_citations

logger = logging.getLogger(__name__)

MARKER = re.compile(r"\[(N?)(\d{1,2})\]")
AGENT_LABELS = {
    "finance_qa": "Concepts",
    "portfolio": "Your portfolio",
    "market": "Markets",
    "goal_planning": "Your goal",
    "news": "News",
    "tax": "Taxes",
}
MERGE_PROMPT = """You combine answers from Finnie's specialist agents into one reply to the user.

Rules:
- Keep every fact and number exactly as given, and keep citation markers like [3] exactly as \
they appear, attached to the same facts. Don't add new markers or facts.
- Remove repetition, put the parts in a sensible order, and use short markdown headings when \
the answer covers several topics.
- Keep the educational tone and any caveats about data freshness or uncertainty. Don't add \
recommendations to buy, sell, or allocate.
- Don't add a disclaimer or a sources list; those are added separately."""


class Citations(BaseModel):
    """The merged sources list and each agent's answer rewritten to use it."""

    sources: list[Source] = Field(default_factory=list)
    answers: dict[str, str] = Field(default_factory=dict)


def unify_citations(results: Sequence[AgentResult]) -> Citations:
    merged: list[Source] = []
    position: dict[str, int] = {}

    def number_for(source: Source) -> int:
        key = source_key(source)
        if key not in position:
            merged.append(source)
            position[key] = len(merged)
        return position[key]

    answers: dict[str, str] = {}
    for result in results:
        by_key = {source_key(s): s for s in result.sources}
        maps = result.data.get("citations", {})

        def rewrite(
            match: re.Match[str], maps: dict[str, Any] = maps, by_key: dict[str, Source] = by_key
        ) -> str:
            kind = "news" if match.group(1) else "kb"
            key = maps.get(kind, {}).get(match.group(2))
            if key is None or key not in by_key:
                return ""  # a marker we can't resolve to a source is dropped
            return f"[{number_for(by_key[key])}]"

        answers[result.agent] = MARKER.sub(rewrite, result.answer)
    # Market data isn't listed as a source: the answer's freshness note covers it.
    return Citations(sources=merged, answers=answers)


MARKER_GROUP = re.compile(r"(?:\[\d{1,2}\])+")


def tidy_citations(text: str, sources: Sequence[Source]) -> tuple[str, list[Source]]:
    """Keep only cited sources, numbered in order of first use, with no repeated markers.

    "[1][2][1]" becomes "[1][2]", and a source no marker points to is dropped, so the
    sources list matches the answer exactly (also after the merge or a guardrail rewrite).
    """
    order: list[int] = []

    def renumber(match: re.Match[str]) -> str:
        numbers: list[int] = []
        for raw in re.findall(r"\d+", match.group(0)):
            n = int(raw)
            if not 1 <= n <= len(sources):
                continue
            if n not in order:
                order.append(n)
            new = order.index(n) + 1
            if new not in numbers:
                numbers.append(new)
        return "".join(f"[{n}]" for n in numbers)

    tidied = MARKER_GROUP.sub(renumber, text)
    return tidied, [sources[n - 1] for n in order]


class Synthesis(BaseModel):
    text: str
    sources: list[Source]
    freshness: list[Freshness]
    agents: list[str]
    merged_by_llm: bool = False


def synthesize(results: Sequence[AgentResult], llm: Any | None) -> Synthesis | None:
    """``None`` when no agent produced an answer (the workflow's fallback handles it)."""
    ok = [r for r in results if r.ok and r.answer]
    if not ok:
        return None
    citations = unify_citations(ok)
    freshness = [f for r in ok for f in r.freshness]
    agents = [r.agent for r in ok]
    if len(ok) == 1:
        text = check_citations(citations.answers[ok[0].agent], len(citations.sources)).text
        text, sources = tidy_citations(text, citations.sources)
        return Synthesis(text=text, sources=sources, freshness=freshness, agents=agents)

    sections = "\n\n".join(
        f"## {AGENT_LABELS.get(agent, agent)}\n{answer}"
        for agent, answer in citations.answers.items()
    )
    merged_by_llm = False
    text = sections
    if llm is not None:
        try:
            reply = llm.invoke(
                [SystemMessage(content=MERGE_PROMPT), HumanMessage(content=sections)]
            )
            candidate = message_text(reply).strip()
            if candidate:
                text, merged_by_llm = candidate, True
        except Exception:
            logger.exception("Merge failed; showing the specialists' answers as sections")
    text = check_citations(text, len(citations.sources)).text  # drop anything invented
    text, sources = tidy_citations(text, citations.sources)
    return Synthesis(
        text=text,
        sources=sources,
        freshness=freshness,
        agents=agents,
        merged_by_llm=merged_by_llm,
    )
