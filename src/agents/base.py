"""The shared agent contract and run loop.

Each specialist agent declares its knowledge base categories, its tools, and a prompt.
``BaseAgent.run`` then:

1. retrieves knowledge base passages for the question (filtered to the agent's categories)
   and numbers them as citable context blocks;
2. runs a tool-calling loop, at most ``workflow.agent_max_iterations`` rounds;
3. strips citations that don't match a context block;
4. returns an ``AgentResult`` with the answer, sources, chart data, data freshness, and
   any hand-off it requested.

``run`` never raises: a failure becomes ``AgentResult(error=...)`` so the workflow can
carry on with the other agents.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from src.core.guardrails import wrap_untrusted
from src.core.models import AgentName, AgentResult, Freshness, Holding, Source, UserProfile
from src.rag.citations import (
    ContextBlock,
    build_context,
    check_citations,
    check_news_citations,
    cited_sources,
)
from src.rag.retriever import RetrievedChunk

if TYPE_CHECKING:
    from src.agents.context import AgentContext

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
FINAL_ANSWER_NUDGE = "Please give your final answer now, using the information gathered so far."
FALLBACK_ANSWER = (
    "I wasn't able to put together a complete answer this time. Could you try rephrasing "
    "the question?"
)

LEVEL_GUIDANCE = {
    "beginner": (
        "The user is a beginner. Write the way you'd explain it to a smart friend with no "
        "finance background:\n"
        "- Define every financial term in plain words in the same sentence where it first "
        'appears, e.g. "an expense ratio, the yearly fee a fund charges". That includes '
        "terms like ETF, index, volatility, beta, Sharpe ratio, moving average, and 401(k).\n"
        "- If they ask for normal or simple words, use everyday language with no formulas.\n"
        "- Keep it short: about 150 words. Use at most two short headings (### only), or "
        "none; prefer short paragraphs and up to five bullets.\n"
        "- Include one small, concrete example."
    ),
    "intermediate": "The user knows the basics: skip elementary definitions and be concise.",
    "advanced": "The user is experienced: be precise and concise; technical terms are fine.",
}


class AgentRequest(BaseModel):
    """Everything an agent needs to answer one question."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str
    profile: UserProfile = Field(default_factory=UserProfile)
    portfolio: list[Holding] | None = None
    history: list[BaseMessage] = Field(default_factory=list)
    prior_results: dict[str, AgentResult] = Field(default_factory=dict)
    tickers: list[str] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list, description="Extra system instructions")
    retrieval_query: str | None = Field(
        default=None, description="Search the knowledge base with this instead of the query"
    )
    user_context: list[str] = Field(
        default_factory=list,
        description="Facts Finnie knows about the user, shown with their question",
    )
    allow_handoff: bool = Field(
        default=True, description="False for an agent that is itself answering a hand-off"
    )


@dataclass
class RunState:
    """Per-run scratchpad shared by the agent and its tools."""

    request: AgentRequest
    agent: AgentName
    blocks: list[ContextBlock] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    freshness: list[Freshness] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    handoffs: list[AgentName] = field(default_factory=list)
    news: list[Source] = field(default_factory=list)  # citable as [N1], [N2], ...
    tool_calls: list[str] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)

    def add_chunks(self, chunks: list[RetrievedChunk]) -> list[ContextBlock]:
        """Number new chunks after the existing blocks; skip chunks already present."""
        present = {b.chunk.chunk.id for b in self.blocks}
        fresh = [c for c in chunks if c.chunk.id not in present]
        _, blocks = build_context(fresh)
        offset = len(self.blocks)
        renumbered = [b.model_copy(update={"number": b.number + offset}) for b in blocks]
        self.blocks.extend(renumbered)
        return renumbered


def format_blocks(blocks: list[ContextBlock]) -> str:
    """Knowledge base passages as numbered text for the prompt: "[n] Title > Section"."""
    return "\n\n".join(
        f"[{b.number}] {b.chunk.chunk.title} > {b.chunk.chunk.section}\n{b.chunk.chunk.text}"
        for b in blocks
    )


def _with_context(request: AgentRequest) -> str:
    """The question, plus facts about the user that the answer should build on.

    Facts sit next to the question rather than in the system prompt, because models give
    the user's turn more weight.
    """
    if not request.user_context:
        return request.query
    notes = "\n".join(f"- {fact}" for fact in request.user_context)
    return f"{request.query}\n\n(Context Finnie has about me:\n{notes})"


def source_key(source: Source) -> str:
    """Stable identity for a source across agents: article id, else URL, else title."""
    return source.article_id or source.url or source.title


def used_news(news: list[Source], cited: list[int], answer: str) -> list[Source]:
    """News articles the answer used: cited as [N#], or (as a fallback) named by title."""
    lowered = answer.lower()
    return [
        source
        for number, source in enumerate(news, start=1)
        if number in cited or source.title[:40].lower() in lowered
    ]


@cache
def load_prompt(name: str) -> str:
    """A prompt file from ``src/agents/prompts/`` (``_policy`` is shared by every agent)."""
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def message_text(message: BaseMessage) -> str:
    """Plain text of a message whose content may be a string or a list of blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


class BaseAgent:
    """Base class for the specialist agents.

    Subclasses only set ``name``, ``description``, ``rag_categories`` (``None`` searches
    every category), and ``tool_names``; prompting, retrieval, and the tool loop live here.
    """

    name: ClassVar[AgentName]
    description: ClassVar[str]
    rag_categories: ClassVar[tuple[str, ...] | None] = ()  # None = all categories
    tool_names: ClassVar[tuple[str, ...]] = ()

    def __init__(self, context: AgentContext) -> None:
        self.context = context

    # ---- prompt -------------------------------------------------------------------------

    def system_prompt(self, state: RunState) -> str:
        """The full system prompt for one run, built from:

        - the shared education policy and this agent's prompt
        - today's date
        - guidance for the user's knowledge level
        - the user's profile and saved portfolio
        - findings from earlier specialists in this turn
        - the retrieved passages, to cite as [n]

        Text that came from outside (passages, other agents' findings) is fenced as
        untrusted, so instructions inside it aren't followed.
        """
        request = state.request
        profile = request.profile
        parts = [
            load_prompt("_policy"),
            load_prompt(self.name),
            f"Today is {date.today():%B %d, %Y}.",
            LEVEL_GUIDANCE[profile.knowledge_level],
            f"The user's risk tolerance is {profile.risk_tolerance}"
            + (f"; age {profile.age}" if profile.age else "")
            + (
                f"; investment horizon {profile.investment_horizon_years} years"
                if profile.investment_horizon_years is not None
                else ""
            )
            + ".",
        ]
        if request.portfolio:
            holdings = ", ".join(f"{h.ticker} x {h.shares:g}" for h in request.portfolio)
            parts.append(f"The user's saved portfolio: {holdings}.")
        if request.prior_results:
            summaries = "\n".join(
                f"- {name}: {result.answer[:600]}"
                for name, result in request.prior_results.items()
                if result.ok
            )
            if summaries:
                parts.append(
                    "Findings from other Finnie specialists earlier in this turn (use them; "
                    "don't repeat their work):\n" + wrap_untrusted("specialist_findings", summaries)
                )
        parts.extend(request.guidance)
        if state.blocks:
            parts.append(
                "Knowledge base passages. Cite them inline as [n] when you use them, and only "
                "use these numbers:\n"
                + wrap_untrusted("knowledge_base_passages", format_blocks(state.blocks))
            )
        else:
            parts.append(
                "No knowledge base passage matched this question. If you answer from general "
                "knowledge, say that it isn't from Finnie's knowledge base, and don't use [n] "
                "citations."
            )
        return "\n\n".join(parts)

    # ---- run ----------------------------------------------------------------------------

    def run(self, request: AgentRequest) -> AgentResult:
        """Answer one request, never raising.

        The steps:
        1. Retrieve passages from this agent's categories.
        2. Run the bounded tool-calling loop.
        3. Drop citations that point at nothing.
        4. Record which source each surviving citation refers to, so the workflow can
           renumber them when merging answers.

        A failure comes back as ``AgentResult(error=...)``, not an exception, so one
        agent failing doesn't sink the turn.
        """
        started = time.perf_counter()
        state = RunState(request=request, agent=self.name)
        try:
            self._retrieve(state)
            answer = self._loop(state)
        except Exception as exc:
            logger.exception("Agent %s failed", self.name, extra={"agent": self.name})
            return AgentResult(
                agent=self.name, error=f"{type(exc).__name__}: {exc}"[:300], data=state.data
            )
        check = check_citations(answer, len(state.blocks))
        news_check = check_news_citations(check.text, len(state.news))
        removed = check.removed + [f"N{n}" for n in news_check.removed]
        if removed:
            logger.info("Removed invalid citations %s", removed, extra={"agent": self.name})
        # Which source each surviving marker points to, so the workflow can renumber
        # citations when it combines several agents' answers into one.
        by_block = {b.number: b.source for b in state.blocks}
        state.data["citations"] = {
            "kb": {str(n): source_key(by_block[n]) for n in check.cited},
            "news": {str(n): source_key(state.news[n - 1]) for n in news_check.cited},
        }
        state.data.setdefault("meta", {}).update(
            {
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "tool_calls": state.tool_calls,
                "tool_errors": state.tool_errors,
                "kb_blocks": len(state.blocks),
            }
        )
        return AgentResult(
            agent=self.name,
            answer=news_check.text.strip() or FALLBACK_ANSWER,
            sources=cited_sources(state.blocks, check.cited)
            + state.sources
            + used_news(state.news, news_check.cited, news_check.text),
            data=state.data,
            freshness=state.freshness,
            handoff=state.handoffs,
        )

    def _retrieve(self, state: RunState) -> None:
        retriever = self.context.retriever
        if retriever is None or self.rag_categories == ():
            return
        try:
            result = retriever.retrieve(
                state.request.retrieval_query or state.request.query,
                categories=list(self.rag_categories) if self.rag_categories else None,
            )
        except Exception:
            logger.exception("Retrieval failed; continuing without knowledge base context")
            return
        state.add_chunks(result.chunks)
        state.data["kb"] = {"confident": result.confident, "widened": result.widened}

    def _tools(self, state: RunState) -> dict[str, BaseTool]:
        from src.agents.tools import build_tools

        names = self.tool_names
        if not state.request.allow_handoff:
            names = tuple(n for n in names if n != "request_handoff")
        return build_tools(names, self.context, state)

    def _loop(self, state: RunState) -> str:
        tools = self._tools(state)
        model = self.context.llm.bind_tools(list(tools.values())) if tools else self.context.llm
        window = self.context.settings.workflow.history_window
        messages: list[BaseMessage] = [
            SystemMessage(content=self.system_prompt(state)),
            *state.request.history[-window:],
            HumanMessage(content=_with_context(state.request)),
        ]
        for _ in range(self.context.settings.workflow.agent_max_iterations):
            reply = model.invoke(messages)
            messages.append(reply)
            if not isinstance(reply, AIMessage) or not reply.tool_calls:
                return message_text(reply)
            for call in reply.tool_calls:
                messages.append(self._execute(call, tools, state))
        # Out of tool rounds: ask for a final answer from what's been gathered.
        final = model.invoke([*messages, HumanMessage(content=FINAL_ANSWER_NUDGE)])
        return message_text(final)

    def _execute(self, call: ToolCall, tools: dict[str, BaseTool], state: RunState) -> ToolMessage:
        name = call["name"]
        state.tool_calls.append(name)
        tool = tools.get(name)
        if tool is None:
            state.tool_errors.append(f"{name}: unknown tool")
            content = f"Error: there is no tool named {name}."
        else:
            try:
                content = str(tool.invoke(call["args"]))
            except Exception as exc:
                logger.warning("Tool %s failed: %s", name, exc, extra={"agent": self.name})
                state.tool_errors.append(f"{name}: {type(exc).__name__}")
                content = f"Error: {name} failed ({type(exc).__name__}: {exc})"[:500]
        return ToolMessage(content=content, tool_call_id=call["id"], name=name)
