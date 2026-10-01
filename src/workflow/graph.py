"""The LangGraph workflow and the ``FinnieAssistant`` entry point.

    START -> ingest -> router -> check_savings -> [agents in parallel per stage] -> collect
          -> ... -> synthesize -> guard -> summarize -> END
    ingest -> respond_blocked -> summarize              (prohibited or oversized input)
    check_savings -> respond_out_of_scope -> summarize  (not about finance)
    check_savings -> ask_savings -> summarize           (how much of the portfolio counts?)
    ingest -> check_savings                             (the savings answer resumes the question)
    ingest -> ask_savings                               (that answer wasn't clear: ask again)

Conversation memory is a LangGraph checkpointer keyed by ``thread_id`` (one per chat
session), so follow-up questions see earlier turns, the saved profile, and the portfolio.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Mapping
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.base import BaseAgent
from src.agents.context import AgentContext, build_agent_context
from src.agents.registry import build_agents
from src.core.models import Holding, UserProfile
from src.workflow.nodes import Deps, TurnOutput, after_ingest, fan_out, make_nodes
from src.workflow.progress import Progress
from src.workflow.state import FinnieState

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 25


def build_graph(
    context: AgentContext,
    checkpointer: Any | None = None,
    agents: Mapping[str, BaseAgent] | None = None,
) -> Any:
    """``agents`` replaces the registry's specialists (tests inject scripted ones)."""
    deps = Deps(context=context, agents=agents or build_agents(context))
    nodes = make_nodes(deps)
    graph = StateGraph(FinnieState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "ingest")
    graph.add_conditional_edges(
        "ingest", after_ingest, ["router", "respond_blocked", "ask_savings", "check_savings"]
    )
    graph.add_edge("router", "check_savings")
    stage_targets = [*deps.agents, "synthesize", "respond_out_of_scope", "ask_savings"]
    graph.add_conditional_edges("check_savings", fan_out, stage_targets)
    for name in deps.agents:
        graph.add_edge(name, "collect")
    graph.add_conditional_edges("collect", fan_out, stage_targets)
    graph.add_edge("synthesize", "guard")
    for name in ("guard", "respond_blocked", "respond_out_of_scope", "ask_savings"):
        graph.add_edge(name, "summarize")
    graph.add_edge("summarize", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


class FinnieAssistant:
    """One assistant per process; conversations are separated by ``thread_id``."""

    def __init__(
        self,
        context: AgentContext | None = None,
        checkpointer: Any | None = None,
        agents: Mapping[str, BaseAgent] | None = None,
    ):
        self.context = context or build_agent_context()
        self.graph = build_graph(self.context, checkpointer, agents)

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

    @staticmethod
    def _update(
        question: str, profile: UserProfile | None, portfolio: list[Holding] | None
    ) -> dict[str, Any]:
        update: dict[str, Any] = {"messages": [HumanMessage(content=question)]}
        if profile is not None:
            update["profile"] = profile.model_dump()
        if portfolio is not None:
            update["portfolio"] = [h.model_dump() for h in portfolio]
        return update

    @staticmethod
    def _finished(state: dict[str, Any], started: float) -> TurnOutput:
        output = TurnOutput.model_validate(state["output"])
        logger.info(
            "Turn finished",
            extra={
                "status": output.status,
                "agents": output.agents,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return output

    def ask(
        self,
        question: str,
        *,
        thread_id: str,
        profile: UserProfile | None = None,
        portfolio: list[Holding] | None = None,
    ) -> TurnOutput:
        """Answer one message. ``profile``/``portfolio`` update what's saved for the thread."""
        started = time.perf_counter()
        update = self._update(question, profile, portfolio)
        return self._finished(self.graph.invoke(update, self._config(thread_id)), started)

    def stream(
        self,
        question: str,
        *,
        thread_id: str,
        profile: UserProfile | None = None,
        portfolio: list[Holding] | None = None,
    ) -> Iterator[Progress | TurnOutput]:
        """Like ``ask``, but yields ``Progress`` events while the turn runs, then the output."""
        started = time.perf_counter()
        config = self._config(thread_id)
        update = self._update(question, profile, portfolio)
        for event in self.graph.stream(update, config, stream_mode="custom"):
            yield Progress.model_validate(event)
        yield self._finished(self.state(thread_id), started)

    def state(self, thread_id: str) -> dict[str, Any]:
        """The saved conversation state (messages, profile, portfolio, summary)."""
        return dict(self.graph.get_state(self._config(thread_id)).values)

    def mermaid(self) -> str:
        """The compiled graph as a Mermaid diagram (used in the docs)."""
        return str(self.graph.get_graph().draw_mermaid())
