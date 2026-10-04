"""Graph state for one conversation.

Everything except ``messages`` is plain JSON-serializable data (pydantic models are dumped
at node boundaries), so the checkpointer can save and restore it without custom types.
Fields marked "per turn" are reset by the ``ingest`` node at the start of every turn.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

RESET = "__reset__"


def merge_results(current: dict[str, Any] | None, update: dict[str, Any] | None) -> dict[str, Any]:
    """Parallel agents each add their own key. ``{RESET: True}`` clears the dict."""
    if update is None:
        return current or {}
    if update.get(RESET):
        return {}
    return {**(current or {}), **update}


class FinnieState(TypedDict, total=False):
    """The LangGraph state for one conversation thread (see the module docstring)."""

    # Persist across turns (checkpointed per conversation thread)
    messages: Annotated[list[AnyMessage], add_messages]
    profile: dict[str, Any]  # UserProfile
    portfolio: list[dict[str, Any]] | None  # list[Holding]
    summary: str | None  # rolling summary of older turns
    title: str | None  # short conversation title for the sidebar
    title_final: bool  # rewritten after the third question; fixed from then on
    goal_savings: dict[str, dict[str, Any]]  # goal label -> GoalSavings
    pending_savings: dict[str, Any] | None  # the goal question waiting on the savings answer

    # Per turn
    question: str  # the user's latest message, as written
    query: str  # standalone version (follow-ups resolved)
    screen: dict[str, Any]  # InputScreen
    route: dict[str, Any] | None  # RouteDecision
    plan: list[list[str]]  # stages of agent names; agents within a stage run in parallel
    stage: int
    handoff_used: bool
    deadline: float  # wall-clock time (epoch seconds) by which agents must finish
    savings_prompt: str | None  # set when this turn asks the savings question
    results: Annotated[dict[str, Any], merge_results]  # agent name -> AgentResult
    output: dict[str, Any] | None  # TurnOutput


class AgentTask(TypedDict):
    """What the dispatcher sends to an agent node."""

    agent: str
    allow_handoff: bool
    state: FinnieState
