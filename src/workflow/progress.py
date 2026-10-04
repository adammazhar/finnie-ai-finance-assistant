"""Progress events for the UI while a turn runs ("Consulting the portfolio specialist...").

Nodes call ``emit``, which writes to LangGraph's custom stream. ``FinnieAssistant.stream``
yields the events as they happen; under ``invoke`` the writer does nothing.
"""

from __future__ import annotations

from typing import Literal

from langgraph.config import get_stream_writer
from pydantic import BaseModel

SPECIALISTS = {
    "finance_qa": "financial concepts",
    "portfolio": "portfolio",
    "market": "markets",
    "goal_planning": "goal planning",
    "news": "news",
    "tax": "tax",
}


class Progress(BaseModel):
    """One progress event as streamed to the UI."""

    kind: Literal["status", "agent_started", "agent_finished"]
    message: str
    agent: str | None = None
    ok: bool = True


def specialist(agent: str) -> str:
    """A readable name for an agent, e.g. "goal planning"."""
    return SPECIALISTS.get(agent, agent.replace("_", " "))


def emit(
    kind: Literal["status", "agent_started", "agent_finished"],
    message: str,
    agent: str | None = None,
    ok: bool = True,
) -> None:
    """Write a progress event to the graph's custom stream (a no-op outside streaming)."""
    get_stream_writer()(Progress(kind=kind, message=message, agent=agent, ok=ok).model_dump())


def status(message: str) -> None:
    """Emit a general status message."""
    emit("status", message)


def agent_started(agent: str) -> None:
    """Emit that a specialist has started."""
    emit("agent_started", f"Consulting the {specialist(agent)} specialist…", agent=agent)


def agent_finished(agent: str, ok: bool) -> None:
    """Emit that a specialist has finished, or ran into a problem when ``ok`` is false."""
    label = specialist(agent).capitalize()
    message = f"{label} specialist finished" if ok else f"{label} specialist ran into a problem"
    emit("agent_finished", message, agent=agent, ok=ok)
