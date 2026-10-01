from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from src.agents.base import AgentRequest
from src.agents.context import AgentContext
from src.core.config import load_settings
from src.core.models import AGENT_NAMES, AgentResult, Freshness, Source
from src.workflow.graph import FinnieAssistant
from tests.fakes.llm import FakeChatModel
from tests.fakes.market_service import FakeMarketService

NOW = datetime(2026, 9, 30, 20, 0, tzinfo=UTC)


def kb_source(article_id: str, title: str | None = None) -> Source:
    return Source(title=title or article_id, kind="knowledge_base", article_id=article_id)


def news_source(url: str, title: str = "Headline") -> Source:
    return Source(title=title, kind="news", url=url)


def market_source(title: str = "TSLA quote (yfinance)") -> Source:
    return Source(title=title, kind="market_data")


def freshness(**kwargs: Any) -> Freshness:
    return Freshness(source="yfinance", as_of=NOW, fetched_at=NOW, **kwargs)


def result(agent: str, answer: str = "", **kwargs: Any) -> AgentResult:
    return AgentResult(agent=agent, answer=answer or f"{agent} answer.", **kwargs)  # type: ignore[arg-type]


def route(*agents: str, **kwargs: Any) -> dict[str, Any]:
    """A structured router reply."""
    return {
        "standalone_query": kwargs.pop("standalone_query", ""),
        "agents": list(agents),
        "confidence": kwargs.pop("confidence", 0.9),
        **kwargs,
    }


class ScriptedAgent:
    """Stands in for a specialist: returns a fixed (or computed) result, records requests."""

    def __init__(self, name: str, reply: AgentResult | Callable[[AgentRequest], AgentResult]):
        self.name = name
        self.reply = reply
        self.requests: list[AgentRequest] = []

    def run(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        if callable(self.reply):
            return self.reply(request)
        return self.reply


@pytest.fixture
def make_context() -> Callable[..., AgentContext]:
    def factory(
        routes: list[Any] | None = None,
        main: list[Any] | None = None,
        fast: list[Any] | None = None,
        settings: Any | None = None,
    ) -> AgentContext:
        return AgentContext(
            llm=FakeChatModel(responses=main or ["merged answer"]),
            fast_llm=FakeChatModel(
                responses=fast or ["summary"], structured_responses=routes or []
            ),
            market=FakeMarketService(),
            settings=settings or load_settings(),
        )

    return factory


@pytest.fixture
def scripted_agents() -> Callable[..., dict[str, ScriptedAgent]]:
    """All six agents scripted; overrides map a name to a result or a function."""

    def factory(**overrides: Any) -> dict[str, ScriptedAgent]:
        return {
            name: ScriptedAgent(name, overrides.get(name) or result(name)) for name in AGENT_NAMES
        }

    return factory


@pytest.fixture
def make_assistant(make_context, scripted_agents):
    def factory(routes: list[Any], agents: dict[str, Any] | None = None, **context_kwargs: Any):
        context = make_context(routes=routes, **context_kwargs)
        team = scripted_agents(**(agents or {}))
        return FinnieAssistant(context=context, agents=team), team

    return factory
