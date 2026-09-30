"""Shared data types passed between agents, tools, the workflow, and the UI."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AgentName = Literal["finance_qa", "portfolio", "market", "goal_planning", "news", "tax"]
AGENT_NAMES: tuple[AgentName, ...] = (
    "finance_qa",
    "portfolio",
    "market",
    "goal_planning",
    "news",
    "tax",
)

KnowledgeLevel = Literal["beginner", "intermediate", "advanced"]
RiskTolerance = Literal["conservative", "moderate", "aggressive"]
DataSource = Literal["alpha_vantage", "yfinance", "cache", "mock"]
FreshnessStatus = Literal["live", "cached", "stale", "mock"]

TICKER_PATTERN = re.compile(r"^\^?[A-Z][A-Z0-9.\-]{0,9}$")


def normalize_ticker(value: str) -> str:
    """Upper-case and validate a ticker symbol (e.g. ``brk.b`` -> ``BRK.B``, ``^GSPC``)."""
    ticker = value.strip().upper().lstrip("$")
    if not TICKER_PATTERN.match(ticker):
        raise ValueError(f"Invalid ticker symbol: {value!r}")
    return ticker


class UserProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knowledge_level: KnowledgeLevel = "beginner"
    risk_tolerance: RiskTolerance = "moderate"
    age: int | None = Field(default=None, ge=13, le=120)
    investment_horizon_years: int | None = Field(default=None, ge=0, le=80)


class Holding(BaseModel):
    """One position in a user's portfolio."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    shares: float = Field(gt=0)
    cost_basis: float | None = Field(default=None, ge=0, description="Total cost, not per share")

    @field_validator("ticker")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return normalize_ticker(value)


class Source(BaseModel):
    """A citation attached to an answer."""

    title: str
    kind: Literal["knowledge_base", "market_data", "news"]
    category: str | None = None
    url: str | None = None
    article_id: str | None = None
    score: float | None = None
    published_at: datetime | None = None


class Freshness(BaseModel):
    """Where a piece of market data came from and how old it is."""

    source: DataSource
    as_of: datetime
    fetched_at: datetime
    is_stale: bool = False
    is_mock: bool = False

    @model_validator(mode="after")
    def _mock_source_is_mock(self) -> Freshness:
        if self.source == "mock":
            self.is_mock = True
        return self

    @property
    def status(self) -> FreshnessStatus:
        if self.is_mock:
            return "mock"
        if self.is_stale:
            return "stale"
        return "cached" if self.source == "cache" else "live"

    def age_minutes(self, now: datetime | None = None) -> float:
        now = now or datetime.now(UTC)
        return max(0.0, (now - self.fetched_at).total_seconds() / 60)

    def label(self, now: datetime | None = None) -> str:
        """Short human-readable badge text, e.g. ``Live · 3 min ago``."""
        if self.status == "mock":
            return "Demo data: live feed unavailable"
        age = _format_age(self.age_minutes(now))
        prefix = {"live": "Live", "cached": "Cached", "stale": "Stale"}[self.status]
        return f"{prefix} · {age}"


def _format_age(minutes: float) -> str:
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{int(minutes)} min ago"
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)} h ago"
    return f"{int(hours / 24)} d ago"


class AgentResult(BaseModel):
    """The output contract every agent returns to the workflow."""

    agent: AgentName
    answer: str = ""
    sources: list[Source] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    freshness: list[Freshness] = Field(default_factory=list)
    handoff: list[AgentName] = Field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
