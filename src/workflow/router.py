"""Decide which specialist agent(s) should answer, and rewrite follow-ups to stand alone.

The primary router is one structured-output call to the fast model. If that call fails
or returns something unusable, ``keyword_route`` decides deterministically, so a turn
always has a route.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from src.core.models import AGENT_NAMES, AgentName, normalize_ticker

logger = logging.getLogger(__name__)

ROUTER_PROMPT = (Path(__file__).parent / "prompts" / "router.md").read_text(encoding="utf-8")


class RouteDecision(BaseModel):
    """Which specialists should answer the latest message, and what the router extracted."""

    standalone_query: str = Field(description="The latest message rewritten to stand alone")
    agents: list[AgentName] = Field(default_factory=list, description="1-3 specialists")
    depends_on: dict[str, list[str]] = Field(default_factory=dict)
    tickers: list[str] = Field(default_factory=list)
    out_of_scope: bool = False
    confidence: float = Field(default=0.5, ge=0, le=1)
    goal: str | None = Field(default=None, description="Short label for a savings goal")
    current_savings: float | None = Field(
        default=None, ge=0, description="Savings toward the goal stated in the latest message"
    )
    source: str = Field(default="llm", description="llm or keyword (fallback)")

    @field_validator("tickers")
    @classmethod
    def _clean_tickers(cls, values: list[str]) -> list[str]:
        cleaned = []
        for value in values:
            try:
                ticker = normalize_ticker(value)
            except ValueError:
                continue
            if ticker not in cleaned:
                cleaned.append(ticker)
        return cleaned


# ---- keyword fallback ---------------------------------------------------------------------

KEYWORDS: dict[AgentName, re.Pattern[str]] = {
    "portfolio": re.compile(
        r"\b(my (portfolio|holdings|investments|allocation|stocks|funds)|i (own|hold|have) \d|"
        r"diversif\w*|allocation|rebalanc\w*|concentrat\w*|holdings?)\b",
        re.I,
    ),
    "market": re.compile(
        r"(\$[A-Z]{1,5}\b|\b(price|quote|trading at|stock market|market today|how is the market|"
        r"s&p|nasdaq|dow|sector|52-week|moving average|rsi|trend)\b)",
        re.I,
    ),
    "goal_planning": re.compile(
        r"\b(goal|retire\w*|on track|save for|saving for|down payment|college fund|"
        r"how much (should|do) i (save|need)|monte carlo|projection|by (age )?\d{2}\b)",
        re.I,
    ),
    "news": re.compile(
        r"\b(news|headlines?|why did .* (drop|fall|rise|jump)|today'?s move)\b", re.I
    ),
    "tax": re.compile(
        r"\b(tax\w*|capital gains?|ira|roth|401\(?k\)?|403\(?b\)?|hsa|529|wash sale|"
        r"deduct\w*|rmd|required minimum|irs|harvest\w*)\b",
        re.I,
    ),
}
TICKER_HINT = re.compile(r"\$([A-Za-z]{1,5})\b|\b([A-Z]{2,5})\b")
NOT_TICKERS = frozenset(
    {
        "I",
        "A",
        "IRA",
        "ETF",
        "ETFS",
        "IRS",
        "HSA",
        "RMD",
        "USA",
        "US",
        "FDIC",
        "SEC",
        "FINRA",
        "CPI",
        "GDP",
        "APR",
        "APY",
        "CD",
        "CDS",
        "IPO",
        "TIPS",
        "REIT",
        "OK",
        "AI",
        "FAQ",
        "PE",
    }
)


HOLDINGS = re.compile(r"\bi (own|hold|have) \d", re.IGNORECASE)


def with_portfolio_for_goals(
    agents: Sequence[str], has_holdings: bool, max_agents: int
) -> list[str]:
    """A goal question that lists holdings needs the portfolio agent to value them first.

    A saved portfolio alone doesn't add it: the user is asked how much of it counts
    toward the goal (``src/workflow/savings.py``).
    """
    if "goal_planning" not in agents or "portfolio" in agents or not has_holdings:
        return list(agents)
    extended = ["portfolio", *agents]
    if len(extended) > max_agents:  # drop the least relevant other agent, never the goal
        extended = [a for a in extended if a in ("portfolio", "goal_planning")] + [
            a for a in extended if a not in ("portfolio", "goal_planning")
        ][: max_agents - 2]
    return extended


def extract_tickers(text: str) -> list[str]:
    """Likely ticker symbols in the text (``$AAPL`` or bare capitals), deduplicated, at most 10."""
    found = []
    for dollar, bare in TICKER_HINT.findall(text):
        candidate = (dollar or bare).upper()
        if candidate in NOT_TICKERS or candidate in found:
            continue
        found.append(candidate)
    return found[:10]


def keyword_route(query: str, max_agents: int = 3) -> RouteDecision:
    """Deterministic routing by keyword scores; finance_qa when nothing matches."""
    scores = {name: len(pattern.findall(query)) for name, pattern in KEYWORDS.items()}
    ranked = [name for name, score in sorted(scores.items(), key=lambda kv: -kv[1]) if score > 0]
    tickers = extract_tickers(query)
    if tickers and not ranked:
        ranked.append("market")
    agents = with_portfolio_for_goals(
        ranked[:max_agents] or ["finance_qa"], bool(HOLDINGS.search(query)), max_agents
    )
    return RouteDecision(
        standalone_query=query,
        agents=agents,
        tickers=tickers,
        confidence=0.4,
        source="keyword",
    )


# ---- LLM router ---------------------------------------------------------------------------


def _context_lines(
    history: list[BaseMessage], summary: str | None, has_portfolio: bool, goals: Sequence[str]
) -> str:
    lines = []
    if summary:
        lines.append(f"Summary of earlier conversation: {summary}")
    for message in history:
        role = "User" if isinstance(message, HumanMessage) else "Finnie"
        text = message.content if isinstance(message.content, str) else str(message.content)
        lines.append(f"{role}: {text[:400]}")
    lines.append(f"The user {'has' if has_portfolio else 'has not'} saved a portfolio.")
    if goals:
        lines.append(f"Goal labels already used in this conversation: {', '.join(goals)}.")
    return "\n".join(lines)


def route_with_llm(
    llm: Any,
    question: str,
    *,
    history: list[BaseMessage],
    summary: str | None,
    has_portfolio: bool,
    max_agents: int,
    min_confidence: float,
    goals: Sequence[str] = (),
) -> RouteDecision:
    """LLM routing with validation; falls back to keywords on any failure."""
    try:
        # function calling: OpenAI's strict JSON-schema mode rejects the free-form
        # ``depends_on`` mapping
        structured = llm.with_structured_output(RouteDecision, method="function_calling")
        decision = structured.invoke(
            [
                SystemMessage(content=ROUTER_PROMPT),
                HumanMessage(
                    content=(
                        "Conversation so far:\n"
                        + _context_lines(history, summary, has_portfolio, goals)
                        + f"\n\nLatest user message: {question}"
                    )
                ),
            ]
        )
        if isinstance(decision, dict):
            decision = RouteDecision.model_validate(decision)
    except Exception as exc:
        logger.warning("LLM router failed (%s); using keyword routing", type(exc).__name__)
        return keyword_route(question, max_agents)

    agents = list(dict.fromkeys(a for a in decision.agents if a in AGENT_NAMES))[:max_agents]
    if decision.out_of_scope:
        return decision.model_copy(update={"agents": []})
    if not agents or decision.confidence < min_confidence:
        agents = agents[:1] or ["finance_qa"]
    # A first message already stands alone; a rewrite could only lose detail from it.
    standalone = (decision.standalone_query.strip() if history else "") or question
    agents = with_portfolio_for_goals(agents, bool(HOLDINGS.search(question)), max_agents)
    return decision.model_copy(update={"agents": agents, "standalone_query": standalone})
