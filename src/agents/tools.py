"""LangChain tools for the agents, wrapping Finnie's domain code.

Each tool returns a compact text summary for the model, and records structured results
(chart data, freshness, sources) on the run's ``RunState`` for the UI. The same domain
functions back the MCP server, so the logic exists once.

Tools are built per run (``build_tools``) so they can write to that run's state.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from src.agents.base import RunState, format_blocks
from src.agents.context import AgentContext
from src.core.guardrails import sanitize_untrusted, wrap_untrusted
from src.core.indicators import build_market_overview, technical_snapshot
from src.core.models import AGENT_NAMES, Holding, Source
from src.core.monte_carlo import (
    LOW_ODDS,
    chance_text,
    inputs_for_profile,
    low_odds_note,
    required_monthly_contribution,
    simulate,
)
from src.core.portfolio import expense_ratio_label, fetch_and_analyze
from src.core.tax import compare_accounts, illustrate_capital_gains
from src.rag.chunking import GLOSSARY_CATEGORY
from src.rag.knowledge_base import CATEGORIES

ToolFactory = Callable[[AgentContext, RunState], BaseTool]
_FACTORIES: dict[str, ToolFactory] = {}


def _register(fn: ToolFactory) -> ToolFactory:
    _FACTORIES[fn.__name__.removeprefix("make_")] = fn
    return fn


def build_tools(
    names: tuple[str, ...], context: AgentContext, state: RunState
) -> dict[str, BaseTool]:
    unknown = [n for n in names if n not in _FACTORIES]
    if unknown:
        raise KeyError(f"Unknown tools: {unknown}")
    return {name: _FACTORIES[name](context, state) for name in names}


def available_tools() -> list[str]:
    return sorted(_FACTORIES)


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}%}"


# ---- knowledge base --------------------------------------------------------------------


class SearchArgs(BaseModel):
    query: str = Field(description="What to look up in Finnie's knowledge base")
    category: str | None = Field(
        default=None, description=f"Optional category: one of {', '.join(CATEGORIES)}"
    )


@_register
def make_search_knowledge_base(context: AgentContext, state: RunState) -> BaseTool:
    def search(query: str, category: str | None = None) -> str:
        if context.retriever is None:
            return "The knowledge base is unavailable right now."
        categories = [category] if category in CATEGORIES else None
        result = context.retriever.retrieve(query, categories=categories)
        blocks = state.add_chunks(result.chunks)
        if not blocks:
            return "No new knowledge base passages matched. Don't cite anything for this."
        return "New citable passages:\n" + wrap_untrusted(
            "knowledge_base_passages", format_blocks(blocks)
        )

    return StructuredTool.from_function(
        func=search,
        name="search_knowledge_base",
        description="Search Finnie's financial education articles. Returns numbered passages "
        "you can cite as [n].",
        args_schema=SearchArgs,
    )


class GlossaryArgs(BaseModel):
    term: str = Field(description="A financial term, e.g. 'expense ratio'")


@_register
def make_lookup_glossary_term(context: AgentContext, state: RunState) -> BaseTool:
    def lookup(term: str) -> str:
        if context.retriever is None:
            return "The glossary is unavailable right now."
        result = context.retriever.retrieve(
            term, categories=[GLOSSARY_CATEGORY], k=2, include_glossary=True
        )
        glossary = [c for c in result.chunks if c.chunk.kind == "glossary"]
        blocks = state.add_chunks(glossary)
        if not blocks:
            return f"No glossary entry closely matches '{term}'."
        return wrap_untrusted("knowledge_base_passages", format_blocks(blocks))

    return StructuredTool.from_function(
        func=lookup,
        name="lookup_glossary_term",
        description="Look up a definition in Finnie's glossary. Returns citable passages.",
        args_schema=GlossaryArgs,
    )


# ---- market data -----------------------------------------------------------------------


def _market_source(label: str, detail: str) -> Source:
    return Source(title=f"{label} ({detail})", kind="market_data")


class TickersArgs(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=15, description="Ticker symbols")


@_register
def make_get_quotes(context: AgentContext, state: RunState) -> BaseTool:
    def get_quotes(tickers: list[str]) -> str:
        batch = context.market.get_quotes(tickers)
        lines = []
        for ticker, quote in batch.quotes.items():
            change = f"{quote.change_percent:+.2f}%" if quote.change_percent is not None else "n/a"
            lines.append(f"{ticker}: ${quote.price:,.2f} ({change}), {quote.freshness.label()}")
            state.freshness.append(quote.freshness)
            state.data.setdefault("quotes", {})[ticker] = quote.model_dump(mode="json")
        for ticker, error in batch.errors.items():
            lines.append(f"{ticker}: unavailable ({error})")
        if batch.quotes:
            state.sources.append(_market_source("Market quotes", ", ".join(batch.quotes)))
        return "\n".join(lines) or "No quotes were available."

    return StructuredTool.from_function(
        func=get_quotes,
        name="get_quotes",
        description="Latest prices and daily change for one or more tickers, with data freshness.",
        args_schema=TickersArgs,
    )


@_register
def make_get_market_overview(context: AgentContext, state: RunState) -> BaseTool:
    def get_market_overview() -> str:
        overview = build_market_overview(context.market)
        state.data["market_overview"] = overview.model_dump(mode="json")
        movers = overview.indices + overview.sectors
        state.freshness.extend(m.freshness for m in movers)
        lines = ["Indices:"]
        lines += [
            f"- {m.name} ({m.ticker}): {m.change_percent:+.2f}%"
            for m in overview.indices
            if m.change_percent is not None
        ]
        lines.append(f"Mood: {overview.mood.label}; volatility {overview.mood.volatility_regime}.")
        lines += overview.mood.summary
        if overview.benchmark:
            lines += overview.benchmark.notes
        if overview.errors:
            lines.append(f"Unavailable: {', '.join(overview.errors)}")
        if movers:
            state.sources.append(_market_source("Market overview", movers[0].freshness.label()))
        return "\n".join(lines)

    return StructuredTool.from_function(
        func=get_market_overview,
        name="get_market_overview",
        description="Snapshot of major index proxies, the 11 sectors, and an S&P 500 "
        "technical read.",
    )


class TickerArgs(BaseModel):
    ticker: str = Field(description="One ticker symbol")


@_register
def make_get_technical_snapshot(context: AgentContext, state: RunState) -> BaseTool:
    def get_technical_snapshot(ticker: str) -> str:
        history = context.market.get_daily_history(ticker, context.settings.analytics.history_days)
        snapshot = technical_snapshot(history)
        state.freshness.append(history.freshness)
        state.data.setdefault("technicals", {})[snapshot.ticker] = snapshot.model_dump(mode="json")
        state.data.setdefault("price_history", {})[snapshot.ticker] = [
            {"date": bar.date.isoformat(), "close": bar.close} for bar in history.bars
        ]
        state.sources.append(
            _market_source(f"{snapshot.ticker} price history", history.freshness.label())
        )
        return (
            f"{snapshot.ticker} last close ${snapshot.price:,.2f} on {snapshot.as_of}. "
            f"Trend: {snapshot.trend}. " + " ".join(snapshot.notes)
        )

    return StructuredTool.from_function(
        func=get_technical_snapshot,
        name="get_technical_snapshot",
        description="One-year price trend for a ticker: moving averages, RSI, volatility, "
        "52-week range, recent crossovers.",
        args_schema=TickerArgs,
    )


@_register
def make_get_company_overview(context: AgentContext, state: RunState) -> BaseTool:
    def get_company_overview(ticker: str) -> str:
        o = context.market.get_company_overview(ticker)
        state.freshness.append(o.freshness)
        state.data.setdefault("overviews", {})[o.ticker] = o.model_dump(mode="json")
        state.sources.append(_market_source(f"{o.ticker} company overview", o.freshness.label()))
        parts = [
            f"{sanitize_untrusted(o.name, 120)} ({o.ticker})",
            f"type: {o.asset_type or 'n/a'}",
            f"sector: {o.sector or 'n/a'}",
        ]
        if o.market_cap:
            parts.append(f"market cap ${o.market_cap / 1e9:,.1f}B")
        if o.pe_ratio:
            parts.append(f"P/E {o.pe_ratio:.1f}")
        if o.dividend_yield is not None:
            parts.append(f"dividend yield {o.dividend_yield:.2%}")
        if o.beta is not None:
            parts.append(f"beta {o.beta:.2f}")
        return ", ".join(parts) + "."

    return StructuredTool.from_function(
        func=get_company_overview,
        name="get_company_overview",
        description="Basic facts for a stock or fund: type, sector, market cap, P/E, "
        "dividend yield, beta.",
        args_schema=TickerArgs,
    )


# ---- portfolio -------------------------------------------------------------------------


class HoldingArg(BaseModel):
    ticker: str
    shares: float = Field(gt=0)
    cost_basis: float | None = Field(default=None, ge=0, description="Total cost, if known")


class PortfolioArgs(BaseModel):
    holdings: list[HoldingArg] | None = Field(
        default=None,
        description="Holdings the user described in this message. Omit to use the user's "
        "saved portfolio.",
    )


@_register
def make_analyze_portfolio(context: AgentContext, state: RunState) -> BaseTool:
    def analyze_portfolio(holdings: list[HoldingArg] | None = None) -> str:
        request = state.request
        rows = [Holding(**h.model_dump()) for h in holdings] if holdings else request.portfolio
        if not rows:
            return "No portfolio was provided. Ask the user to list their holdings."
        profile = context.risk_profiles[request.profile.risk_tolerance]
        analysis = fetch_and_analyze(
            rows,
            context.market,
            profile=profile,
            catalog=context.catalog,
            config=context.settings.analytics,
        )
        state.data["holdings"] = [h.model_dump(mode="json") for h in rows]
        state.data["portfolio_analysis"] = analysis.model_dump(mode="json")
        state.freshness.extend(analysis.freshness)
        state.sources.append(
            _market_source("Portfolio prices", f"{len(analysis.holdings)} holdings")
        )

        lines = [
            f"Total value ${analysis.total_value:,.2f}.",
            "Asset mix: " + ", ".join(f"{k} {v:.0%}" for k, v in analysis.asset_allocation.items()),
            "Holdings: " + ", ".join(f"{h.ticker} {h.weight:.0%}" for h in analysis.holdings),
            f"Diversification score {analysis.diversification_score:.0f}/100; "
            f"risk {analysis.risk_score}/10 ({analysis.risk_level}).",
            expense_ratio_label(analysis),
        ]
        m = analysis.risk_metrics
        if m:
            sharpe = "n/a" if m.sharpe_ratio is None else f"{m.sharpe_ratio:.2f}"
            beta = "n/a" if m.beta is None else f"{m.beta:.2f}"
            lines.append(
                f"Past year: return {_pct(m.annual_return)}, volatility "
                f"{_pct(m.annual_volatility)}, max drawdown {_pct(m.max_drawdown)}, beta {beta}, "
                f"Sharpe {sharpe} using risk-free rate {m.risk_free.label()}."
            )
        if analysis.unrealized_gain is not None:
            lines.append(f"Unrealized gain ${analysis.unrealized_gain:,.2f}.")
        lines += [f"Observation: {o}" for o in analysis.observations]
        return "\n".join(lines)

    return StructuredTool.from_function(
        func=analyze_portfolio,
        name="analyze_portfolio",
        description="Analyze holdings: value, allocation, diversification, fees, risk, and "
        "past-year metrics, with educational observations.",
        args_schema=PortfolioArgs,
    )


# ---- goals -----------------------------------------------------------------------------


class GoalArgs(BaseModel):
    target_amount: float = Field(gt=0, description="Goal amount in today's dollars")
    years: int = Field(ge=1, le=60)
    current_balance: float = Field(default=0, ge=0)
    monthly_contribution: float = Field(default=0, ge=0)
    risk_tolerance: Literal["conservative", "moderate", "aggressive"] | None = Field(
        default=None, description="Defaults to the user's profile"
    )
    inflation_adjusted: bool = True


@_register
def make_project_goal(context: AgentContext, state: RunState) -> BaseTool:
    def project_goal(
        target_amount: float,
        years: int,
        current_balance: float = 0,
        monthly_contribution: float = 0,
        risk_tolerance: str | None = None,
        inflation_adjusted: bool = True,
    ) -> str:
        tolerance = risk_tolerance or state.request.profile.risk_tolerance
        profile = context.risk_profiles[tolerance]  # type: ignore[index]
        mc = context.settings.analytics.monte_carlo
        inputs = inputs_for_profile(
            profile,
            mc,
            current_balance=current_balance,
            monthly_contribution=monthly_contribution,
            years=years,
            target_amount=target_amount,
            target_in_todays_dollars=inflation_adjusted,
            seed=42,
        )
        result = simulate(inputs)
        needed = required_monthly_contribution(inputs, mc.target_success_probability)
        state.data["goal_projection"] = result.model_dump(mode="json") | {
            "required_monthly_contribution": needed,
            "target_success_probability": mc.target_success_probability,
            "risk_profile": profile.model_dump(mode="json"),
        }
        p = result.final_percentiles
        return (
            f"Projection ({result.dollars} dollars, {profile.label.lower()} assumptions: "
            f"{profile.expected_return:.1%} expected return, {profile.volatility:.0%} "
            f"volatility, {mc.simulations:,} simulated paths): probability of reaching "
            f"${target_amount:,.0f} in {years} years is "
            f"{chance_text(result.success_probability)}. "
            f"Ending balance range: P10 ${p[10]:,.0f}, median ${p[50]:,.0f}, P90 ${p[90]:,.0f}. "
            f"With a steady return and no market swings: "
            f"${result.deterministic_final:,.0f}. "
            f"Monthly contribution for a {mc.target_success_probability:.0%} probability: "
            f"${needed:,.2f}. These are hypothetical, not forecasts."
            + (f" {low_odds_note()}" if result.success_probability < LOW_ODDS else "")
        )

    return StructuredTool.from_function(
        func=project_goal,
        name="project_goal",
        description="Monte Carlo projection for saving toward a target amount, plus the "
        "monthly contribution needed for an 80% probability of success.",
        args_schema=GoalArgs,
    )


# ---- news ------------------------------------------------------------------------------


class NewsArgs(BaseModel):
    ticker: str | None = Field(default=None, description="A ticker, for company news")
    query: str | None = Field(default=None, description="A topic, for general market news")
    limit: int = Field(default=5, ge=1, le=10)


@_register
def make_get_news(context: AgentContext, state: RunState) -> BaseTool:
    def get_news(ticker: str | None = None, query: str | None = None, limit: int = 5) -> str:
        feed = context.market.get_news(ticker=ticker, query=query, limit=limit)
        state.freshness.append(feed.freshness)
        state.data["news"] = feed.model_dump(mode="json")
        if not feed.articles:
            return f"No recent news found for {feed.query}."
        lines = []
        for a in feed.articles:
            state.news.append(
                Source(title=a.title, kind="news", url=a.url, published_at=a.published_at)
            )
            when = a.published_at.date().isoformat() if a.published_at else "date unknown"
            title = sanitize_untrusted(a.title, 200)
            source = sanitize_untrusted(a.source or "unknown source", 80)
            summary = sanitize_untrusted(a.summary or "", 300)
            lines.append(f"[N{len(state.news)}] {title} ({source}, {when}). {summary}")
        return (
            "Cite each article you mention as [N1], [N2], ... matching the markers below.\n"
            + wrap_untrusted("news_articles", "\n".join(lines))
        )

    return StructuredTool.from_function(
        func=get_news,
        name="get_news",
        description="Recent news for a ticker or a topic, with sources and dates.",
        args_schema=NewsArgs,
    )


# ---- tax -------------------------------------------------------------------------------


class TaxFiguresArgs(BaseModel):
    keys: list[str] | None = Field(default=None, description="Figure keys; omit to list all")


@_register
def make_get_tax_figures(context: AgentContext, state: RunState) -> BaseTool:
    def get_tax_figures(keys: list[str] | None = None) -> str:
        ref = context.tax
        figures = [ref.figures[k] for k in keys or ref.figures if k in ref.figures]
        missing = [k for k in keys or [] if k not in ref.figures]
        lines = [f"Tax year {ref.tax_year} ({ref.jurisdiction}):"]
        for f in figures:
            status = "verified" if f.verified else "UNCONFIRMED"
            lines.append(f"- {f.key}: {f.label} = {f.value} [{status}; {f.source_url}]")
            state.sources.append(
                Source(title=f"IRS: {f.label}", kind="knowledge_base", url=f.source_url)
            )
        if missing:
            lines.append(f"Unknown keys: {', '.join(missing)}. Available: {', '.join(ref.figures)}")
        return "\n".join(lines)

    return StructuredTool.from_function(
        func=get_tax_figures,
        name="get_tax_figures",
        description="Official tax-year figures (contribution limits, thresholds, rates) with "
        "their IRS source.",
        args_schema=TaxFiguresArgs,
    )


class AccountsArgs(BaseModel):
    account_types: list[str] | None = Field(
        default=None,
        description="Any of: traditional_401k, roth_401k, traditional_ira, roth_ira, hsa, "
        "plan_529, taxable_brokerage. Omit for all.",
    )


@_register
def make_compare_tax_accounts(context: AgentContext, state: RunState) -> BaseTool:
    def compare_tax_accounts(account_types: list[str] | None = None) -> str:
        rows = compare_accounts(context.tax, account_types)
        state.data["account_comparison"] = [r.model_dump(mode="json") for r in rows]
        lines = []
        for r in rows:
            limits = "; ".join(f"{f.label}: {f.value}" for f in r.limits)
            lines.append(
                f"{r.name}: contributions {r.contributions}; growth {r.growth}; withdrawals "
                f"{r.withdrawals}; early withdrawal {r.early_withdrawal}; RMDs "
                f"{r.required_distributions}; limits: {limits}."
            )
        return "\n".join(lines)

    return StructuredTool.from_function(
        func=compare_tax_accounts,
        name="compare_tax_accounts",
        description="Side-by-side tax treatment and limits for retirement, health, education, "
        "and taxable accounts.",
        args_schema=AccountsArgs,
    )


class GainsArgs(BaseModel):
    gain: float = Field(gt=0)
    purchase_date: date
    sale_date: date
    taxable_income: float = Field(ge=0, description="Taxable income before the gain")
    filing_status: Literal["single", "married_joint"] = "single"


@_register
def make_illustrate_capital_gains(context: AgentContext, state: RunState) -> BaseTool:
    def illustrate(
        gain: float,
        purchase_date: date,
        sale_date: date,
        taxable_income: float,
        filing_status: str = "single",
    ) -> str:
        result = illustrate_capital_gains(
            context.tax,
            gain=gain,
            purchase_date=purchase_date,
            sale_date=sale_date,
            taxable_income=taxable_income,
            filing_status=filing_status,  # type: ignore[arg-type]
        )
        state.data["capital_gains"] = result.model_dump(mode="json")
        term = "long-term" if result.long_term else "short-term"
        return (
            f"This sale is {term} (long-term starts {result.first_long_term_sale_date}). "
            f"Illustrative federal tax: ${result.tax_if_short_term:,.2f} if short-term vs "
            f"${result.tax_if_long_term:,.2f} if long-term (difference "
            f"${result.difference:,.2f}). Caveats: " + " ".join(result.caveats)
        )

    return StructuredTool.from_function(
        func=illustrate,
        name="illustrate_capital_gains",
        description="Simplified federal tax illustration comparing short- vs long-term "
        "treatment of a gain.",
        args_schema=GainsArgs,
    )


# ---- coordination ----------------------------------------------------------------------


class HandoffArgs(BaseModel):
    agent: Literal["finance_qa", "portfolio", "market", "goal_planning", "news", "tax"]
    reason: str = Field(description="Why that specialist should also weigh in")


@_register
def make_request_handoff(context: AgentContext, state: RunState) -> BaseTool:
    def request_handoff(agent: str, reason: str) -> str:
        if not state.request.allow_handoff:
            return "Not handed off: hand-offs are disabled while answering a hand-off."
        if agent == state.agent or agent not in AGENT_NAMES:
            return "Not handed off: choose a different specialist."
        if state.handoffs and agent not in state.handoffs:
            return (
                "Not handed off: only one hand-off is allowed per question, and one was "
                "already requested. Finish your answer."
            )
        if agent not in state.handoffs:
            state.handoffs.append(agent)
        state.data.setdefault("handoff_reasons", {})[agent] = reason
        return f"OK: the {agent} specialist will also respond after you. Finish your part."

    return StructuredTool.from_function(
        func=request_handoff,
        name="request_handoff",
        description="Ask another Finnie specialist to add their part of the answer, e.g. "
        "the tax specialist for tax consequences.",
        args_schema=HandoffArgs,
    )
