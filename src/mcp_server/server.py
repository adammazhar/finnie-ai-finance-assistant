"""Finnie's MCP server: market data, portfolio analytics, goal projections, and the
knowledge base as MCP tools, resources, and a prompt.

The server never calls Finnie's own LLM. The connecting client (Claude Desktop, Claude
Code, ...) does the reasoning; Finnie supplies data and calculations, which avoids paying
for two models per question. Every tool result carries the education-only disclaimer,
and market data carries its time and freshness. Failures come back as MCP tool errors
with a plain message, never a stack trace.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from pydantic import BaseModel, Field

from src.core.config import Settings, get_settings
from src.core.guardrails import SHORT_DISCLAIMER
from src.core.indicators import MoverSummary, build_market_overview
from src.core.market_hours import price_time_label
from src.core.models import Holding, RiskTolerance
from src.core.monte_carlo import (
    LOW_ODDS,
    chance_text,
    inputs_for_profile,
    low_odds_note,
    required_monthly_contribution,
    simulate,
)
from src.core.portfolio import (
    PortfolioError,
    expense_ratio_label,
    fetch_and_analyze,
    fund_expense_ratio,
)
from src.core.reference import RiskProfile, SecurityCatalog, get_catalog, get_risk_profiles
from src.core.tax import TaxReference, compare_accounts, get_tax_reference
from src.data.errors import MarketDataError
from src.rag.knowledge_base import CATEGORIES, Article, Glossary, load_articles, load_glossary
from src.rag.retriever import Retriever
from src.web_app.formatting import domain, match_label, snippet

logger = logging.getLogger(__name__)

INSTRUCTIONS = (
    "Finnie is a financial education assistant. These tools return market data, portfolio "
    "analytics, goal projections, and passages from a curated knowledge base with sources. "
    "Use them to explain and teach. Don't tell the user to buy, sell, or allocate; Finnie "
    "provides educational information only, not financial, investment, tax, or legal advice."
)
AccountTypeName = Literal[
    "traditional_401k",
    "roth_401k",
    "traditional_ira",
    "roth_ira",
    "hsa",
    "plan_529",
    "taxable_brokerage",
]


@dataclass
class Services:
    """What the tools read from. Tests pass fakes; production builds the real ones."""

    market: Any
    settings: Settings
    retriever: Callable[[], Retriever | None]  # loaded on first search (slow: embedding model)
    catalog: SecurityCatalog = field(default_factory=get_catalog)
    risk_profiles: dict[RiskTolerance, RiskProfile] = field(default_factory=get_risk_profiles)
    tax: TaxReference = field(default_factory=get_tax_reference)
    articles: Callable[[], list[Article]] = load_articles
    glossary: Callable[[], Glossary] = load_glossary


def default_services(*, warm: bool = True) -> Services:
    """The real services. With ``warm``, the knowledge base (embedding model and index,
    tens of seconds on a cold start) loads in the background as soon as the server starts,
    so the first search doesn't run into a client's tool timeout."""
    from src.data.service import get_market_data_service
    from src.rag.retriever import get_retriever

    lock = threading.Lock()  # the warm-up and a first search wait for one load

    def retriever() -> Retriever | None:
        with lock:
            try:
                return get_retriever()
            except Exception:
                logger.exception("Knowledge base unavailable")
                return None

    if warm:
        threading.Thread(target=retriever, name="finnie-kb-warmup", daemon=True).start()
    return Services(market=get_market_data_service(), settings=get_settings(), retriever=retriever)


# ---- result models (become the tools' output schemas) ------------------------------------


class QuoteResult(BaseModel):
    ticker: str
    price: float
    change_percent: float | None
    currency: str
    price_time: str = Field(
        description="When the price is from, live/delayed/last close, and market status"
    )
    data_freshness: str
    disclaimer: str = SHORT_DISCLAIMER


class IndexRow(BaseModel):
    name: str
    index_symbol: str | None
    index_level: float | None
    index_change_percent: float | None
    tracking_etf: str
    etf_price: float
    etf_change_percent: float | None


class SectorRow(BaseModel):
    sector: str
    etf: str
    change_percent: float | None


class MarketOverviewResult(BaseModel):
    indices: list[IndexRow]
    sectors: list[SectorRow] = Field(description="Sector ETFs, strongest first")
    summary: list[str]
    price_time: str | None
    unavailable: list[str]
    disclaimer: str = SHORT_DISCLAIMER


class HoldingInput(BaseModel):
    ticker: str = Field(description="Ticker symbol, e.g. VTI")
    shares: float = Field(gt=0)
    cost_basis: float | None = Field(default=None, ge=0, description="Total cost, optional")


class HoldingRow(BaseModel):
    ticker: str
    name: str
    value: float
    weight: float
    asset_class: str
    expense_ratio: float | None


class PortfolioResult(BaseModel):
    total_value: float
    holdings: list[HoldingRow]
    asset_allocation: dict[str, float]
    sector_allocation: dict[str, float]
    diversification_score: float = Field(description="0 to 100")
    risk_level: str
    risk_score: float = Field(description="1 to 10")
    expense_ratio: str = Field(description="Portfolio expense ratio, funds only")
    past_year: dict[str, float | None] | None
    observations: list[str]
    missing_prices: list[str]
    data_freshness: list[str]
    disclaimer: str = SHORT_DISCLAIMER


class GoalResult(BaseModel):
    chance_of_reaching_goal: str
    success_probability: float
    ending_balance: dict[str, float] = Field(description="P10, median, and P90 outcomes")
    steady_return_balance: float
    monthly_contribution_for_80_percent: float
    dollars: str
    assumptions: dict[str, Any]
    note: str | None
    disclaimer: str = SHORT_DISCLAIMER


class Passage(BaseModel):
    title: str
    section: str
    category: str
    relevance: str
    text: str
    resource_uri: str | None = Field(description="Read the whole article with this resource")
    sources: list[str]


class SearchResult(BaseModel):
    query: str
    results: list[Passage]
    disclaimer: str = SHORT_DISCLAIMER


class TaxAccountResult(BaseModel):
    name: str
    contributions: str
    growth: str
    withdrawals: str
    early_withdrawal: str
    required_distributions: str
    limits: list[dict[str, Any]]
    notes: str
    tax_year: int
    disclaimer: str = SHORT_DISCLAIMER


# ---- server --------------------------------------------------------------------------------


def _index_rows(overview: Any) -> list[IndexRow]:
    rows = []
    for etf in overview.indices:
        level: MoverSummary | None = overview.levels.get(etf.ticker)
        rows.append(
            IndexRow(
                name=etf.name,
                index_symbol=level.ticker if level else None,
                index_level=round(level.price, 2) if level else None,
                index_change_percent=level.change_percent if level else None,
                tracking_etf=etf.ticker,
                etf_price=round(etf.price, 2),
                etf_change_percent=etf.change_percent,
            )
        )
    return rows


def build_server(services: Services | None = None) -> MCPServer:
    finnie = services or default_services()
    # The SDK logs failed tool calls with their arguments at INFO; keep those out of logs.
    server = MCPServer("Finnie", instructions=INSTRUCTIONS, version="1.0.0", log_level="WARNING")
    logging.getLogger("mcp").setLevel(logging.WARNING)

    @server.tool(title="Stock or fund quote")
    def get_stock_quote(ticker: str) -> QuoteResult:
        """Latest price for a stock, ETF, or index (e.g. AAPL, VTI, ^GSPC), with the exact
        time of the price, whether it's live or delayed, and whether the market is open."""
        try:
            quote = finnie.market.get_quote(ticker)
        except (MarketDataError, ValueError) as exc:
            raise ToolError(f"No quote for {ticker!r}: {exc}") from None
        return QuoteResult(
            ticker=quote.ticker,
            price=round(quote.price, 2),
            change_percent=quote.change_percent,
            currency=quote.currency,
            price_time=price_time_label(quote.freshness.as_of),
            data_freshness=quote.freshness.label(),
        )

    @server.tool(title="Market overview")
    def get_market_overview() -> MarketOverviewResult:
        """Major U.S. indexes (S&P 500, Nasdaq-100, Dow, Russell 2000) with the ETFs that
        track them, today's sector moves, and a short summary of the market's mood."""
        try:
            overview = build_market_overview(finnie.market)
        except MarketDataError as exc:
            raise ToolError(f"Market data is unavailable right now: {exc}") from None
        first = overview.indices[0].freshness.as_of if overview.indices else None
        return MarketOverviewResult(
            indices=_index_rows(overview),
            sectors=[
                SectorRow(sector=s.name, etf=s.ticker, change_percent=s.change_percent)
                for s in overview.sectors
            ],
            summary=overview.mood.summary,
            price_time=price_time_label(first) if first else None,
            unavailable=list(overview.errors),
        )

    @server.tool(title="Analyze a portfolio")
    def analyze_portfolio(
        holdings: list[HoldingInput], risk_tolerance: RiskTolerance = "moderate"
    ) -> PortfolioResult:
        """Value, allocation, diversification, risk, fees, and past-year risk measures for a
        list of holdings, compared with a typical mix for the given risk tolerance."""
        try:
            analysis = fetch_and_analyze(
                [Holding(**h.model_dump()) for h in holdings],
                finnie.market,
                profile=finnie.risk_profiles[risk_tolerance],
                catalog=finnie.catalog,
                config=finnie.settings.analytics,
            )
        except (PortfolioError, ValueError) as exc:
            raise ToolError(str(exc).splitlines()[0]) from None
        m = analysis.risk_metrics
        return PortfolioResult(
            total_value=analysis.total_value,
            holdings=[
                HoldingRow(
                    ticker=h.ticker,
                    name=h.name,
                    value=round(h.value, 2),
                    weight=round(h.weight, 4),
                    asset_class=h.asset_class,
                    expense_ratio=h.expense_ratio,
                )
                for h in analysis.holdings
            ],
            asset_allocation=analysis.asset_allocation,
            sector_allocation=analysis.sector_allocation,
            diversification_score=analysis.diversification_score,
            risk_level=analysis.risk_level,
            risk_score=analysis.risk_score,
            expense_ratio=expense_ratio_label(analysis)
            if fund_expense_ratio(analysis.holdings) is not None
            else "No funds, so no expense ratio",
            past_year={
                "return": m.annual_return,
                "volatility": m.annual_volatility,
                "max_drawdown": m.max_drawdown,
                "beta": m.beta,
                "sharpe_ratio": m.sharpe_ratio,
            }
            if m
            else None,
            observations=analysis.observations,
            missing_prices=analysis.missing_prices,
            data_freshness=sorted({f.label() for f in analysis.freshness}),
        )

    @server.tool(title="Project a savings goal")
    def project_financial_goal(
        target_amount: float,
        years: int,
        current_savings: float = 0,
        monthly_contribution: float = 0,
        risk_tolerance: RiskTolerance = "moderate",
        target_in_todays_dollars: bool = True,
    ) -> GoalResult:
        """Monte Carlo projection of saving toward a target: the chance of reaching it, the
        range of outcomes, and the monthly contribution an 80% chance would need. A
        hypothetical illustration under simplified assumptions, not a forecast."""
        profile = finnie.risk_profiles[risk_tolerance]
        mc = finnie.settings.analytics.monte_carlo
        try:
            inputs = inputs_for_profile(
                profile,
                mc,
                current_balance=current_savings,
                monthly_contribution=monthly_contribution,
                years=years,
                target_amount=target_amount,
                target_in_todays_dollars=target_in_todays_dollars,
                seed=42,
            )
        except ValueError as exc:
            raise ToolError(f"Invalid goal: {str(exc).splitlines()[0]}") from None
        result = simulate(inputs)
        p = result.final_percentiles
        return GoalResult(
            chance_of_reaching_goal=chance_text(result.success_probability),
            success_probability=result.success_probability,
            ending_balance={"p10": p[10], "median": p[50], "p90": p[90]},
            steady_return_balance=result.deterministic_final,
            monthly_contribution_for_80_percent=required_monthly_contribution(
                inputs, mc.target_success_probability
            ),
            dollars=result.dollars,
            assumptions={
                "risk_tolerance": risk_tolerance,
                "expected_return": profile.expected_return,
                "volatility": profile.volatility,
                "inflation": inputs.inflation,
                "simulations": inputs.simulations,
            },
            note=low_odds_note() if result.success_probability < LOW_ODDS else None,
        )

    @server.tool(
        title="Search the knowledge base",
        description=(
            "Search Finnie's curated financial education articles and glossary. Returns the "
            "best passages, strongest first, with their sources; read a whole article with "
            "its resource_uri. Optional category (results widen to other categories when it has "
            "too few matches): " + ", ".join(CATEGORIES) + "."
        ),
    )
    def search_financial_knowledge(
        query: str, category: str | None = None, limit: int = 4
    ) -> SearchResult:
        retriever = finnie.retriever()
        if retriever is None:
            raise ToolError("The knowledge base index isn't available on this server.")
        if category is not None and category not in CATEGORIES:
            raise ToolError(f"Unknown category {category!r}. Use one of: {', '.join(CATEGORIES)}")
        try:
            found = retriever.retrieve(
                query, categories=[category] if category else None, k=max(1, min(limit, 8))
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        passages = []
        for item in sorted(found.chunks, key=lambda c: c.score, reverse=True):
            chunk = item.chunk
            article = chunk.kind == "article"
            passages.append(
                Passage(
                    title=chunk.title,
                    section=chunk.section,
                    category=CATEGORIES.get(chunk.category, "Glossary"),
                    relevance=match_label(item.score),
                    text=snippet(chunk.text, 600),
                    resource_uri=f"finnie://articles/{chunk.article_id}" if article else None,
                    sources=[f"{s.name} ({domain(s.url)}): {s.url}" for s in chunk.sources],
                )
            )
        return SearchResult(query=query, results=passages)

    @server.tool(title="Explain a tax-advantaged account")
    def explain_tax_account(account_type: AccountTypeName) -> TaxAccountResult:
        """How an account type works for taxes: contributions, growth, withdrawals, early
        withdrawal rules, required distributions, and this year's limits (from IRS figures)."""
        [row] = compare_accounts(finnie.tax, [account_type])
        return TaxAccountResult(
            name=row.name,
            contributions=row.contributions,
            growth=row.growth,
            withdrawals=row.withdrawals,
            early_withdrawal=row.early_withdrawal,
            required_distributions=row.required_distributions,
            limits=[
                {"label": f.label, "value": f.value, "source": f.source_url} for f in row.limits
            ],
            notes=row.notes,
            tax_year=finnie.tax.tax_year,
        )

    @server.resource(
        "finnie://articles/{article_id}",
        title="Knowledge base article",
        description="A full Finnie article in markdown, with its sources",
        mime_type="text/markdown",
    )
    def read_article(article_id: str) -> str:
        found = next((a for a in finnie.articles() if a.meta.id == article_id), None)
        if found is None:
            raise ResourceNotFoundError(f"No article {article_id!r}")
        sources = "\n".join(f"- [{s.name}]({s.url})" for s in found.meta.sources)
        return (
            f"# {found.meta.title}\n\n{found.body}\n\n## Sources\n{sources}\n\n{SHORT_DISCLAIMER}"
        )

    @server.resource(
        "finnie://glossary",
        title="Glossary",
        description="Every glossary term and definition, A to Z",
        mime_type="text/markdown",
    )
    def read_glossary() -> str:
        glossary = finnie.glossary()
        terms = sorted(glossary.terms, key=lambda t: t.term.lower())
        body = "\n".join(f"- **{t.term}**: {t.definition}" for t in terms)
        sources = "\n".join(f"- [{s.name}]({s.url})" for s in glossary.sources)
        return f"# Finnie glossary\n\n{body}\n\n## Sources\n{sources}"

    @server.prompt(title="Explain like I'm a beginner")
    def explain_like_beginner(topic: str) -> str:
        """Explain a financial topic in plain language, grounded in Finnie's sources."""
        return (
            f"Explain {topic} to someone new to investing. Use search_financial_knowledge "
            "first and cite the sources it returns. Use short paragraphs, define any jargon, "
            "and include one small, clearly hypothetical example. Teach how it works and the "
            "trade-offs; don't tell me what to buy, sell, or do with my money. End with: "
            f"{SHORT_DISCLAIMER}"
        )

    return server
