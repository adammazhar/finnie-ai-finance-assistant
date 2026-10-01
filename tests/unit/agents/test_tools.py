from datetime import date

import pytest

from src.agents.base import AgentRequest, RunState
from src.agents.tools import available_tools, build_tools
from src.core.models import Holding, UserProfile
from tests.fakes.market_service import FakeMarketService


def run_tool(context, name, args=None, state=None, agent="finance_qa", **request):
    state = state or RunState(request=AgentRequest(query="q", **request), agent=agent)
    tool = build_tools((name,), context, state)[name]
    return tool.invoke(args or {}), state


def test_registry():
    assert set(available_tools()) == {
        "search_knowledge_base",
        "lookup_glossary_term",
        "get_quotes",
        "get_market_overview",
        "get_technical_snapshot",
        "get_company_overview",
        "analyze_portfolio",
        "project_goal",
        "get_news",
        "get_tax_figures",
        "compare_tax_accounts",
        "illustrate_capital_gains",
        "request_handoff",
    }
    with pytest.raises(KeyError, match="teleport"):
        build_tools(("teleport",), None, None)


def test_search_knowledge_base(make_context):
    context = make_context()
    out, state = run_tool(context, "search_knowledge_base", {"query": "bear market decline"})
    assert out.startswith("New citable passages") and "[1] Bull and Bear Markets" in out
    again, _ = run_tool(
        context, "search_knowledge_base", {"query": "bear market decline"}, state=state
    )
    assert "No new knowledge base passages" in again
    filtered, _ = run_tool(
        context, "search_knowledge_base", {"query": "diversification", "category": "taxes"}
    )
    assert "Capital Gains" in filtered or "Diversification" in filtered  # widened if needed
    bogus, _ = run_tool(
        context, "search_knowledge_base", {"query": "diversification", "category": "astrology"}
    )
    assert "Diversification" in bogus  # an unknown category is ignored, not an error
    none, _ = run_tool(
        make_context(retriever_override=None), "search_knowledge_base", {"query": "x"}
    )
    assert "unavailable" in none


def test_lookup_glossary_term(make_context):
    out, state = run_tool(make_context(), "lookup_glossary_term", {"term": "expense ratio"})
    assert (
        "[1] Expense ratio > Definition" in out and state.blocks[0].chunk.chunk.kind == "glossary"
    )
    missing, _ = run_tool(make_context(), "lookup_glossary_term", {"term": "zzzz qqqq"})
    assert "No glossary entry" in missing
    none, _ = run_tool(make_context(retriever_override=None), "lookup_glossary_term", {"term": "x"})
    assert "unavailable" in none


def test_get_quotes(make_context):
    out, state = run_tool(make_context(), "get_quotes", {"tickers": ["aapl", "ZZZZ"]})
    assert "AAPL: $230.00 (+1.01%), Live" in out and "ZZZZ: unavailable (not found)" in out
    assert state.data["quotes"]["AAPL"]["price"] == 230 and len(state.freshness) == 1
    assert state.sources[0].kind == "market_data" and "AAPL" in state.sources[0].title
    empty, state2 = run_tool(make_context(), "get_quotes", {"tickers": ["ZZZZ"]})
    assert "ZZZZ: unavailable" in empty and state2.sources == []


def test_get_quotes_with_no_results_at_all(make_context):
    class Empty(FakeMarketService):
        def get_quotes(self, tickers):
            from src.data.models import BatchQuotes

            return BatchQuotes(quotes={})

    out, _ = run_tool(make_context(market=Empty()), "get_quotes", {"tickers": ["AAPL"]})
    assert out == "No quotes were available."


def test_get_market_overview(make_context):
    out, state = run_tool(make_context(), "get_market_overview")
    assert "Indices:" in out and "S&P 500 (SPY)" in out and "Mood:" in out
    assert len(state.data["market_overview"]["sectors"]) == 11
    assert len(state.freshness) == 15 and state.sources[0].title.startswith("Market overview")


def test_get_market_overview_partial(make_context, market):
    market.fail = {"get_daily_history"}
    out, _ = run_tool(make_context(), "get_market_overview")
    assert "Unavailable: SPY history" in out


def test_get_market_overview_with_nothing_available(make_context):
    class Down(FakeMarketService):
        def get_quotes(self, tickers):
            from src.data.models import BatchQuotes

            return BatchQuotes(quotes={}, errors={t: "down" for t in tickers})

        def get_daily_history(self, ticker, days=252):
            from src.data.errors import DataUnavailableError

            raise DataUnavailableError("down")

    out, state = run_tool(make_context(market=Down()), "get_market_overview")
    assert "Mood: unknown" in out and state.sources == []


def test_get_technical_snapshot(make_context):
    out, state = run_tool(make_context(), "get_technical_snapshot", {"ticker": "VTI"})
    assert out.startswith("VTI last close $") and "Trend:" in out
    assert len(state.data["price_history"]["VTI"]) == 252
    assert state.data["technicals"]["VTI"]["sma_50"] is not None


def test_get_company_overview(make_context):
    out, state = run_tool(make_context(), "get_company_overview", {"ticker": "AAPL"})
    assert out == (
        "AAPL Inc. (AAPL), type: Equity, sector: Technology, market cap $3,400.0B, "
        "P/E 33.2, dividend yield 0.44%, beta 1.20."
    )
    assert "AAPL" in state.data["overviews"]


def test_company_overview_with_sparse_data(make_context):
    class Sparse(FakeMarketService):
        def get_company_overview(self, ticker):
            from src.data.models import CompanyOverview
            from tests.fakes.market_service import fresh

            return CompanyOverview(ticker="X", name="X Fund", freshness=fresh())

    out, _ = run_tool(make_context(market=Sparse()), "get_company_overview", {"ticker": "X"})
    assert out == "X Fund (X), type: n/a, sector: n/a."


def test_analyze_portfolio_from_message_and_saved(make_context):
    holdings = [
        {"ticker": "VTI", "shares": 10, "cost_basis": 2500},
        {"ticker": "BND", "shares": 20},
    ]
    out, state = run_tool(
        make_context(), "analyze_portfolio", {"holdings": holdings}, agent="portfolio"
    )
    assert "Total value $4,400.00." in out and "Diversification score" in out
    assert "Sharpe" in out and "13-week U.S. Treasury bill yield" in out
    assert "Unrealized gain $500.00." in out and "Observation:" in out
    assert [h["ticker"] for h in state.data["holdings"]] == ["VTI", "BND"]
    assert state.data["portfolio_analysis"]["total_value"] == 4400

    saved, _ = run_tool(
        make_context(),
        "analyze_portfolio",
        agent="portfolio",
        portfolio=[Holding(ticker="AAPL", shares=1)],
    )
    assert "Total value $230.00." in saved
    none, _ = run_tool(make_context(), "analyze_portfolio", agent="portfolio")
    assert "No portfolio was provided" in none


def test_analyze_portfolio_without_history(make_context, market):
    market.fail = {"get_daily_history"}
    out, _ = run_tool(
        make_context(),
        "analyze_portfolio",
        {"holdings": [{"ticker": "VTI", "shares": 1}]},
        agent="portfolio",
    )
    assert "Past year" not in out and "Not enough price history" in out


def test_analyze_portfolio_metrics_without_sharpe_or_beta(make_context, monkeypatch):
    from src.agents import tools as tools_module
    from src.core.portfolio import fetch_and_analyze

    def no_beta(*args, **kwargs):
        analysis = fetch_and_analyze(*args, **kwargs)
        analysis.risk_metrics = analysis.risk_metrics.model_copy(
            update={"beta": None, "sharpe_ratio": None}
        )
        return analysis

    monkeypatch.setattr(tools_module, "fetch_and_analyze", no_beta)
    out, _ = run_tool(
        make_context(),
        "analyze_portfolio",
        {"holdings": [{"ticker": "VTI", "shares": 1}]},
        agent="portfolio",
    )
    assert "beta n/a, Sharpe n/a" in out


def test_project_goal(make_context):
    args = {
        "target_amount": 100_000,
        "years": 10,
        "current_balance": 10_000,
        "monthly_contribution": 500,
    }
    out, state = run_tool(
        make_context(),
        "project_goal",
        args,
        agent="goal_planning",
        profile=UserProfile(risk_tolerance="conservative"),
    )
    assert "conservative assumptions: 4.5% expected return, 6% volatility" in out
    assert "probability of reaching $100,000 in 10 years" in out and "hypothetical" in out
    projection = state.data["goal_projection"]
    assert projection["required_monthly_contribution"] > 0
    assert projection["dollars"] == "today's" and len(projection["yearly"]) == 11
    aggressive, _ = run_tool(
        make_context(),
        "project_goal",
        args | {"risk_tolerance": "aggressive", "inflation_adjusted": False},
        agent="goal_planning",
    )
    assert "aggressive assumptions" in aggressive and "(nominal dollars" in aggressive
    assert "The odds are low" not in aggressive


def test_project_goal_low_odds_explains_what_changes_the_outcome(make_context):
    out, _ = run_tool(
        make_context(),
        "project_goal",
        {"target_amount": 5_000_000, "years": 5, "monthly_contribution": 100},
        agent="goal_planning",
    )
    assert "in 5 years is under 1%." in out
    assert "more time to save, a higher monthly contribution, or a smaller target" in out


def test_get_news(make_context):
    out, state = run_tool(make_context(), "get_news", {"ticker": "AAPL"})
    assert out.startswith("Cite each article you mention as [N1]")
    assert "<untrusted_news_articles>" in out and out.endswith("</untrusted_news_articles>")
    assert "[N1] Stocks rise on rate hopes (Example Wire, 2026-09-30)" in out
    # articles become sources only if the answer cites them (see test_safety.py)
    assert state.sources == [] and state.news[0].kind == "news"
    assert state.news[0].url.startswith("https://")
    empty, _ = run_tool(
        make_context(market=FakeMarketService(news=[])), "get_news", {"query": "inflation"}
    )
    assert empty == "No recent news found for inflation."


def test_news_without_date_or_source(make_context):
    from src.data.models import NewsArticle

    market = FakeMarketService(news=[NewsArticle(title="Untitled wire story")])
    out, _ = run_tool(make_context(market=market), "get_news", {"query": "x"})
    assert "Untitled wire story (unknown source, date unknown)." in out


def test_tax_tools(make_context):
    out, state = run_tool(
        make_context(), "get_tax_figures", {"keys": ["ira_contribution_limit", "nope"]}, agent="tax"
    )
    assert "ira_contribution_limit" in out and "= 7500" in out and "[verified;" in out
    assert "Unknown keys: nope" in out and state.sources[0].url.startswith("https://www.irs.gov/")
    everything, _ = run_tool(make_context(), "get_tax_figures", agent="tax")
    assert everything.count("\n- ") >= 17

    table, state = run_tool(
        make_context(), "compare_tax_accounts", {"account_types": ["roth_ira", "hsa"]}, agent="tax"
    )
    assert table.startswith("Roth IRA: contributions") and "Health Savings Account" in table
    assert len(state.data["account_comparison"]) == 2

    gains, state = run_tool(
        make_context(),
        "illustrate_capital_gains",
        {
            "gain": 10_000,
            "purchase_date": "2025-03-01",
            "sale_date": "2026-03-01",
            "taxable_income": 40_000,
        },
        agent="tax",
    )
    assert "short-term (long-term starts 2026-03-02)" in gains
    assert "$1,200.00 if short-term vs $82.50 if long-term" in gains
    assert state.data["capital_gains"]["purchase_date"] == date(2025, 3, 1).isoformat()


def test_unverified_tax_figure_is_labelled(make_context):
    context = make_context()
    figure = context.tax.figures["ira_contribution_limit"]
    context.tax = context.tax.model_copy(
        update={
            "figures": context.tax.figures
            | {"ira_contribution_limit": figure.model_copy(update={"status": "VERIFY"})}
        }
    )
    out, _ = run_tool(context, "get_tax_figures", {"keys": ["ira_contribution_limit"]})
    assert "[UNCONFIRMED;" in out


def test_request_handoff(make_context):
    out, state = run_tool(
        make_context(),
        "request_handoff",
        {"agent": "tax", "reason": "capital gains"},
        agent="portfolio",
    )
    assert out.startswith("OK: the tax specialist") and state.handoffs == ["tax"]
    run_tool(make_context(), "request_handoff", {"agent": "tax", "reason": "again"}, state=state)
    assert state.handoffs == ["tax"] and state.data["handoff_reasons"]["tax"] == "again"
    self_handoff, _ = run_tool(
        make_context(), "request_handoff", {"agent": "portfolio", "reason": "x"}, agent="portfolio"
    )
    assert self_handoff.startswith("Not handed off")
