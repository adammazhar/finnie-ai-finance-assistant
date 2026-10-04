"""Finnie's MCP tools, resources, and prompt, through an in-memory MCP client."""

from __future__ import annotations

import logging
from typing import Any

from mcp import Client, MCPError

from src.core.guardrails import SHORT_DISCLAIMER
from src.mcp_server import server as server_module
from src.mcp_server.server import Services, build_server, default_services
from src.rag.knowledge_base import load_articles
from tests.fakes.market_service import FakeMarketService
from tests.unit.mcp_server.support import with_client

TOOLS = {
    "get_stock_quote",
    "get_market_overview",
    "analyze_portfolio",
    "project_financial_goal",
    "search_financial_knowledge",
    "explain_tax_account",
}


def call(services: Services, tool: str, args: dict[str, Any] | None = None) -> Any:
    async def use(client: Client) -> Any:
        return await client.call_tool(tool, args or {})

    return with_client(services, use)


def ok(services: Services, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    result = call(services, tool, args)
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return result.structured_content


def error_text(services: Services, tool: str, args: dict[str, Any] | None = None) -> str:
    result = call(services, tool, args)
    assert result.is_error
    return result.content[0].text


def test_lists_tools_with_titles_and_output_schemas(services):
    async def use(client: Client) -> Any:
        return await client.list_tools()

    tools = with_client(services, use).tools
    assert {t.name for t in tools} == TOOLS
    for tool in tools:
        assert tool.title and tool.description
        assert tool.output_schema is not None
    search = next(t for t in tools if t.name == "search_financial_knowledge")
    assert "Taxes" not in search.description and "taxes" in search.description
    tax = next(t for t in tools if t.name == "explain_tax_account")
    assert "roth_ira" in tax.input_schema["properties"]["account_type"]["enum"]


def test_server_identity_and_instructions(services):
    async def use(client: Client) -> Any:
        return client.server_info, client.instructions

    info, instructions = with_client(services, use)
    assert info.name == "Finnie"
    assert "not financial, investment, tax, or legal advice" in instructions


def test_quote_has_price_time_and_disclaimer(services):
    quote = ok(services, "get_stock_quote", {"ticker": "vti"})
    assert quote["ticker"] == "VTI"
    assert quote["price"] == 300.0
    assert "Sep 30, 2026" in quote["price_time"]
    assert quote["data_freshness"]
    assert quote["disclaimer"] == SHORT_DISCLAIMER


def test_unknown_ticker_is_a_tool_error(services):
    assert "No quote for 'ZZZZ'" in error_text(services, "get_stock_quote", {"ticker": "ZZZZ"})


def test_failed_tool_arguments_stay_out_of_logs(services, caplog):
    caplog.set_level(logging.DEBUG)
    error_text(services, "get_stock_quote", {"ticker": "SECRETTICKER"})
    assert "SECRETTICKER" not in caplog.text


def test_market_overview_rows(services):
    overview = ok(services, "get_market_overview")
    spy = next(row for row in overview["indices"] if row["tracking_etf"] == "SPY")
    assert spy["index_symbol"] == "^GSPC" and spy["index_level"] == 6745.12
    assert spy["etf_price"] == 600.0
    assert len(overview["sectors"]) == 11
    assert overview["summary"] and overview["price_time"]


def test_market_overview_without_index_levels(make_services):
    class NoLevels(FakeMarketService):
        def get_quotes(self, tickers):
            return super().get_quotes([t for t in tickers if not t.startswith("^")])

    overview = ok(make_services(NoLevels()), "get_market_overview")
    assert all(row["index_level"] is None for row in overview["indices"])


def test_market_overview_with_nothing_priced(make_services):
    class Empty(FakeMarketService):
        def get_quotes(self, tickers):
            return super().get_quotes([])

    overview = ok(make_services(Empty()), "get_market_overview")
    assert overview["indices"] == [] and overview["price_time"] is None


def test_market_outage_is_a_tool_error(make_services):
    market = FakeMarketService(fail={"get_quotes"})
    assert "unavailable" in error_text(make_services(market), "get_market_overview")


def test_analyze_portfolio(services):
    result = ok(
        services,
        "analyze_portfolio",
        {"holdings": [{"ticker": "VTI", "shares": 10}, {"ticker": "BND", "shares": 20}]},
    )
    assert result["total_value"] == 4400.0
    assert {h["ticker"] for h in result["holdings"]} == {"VTI", "BND"}
    assert result["expense_ratio"].endswith("%") or "%" in result["expense_ratio"]
    assert result["past_year"]["volatility"] is not None
    assert 0 <= result["diversification_score"] <= 100


def test_stock_only_portfolio_has_no_expense_ratio(services):
    result = ok(services, "analyze_portfolio", {"holdings": [{"ticker": "AAPL", "shares": 1}]})
    assert result["expense_ratio"] == "No funds, so no expense ratio"


def test_portfolio_without_history_has_no_past_year(make_services):
    market = FakeMarketService(fail={"get_daily_history"})
    result = ok(
        make_services(market), "analyze_portfolio", {"holdings": [{"ticker": "VTI", "shares": 1}]}
    )
    assert result["past_year"] is None


def test_empty_portfolio_is_a_tool_error(services):
    assert "no holdings" in error_text(services, "analyze_portfolio", {"holdings": []})


def test_goal_projection(services):
    goal = ok(
        services,
        "project_financial_goal",
        {"target_amount": 50000, "years": 10, "current_savings": 5000, "monthly_contribution": 300},
    )
    assert 0 <= goal["success_probability"] <= 1
    assert set(goal["ending_balance"]) == {"p10", "median", "p90"}
    assert goal["monthly_contribution_for_80_percent"] > 300
    assert goal["assumptions"]["risk_tolerance"] == "moderate"


def test_unlikely_goal_gets_the_low_odds_note(services):
    goal = ok(services, "project_financial_goal", {"target_amount": 10_000_000, "years": 5})
    assert goal["success_probability"] < 0.25 and goal["note"]


def test_easy_goal_has_no_note(services):
    goal = ok(
        services,
        "project_financial_goal",
        {"target_amount": 1000, "years": 5, "current_savings": 5000},
    )
    assert goal["note"] is None


def test_invalid_goal_is_a_tool_error(services):
    text = error_text(services, "project_financial_goal", {"target_amount": 1000, "years": 0})
    assert "Invalid goal" in text


def test_search_returns_passages_with_sources(services):
    found = ok(services, "search_financial_knowledge", {"query": "what is an ETF", "limit": 50})
    first = found["results"][0]
    assert first["title"] == "What Is an ETF?"
    assert first["resource_uri"] == "finnie://articles/funds_etfs-001"
    assert first["sources"][0].startswith("What Is an ETF? source (investor.gov)")
    glossary = [p for p in found["results"] if p["category"] == "Glossary"]
    assert all(p["resource_uri"] is None for p in glossary)


def test_search_by_category(services):
    found = ok(
        services,
        "search_financial_knowledge",
        {"query": "capital gains", "category": "taxes", "limit": 1},
    )
    assert [p["title"] for p in found["results"]] == ["Capital Gains Basics"]


def test_search_errors(make_services, services):
    assert "isn't available" in error_text(
        make_services(search=False), "search_financial_knowledge", {"query": "ETF"}
    )
    assert "Unknown category 'crypto'" in error_text(
        services, "search_financial_knowledge", {"query": "ETF", "category": "crypto"}
    )
    assert "empty" in error_text(services, "search_financial_knowledge", {"query": "  "})


def test_explain_tax_account(services):
    roth = ok(services, "explain_tax_account", {"account_type": "roth_ira"})
    assert roth["name"] == "Roth IRA"
    assert roth["limits"] and roth["limits"][0]["source"].startswith("https://www.irs.gov/")
    assert roth["tax_year"] >= 2026


def test_tax_account_type_is_validated(services):
    assert call(services, "explain_tax_account", {"account_type": "crypto_ira"}).is_error


def test_article_and_glossary_resources(services):
    article = load_articles()[0]

    async def use(client: Client) -> Any:
        templates = await client.list_resource_templates()
        resources = await client.list_resources()
        text = await client.read_resource(f"finnie://articles/{article.meta.id}")
        glossary = await client.read_resource("finnie://glossary")
        return templates, resources, text, glossary

    templates, resources, text, glossary = with_client(services, use)
    assert templates.resource_templates[0].uri_template == "finnie://articles/{article_id}"
    assert [str(r.uri) for r in resources.resources] == ["finnie://glossary"]
    body = text.contents[0].text
    assert body.startswith(f"# {article.meta.title}") and "## Sources" in body
    assert glossary.contents[0].text.startswith("# Finnie glossary\n\n- **")


def test_unknown_article_is_an_error(services):
    async def use(client: Client) -> Any:
        try:
            await client.read_resource("finnie://articles/nope")
        except MCPError as exc:
            return exc
        return None

    error = with_client(services, use)
    assert isinstance(error, MCPError) and "No article 'nope'" in str(error)


def test_beginner_prompt(services):
    async def use(client: Client) -> Any:
        prompts = await client.list_prompts()
        prompt = await client.get_prompt("explain_like_beginner", {"topic": "index funds"})
        return prompts, prompt

    prompts, prompt = with_client(services, use)
    assert [p.name for p in prompts.prompts] == ["explain_like_beginner"]
    text = prompt.messages[0].content.text
    assert text.startswith("Explain index funds") and "search_financial_knowledge" in text


def test_default_services(monkeypatch):
    import src.data.service as data_service
    import src.rag.retriever as rag_retriever

    market = FakeMarketService()
    monkeypatch.setattr(data_service, "get_market_data_service", lambda: market)
    monkeypatch.setattr(rag_retriever, "get_retriever", lambda: "index")
    services = default_services(warm=False)
    assert services.market is market
    assert services.retriever() == "index"

    def broken():
        raise RuntimeError("no index on disk")

    monkeypatch.setattr(rag_retriever, "get_retriever", broken)
    assert default_services(warm=False).retriever() is None


def test_default_services_warm_the_knowledge_base_in_the_background(monkeypatch):
    import threading

    import src.data.service as data_service
    import src.rag.retriever as rag_retriever

    loaded = threading.Event()
    threads: list[str] = []

    def load():
        threads.append(threading.current_thread().name)
        loaded.set()
        return "index"

    monkeypatch.setattr(data_service, "get_market_data_service", FakeMarketService)
    monkeypatch.setattr(rag_retriever, "get_retriever", load)
    default_services()
    assert loaded.wait(timeout=10)
    assert threads == ["finnie-kb-warmup"]


def test_build_server_defaults_to_real_services(monkeypatch, services):
    monkeypatch.setattr(server_module, "default_services", lambda: services)
    assert build_server().name == "Finnie"
