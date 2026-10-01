"""Each specialist end to end with a scripted model: the right tools, data, and sources."""

import pytest

from src.agents import context as context_module
from src.agents.base import AgentRequest, load_prompt
from src.agents.registry import AGENT_CLASSES, build_agents
from src.agents.tools import available_tools
from src.core.models import AGENT_NAMES, Holding
from src.rag.chunking import GLOSSARY_CATEGORY
from src.rag.knowledge_base import CATEGORIES
from tests.unit.agents.conftest import tool_call


def test_registry_has_all_six_with_valid_config(make_context):
    agents = build_agents(make_context())
    assert set(agents) == set(AGENT_NAMES)
    tools = set(available_tools())
    for name, cls in AGENT_CLASSES.items():
        assert cls.name == name and cls.description
        assert set(cls.tool_names) <= tools and "request_handoff" in cls.tool_names
        assert load_prompt(name).startswith("Your role:")
        if cls.rag_categories is not None:
            assert set(cls.rag_categories) <= set(CATEGORIES) | {GLOSSARY_CATEGORY}


def test_portfolio_agent(make_context):
    context = make_context(
        tool_call(
            "analyze_portfolio",
            {"holdings": [{"ticker": "VTI", "shares": 10}, {"ticker": "AAPL", "shares": 5}]},
        ),
        "Your portfolio is worth $4,150. Diversification spreads risk [1].",
    )
    result = build_agents(context)["portfolio"].run(
        AgentRequest(query="How is my diversification across investments? 10 VTI, 5 AAPL")
    )
    assert result.ok and result.data["portfolio_analysis"]["total_value"] == 4150
    assert [h["ticker"] for h in result.data["holdings"]] == ["VTI", "AAPL"]
    kinds = {s.kind for s in result.sources}
    assert kinds == {"knowledge_base", "market_data"} and result.freshness


def test_market_agent(make_context):
    context = make_context(
        tool_call("get_market_overview"),
        tool_call("get_technical_snapshot", {"ticker": "SPY"}, call_id="c2"),
        "Markets are mixed today; SPY is in an uptrend.",
    )
    result = build_agents(context)["market"].run(AgentRequest(query="How is the market today?"))
    assert result.data["meta"]["tool_calls"] == ["get_market_overview", "get_technical_snapshot"]
    assert "market_overview" in result.data and "SPY" in result.data["price_history"]


def test_goal_agent_uses_prior_portfolio_result(make_context):
    from src.core.models import AgentResult

    context = make_context(
        tool_call(
            "project_goal",
            {
                "target_amount": 500_000,
                "years": 20,
                "current_balance": 4150,
                "monthly_contribution": 800,
            },
        ),
        "There's roughly a 60% chance under these assumptions.",
    )
    request = AgentRequest(
        query="Will I reach $500k in 20 years?",
        prior_results={"portfolio": AgentResult(agent="portfolio", answer="Total value $4,150.")},
    )
    result = build_agents(context)["goal_planning"].run(request)
    assert "portfolio: Total value $4,150." in context.llm.calls[0][0].content
    assert result.data["goal_projection"]["inputs"]["current_balance"] == 4150


def test_news_agent(make_context):
    context = make_context(tool_call("get_news", {"ticker": "AAPL"}), "Stocks rose [N1].")
    result = build_agents(context)["news"].run(AgentRequest(query="Apple news"))
    assert [s.kind for s in result.sources] == ["news"] and result.data["news"]["articles"]


def test_tax_agent_with_handoff(make_context):
    context = make_context(
        tool_call(
            "illustrate_capital_gains",
            {
                "gain": 5000,
                "purchase_date": "2024-01-02",
                "sale_date": "2026-01-05",
                "taxable_income": 60000,
            },
        ),
        tool_call(
            "request_handoff", {"agent": "portfolio", "reason": "check holdings"}, call_id="c2"
        ),
        "Long-term gains are usually taxed at lower rates [1].",
    )
    result = build_agents(context)["tax"].run(
        AgentRequest(
            query="How is a gain taxed after holding two years?",
            portfolio=[Holding(ticker="VTI", shares=1)],
        )
    )
    assert result.data["capital_gains"]["long_term"] is True
    assert result.handoff == ["portfolio"]
    assert any(s.article_id == "taxes-002" for s in result.sources)


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_every_agent_survives_a_broken_model(make_context, name):
    result = build_agents(make_context(RuntimeError("boom")))[name].run(AgentRequest(query="q"))
    assert result.agent == name and result.error == "RuntimeError: boom"


def test_build_agent_context_wires_production_dependencies(monkeypatch):
    from src.core import llm as llm_module
    from src.data import service as service_module
    from src.rag import retriever as retriever_module

    monkeypatch.setattr(llm_module, "get_llm", lambda tier, settings=None: f"llm-{tier}")
    monkeypatch.setattr(service_module, "get_market_data_service", lambda: "market")
    monkeypatch.setattr(retriever_module, "get_retriever", lambda: "retriever")
    context = context_module.build_agent_context()
    assert (context.llm, context.fast_llm, context.market, context.retriever) == (
        "llm-main",
        "llm-fast",
        "market",
        "retriever",
    )
    assert context.tax.tax_year == 2026 and "VTI" in context.catalog

    def broken():
        raise OSError("no index")

    monkeypatch.setattr(retriever_module, "get_retriever", broken)
    assert context_module.build_agent_context().retriever is None
