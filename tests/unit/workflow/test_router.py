import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.workflow.router import (
    ROUTER_PROMPT,
    RouteDecision,
    extract_tickers,
    keyword_route,
    route_with_llm,
    with_portfolio_for_goals,
)
from tests.fakes.llm import FakeChatModel
from tests.unit.workflow.conftest import route


def ask(llm, question="What is an ETF?", history=None, **kwargs):
    defaults = dict(summary=None, has_portfolio=False, max_agents=3, min_confidence=0.5)
    return route_with_llm(llm, question, history=history or [], **(defaults | kwargs))


# ---- keyword fallback ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "question, expected",
    [
        ("How diversified is my portfolio?", "portfolio"),
        ("What's the price of $AAPL?", "market"),
        ("Am I on track to retire by 60?", "goal_planning"),
        ("Any news on Nvidia today?", "news"),
        ("How are capital gains taxed in a Roth IRA?", "tax"),
    ],
)
def test_keyword_route_picks_specialist(question, expected):
    decision = keyword_route(question)
    assert decision.agents[0] == expected
    assert decision.source == "keyword"
    assert decision.standalone_query == question


def test_keyword_route_defaults_to_finance_qa():
    assert keyword_route("What does compound interest mean?").agents == ["finance_qa"]


def test_keyword_route_bare_ticker_goes_to_market():
    decision = keyword_route("Tell me about NVDA")
    assert decision.agents == ["market"]
    assert decision.tickers == ["NVDA"]


def test_keyword_route_caps_agents():
    question = "my portfolio price news tax goal"
    assert len(keyword_route(question, max_agents=2).agents) == 2


def test_keyword_route_adds_portfolio_only_for_listed_holdings():
    assert keyword_route("I hold 10 VTI. On track to retire?").agents[:2] == [
        "portfolio",
        "goal_planning",
    ]
    # a saved portfolio alone doesn't: the user is asked how much of it counts
    assert keyword_route("Am I on track to retire?").agents == ["goal_planning"]


def test_extract_tickers_skips_acronyms_and_duplicates():
    assert extract_tickers("Is $tsla in my IRA? TSLA vs VTI, ETF or REIT") == ["TSLA", "VTI"]


def test_route_decision_cleans_tickers():
    decision = RouteDecision(
        standalone_query="q", tickers=["tsla", "TSLA", "not a ticker!", "brk.b"]
    )
    assert decision.tickers == ["TSLA", "BRK.B"]


# ---- LLM router ---------------------------------------------------------------------------


def test_llm_route_used_and_validated():
    llm = FakeChatModel(structured_responses=[route("tax", "market", tickers=["tsla"])])
    decision = ask(llm, "Taxes if I sell TSLA?")
    assert decision.agents == ["tax", "market"]
    assert decision.tickers == ["TSLA"]
    assert decision.source == "llm"
    assert llm.structured_schemas == [RouteDecision]
    system, human = llm.calls[0]
    assert system.content == ROUTER_PROMPT
    assert "Latest user message: Taxes if I sell TSLA?" in human.content
    assert "has not saved a portfolio" in human.content


def test_llm_route_accepts_dict_replies():
    class DictLLM:
        def with_structured_output(self, schema, **kwargs):
            return self

        def invoke(self, messages):
            return route("news")

    assert ask(DictLLM()).agents == ["news"]


def test_first_message_is_never_rewritten():
    """Regression: a rewrite dropped the user's holdings from their first message."""
    question = "I hold 40 VTI. Am I on track for $400,000?"
    llm = FakeChatModel(structured_responses=[route("goal_planning", standalone_query="On track?")])
    assert ask(llm, question).standalone_query == question


def test_follow_up_uses_standalone_rewrite_and_context():
    history = [HumanMessage(content="I own TSLA"), AIMessage(content="Noted.")]
    llm = FakeChatModel(
        structured_responses=[route("tax", standalone_query="Taxes if I sell my TSLA shares?")]
    )
    decision = ask(
        llm, "What about taxes?", history=history, summary="User is 35.", has_portfolio=True
    )
    assert decision.standalone_query == "Taxes if I sell my TSLA shares?"
    context = llm.calls[0][1].content
    assert "Summary of earlier conversation: User is 35." in context
    assert "User: I own TSLA" in context and "Finnie: Noted." in context
    assert "has saved a portfolio" in context


def test_blank_rewrite_falls_back_to_question():
    llm = FakeChatModel(structured_responses=[route("tax", standalone_query="  ")])
    assert ask(llm, "And taxes?", history=[HumanMessage(content="hi")]).standalone_query == (
        "And taxes?"
    )


def test_llm_failure_falls_back_to_keywords():
    llm = FakeChatModel(structured_responses=[RuntimeError("timeout")])
    decision = ask(llm, "What's the price of $AAPL?")
    assert decision.source == "keyword"
    assert decision.agents == ["market"]


def test_out_of_scope_has_no_agents():
    llm = FakeChatModel(structured_responses=[route("finance_qa", out_of_scope=True)])
    decision = ask(llm, "Best lasagna recipe?")
    assert decision.out_of_scope and decision.agents == []


def test_low_confidence_keeps_only_top_agent():
    llm = FakeChatModel(structured_responses=[route("tax", "news", confidence=0.2)])
    assert ask(llm).agents == ["tax"]


def test_no_agents_defaults_to_finance_qa():
    llm = FakeChatModel(structured_responses=[route()])
    assert ask(llm).agents == ["finance_qa"]


def test_agents_deduplicated_and_capped():
    llm = FakeChatModel(structured_responses=[route("tax", "tax", "news", "market")])
    assert ask(llm, max_agents=2).agents == ["tax", "news"]


def test_goal_question_with_listed_holdings_adds_portfolio():
    llm = FakeChatModel(structured_responses=[route("goal_planning", "market")] * 3)
    assert ask(llm, "I hold 40 VTI. On track?").agents == ["portfolio", "goal_planning", "market"]
    assert ask(llm, "I own 5 VTI. On track?", max_agents=2).agents == [
        "portfolio",
        "goal_planning",
    ]
    assert ask(llm, "On track?", has_portfolio=True).agents == ["goal_planning", "market"]


def test_goal_label_and_stated_savings_pass_through():
    llm = FakeChatModel(
        structured_responses=[route("goal_planning", goal="retirement", current_savings=20000)]
    )
    decision = ask(llm, "With $20,000 saved, can I retire?", goals=["retirement", "house"])
    assert (decision.goal, decision.current_savings) == ("retirement", 20000)
    assert "Goal labels already used in this conversation: retirement, house." in (
        llm.calls[0][1].content
    )


def test_with_portfolio_for_goals_leaves_other_cases():
    assert with_portfolio_for_goals(["tax"], True, 3) == ["tax"]
    assert with_portfolio_for_goals(["goal_planning"], False, 3) == ["goal_planning"]
    assert with_portfolio_for_goals(["portfolio", "goal_planning"], True, 3) == [
        "portfolio",
        "goal_planning",
    ]
