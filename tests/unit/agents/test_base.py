from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agents.base import (
    FALLBACK_ANSWER,
    FINAL_ANSWER_NUDGE,
    AgentRequest,
    RunState,
    load_prompt,
    message_text,
)
from src.agents.finance_qa import FinanceQAAgent
from src.agents.market import MarketAgent
from src.agents.news import NewsAgent
from src.core.models import AgentResult, Holding, UserProfile
from tests.unit.agents.conftest import tool_call


def test_answer_with_kb_citations_and_invented_ones_removed(make_context):
    agent = FinanceQAAgent(make_context("Diversification spreads risk [1]. See also [7]."))
    result = agent.run(AgentRequest(query="What is diversification?"))
    assert result.ok and result.agent == "finance_qa"
    assert result.answer == "Diversification spreads risk [1]. See also."
    assert [s.article_id for s in result.sources] == ["portfolio_management-001"]
    assert result.sources[0].kind == "knowledge_base"
    meta = result.data["meta"]
    assert meta["kb_blocks"] >= 1 and meta["tool_calls"] == [] and meta["latency_ms"] >= 0
    assert result.data["kb"]["confident"] is True


def test_system_prompt_includes_policy_profile_portfolio_and_context(make_context):
    context = make_context("ok")
    agent = FinanceQAAgent(context)
    request = AgentRequest(
        query="What is diversification?",
        profile=UserProfile(
            knowledge_level="advanced",
            risk_tolerance="aggressive",
            age=40,
            investment_horizon_years=25,
        ),
        portfolio=[Holding(ticker="VTI", shares=10)],
        prior_results={
            "portfolio": AgentResult(agent="portfolio", answer="Total value $3,000."),
            "news": AgentResult(agent="news", error="down"),
        },
        guidance=["Extra instruction for this turn."],
        history=[HumanMessage(content="earlier question"), AIMessage(content="earlier answer")],
    )
    agent.run(request)
    messages = context.llm.calls[0]
    system = messages[0]
    assert isinstance(system, SystemMessage)
    text = system.content
    assert load_prompt("_policy").splitlines()[0] in text
    assert "Finance Q&A specialist" in text
    assert (
        "experienced" in text
        and "risk tolerance is aggressive; age 40; investment horizon 25" in text
    )
    assert "VTI x 10" in text and "portfolio: Total value $3,000." in text
    assert "news:" not in text  # failed results aren't passed along
    assert "Extra instruction for this turn." in text
    assert "[1] Diversification > Why it matters" in text
    assert [m.content for m in messages[1:]] == [
        "earlier question",
        "earlier answer",
        "What is diversification?",
    ]


def test_prompt_without_kb_matches_says_so(make_context):
    context = make_context("General answer.")
    result = FinanceQAAgent(context).run(AgentRequest(query="zzz qqq www"))
    assert "No knowledge base passage matched" in context.llm.calls[0][0].content
    assert result.sources == [] and result.data["kb"]["confident"] is False


def test_history_window_is_applied(make_context):
    context = make_context("ok")
    history = [HumanMessage(content=f"m{i}") for i in range(30)]
    FinanceQAAgent(context).run(AgentRequest(query="q", history=history))
    window = context.settings.workflow.history_window
    sent = context.llm.calls[0][1:-1]
    assert len(sent) == window and sent[0].content == f"m{30 - window}"


def test_tool_loop_adds_citable_blocks(make_context):
    context = make_context(
        tool_call("lookup_glossary_term", {"term": "expense ratio"}),
        "A fund's yearly fee [2] matters for diversification [1].",
    )
    result = FinanceQAAgent(context).run(AgentRequest(query="diversification"))
    second = context.llm.calls[1]
    tool_message = second[-1]
    assert isinstance(tool_message, ToolMessage) and tool_message.name == "lookup_glossary_term"
    assert "[2] Expense ratio > Definition" in tool_message.content
    assert {s.article_id for s in result.sources} == {
        "portfolio_management-001",
        "glossary:expense-ratio",
    }
    assert result.data["meta"]["tool_calls"] == ["lookup_glossary_term"]
    assert context.llm.bound_tools  # tools were bound for the loop


def test_unknown_tool_and_failing_tool_are_reported_to_the_model(make_context, market):
    market.fail = {"get_quotes"}
    context = make_context(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "teleport", "args": {}, "id": "a"},
                {"name": "get_quotes", "args": {"tickers": ["AAPL"]}, "id": "b"},
            ],
        ),
        "Live data is unavailable right now.",
    )
    result = MarketAgent(context).run(AgentRequest(query="AAPL price"))
    tool_messages = [m for m in context.llm.calls[1] if isinstance(m, ToolMessage)]
    assert "no tool named teleport" in tool_messages[0].content
    assert "get_quotes failed (DataUnavailableError" in tool_messages[1].content
    assert result.ok and result.data["meta"]["tool_errors"] == [
        "teleport: unknown tool",
        "get_quotes: DataUnavailableError",
    ]


def test_invalid_tool_arguments_are_reported(make_context):
    context = make_context(tool_call("get_quotes", {"tickers": []}), "Done.")
    result = MarketAgent(context).run(AgentRequest(query="price"))
    assert "Error: get_quotes failed (ValidationError" in context.llm.calls[1][-1].content
    assert result.ok


def test_runaway_tool_calls_end_with_a_final_answer(make_context):
    looping = tool_call("lookup_glossary_term", {"term": "etf"})
    context = make_context(looping)  # repeats forever
    result = FinanceQAAgent(context).run(AgentRequest(query="q"))
    limit = context.settings.workflow.agent_max_iterations
    assert len(context.llm.calls) == limit + 1
    assert context.llm.calls[-1][-1].content == FINAL_ANSWER_NUDGE
    assert result.answer == FALLBACK_ANSWER  # the final reply had no text


def test_llm_failure_becomes_error_result(make_context):
    result = FinanceQAAgent(make_context(RuntimeError("provider down"))).run(
        AgentRequest(query="q")
    )
    assert not result.ok and result.error == "RuntimeError: provider down" and result.answer == ""


def test_retrieval_failure_and_missing_retriever(make_context):
    class Broken:
        def retrieve(self, *args, **kwargs):
            raise OSError("index missing")

    result = FinanceQAAgent(make_context("Fine.", retriever_override=Broken())).run(
        AgentRequest(query="q")
    )
    assert result.ok and result.data["meta"]["kb_blocks"] == 0
    assert (
        FinanceQAAgent(make_context("Fine.", retriever_override=None))
        .run(AgentRequest(query="q"))
        .ok
    )


def test_agent_without_rag_categories_skips_retrieval(make_context):
    class NoRag(NewsAgent):
        rag_categories = ()

    result = NoRag(make_context("News summary.")).run(AgentRequest(query="market news"))
    assert "kb" not in result.data and result.data["meta"]["kb_blocks"] == 0


def test_add_chunks_skips_duplicates_and_renumbers(retriever):
    state = RunState(request=AgentRequest(query="q"), agent="finance_qa")
    first = state.add_chunks(retriever.retrieve("diversification investments").chunks)
    again = state.add_chunks(retriever.retrieve("diversification investments").chunks)
    more = state.add_chunks(retriever.retrieve("capital gain long-term").chunks)
    assert [b.number for b in first] == list(range(1, len(first) + 1))
    assert again == []
    assert more[0].number == len(first) + 1


def test_message_text_handles_block_content():
    assert message_text(AIMessage(content="plain")) == "plain"
    blocks = AIMessage(
        content=[
            {"type": "text", "text": "a"},
            "b",
            {"type": "tool_use", "id": "x", "name": "t", "input": {}},
        ]
    )
    assert message_text(blocks) == "ab"


def test_only_failed_prior_results_are_left_out_entirely(make_context):
    context = make_context("ok")
    FinanceQAAgent(context).run(
        AgentRequest(query="q", prior_results={"news": AgentResult(agent="news", error="down")})
    )
    assert "Findings from other Finnie specialists" not in context.llm.calls[0][0].content
