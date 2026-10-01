from langchain_core.messages import AIMessage, HumanMessage

from src.workflow.evaluation import (
    KEYWORD_FLOOR,
    RoutingCase,
    evaluate,
    keyword_router,
    llm_router,
    load_cases,
)
from src.workflow.router import RouteDecision
from tests.fakes.llm import FakeChatModel
from tests.unit.workflow.conftest import route


def decision(*agents, out_of_scope=False, source="llm"):
    return RouteDecision(
        standalone_query="q", agents=list(agents), out_of_scope=out_of_scope, source=source
    )


def test_case_scoring():
    case = RoutingCase(question="q", agents=["market", "news"], alternatives=[["market"]])
    assert case.accepts(decision("market", "news", "tax"))
    assert case.accepts(decision("market"))
    assert not case.accepts(decision("news"))
    assert case.exact(decision("market")) and not case.exact(decision("market", "tax"))
    assert not case.accepts(decision(out_of_scope=True))

    declined = RoutingCase(question="lasagna?", out_of_scope=True)
    assert declined.accepts(decision(out_of_scope=True)) and declined.exact(
        decision(out_of_scope=True)
    )
    assert not declined.accepts(decision("finance_qa"))


def test_case_history_alternates_roles():
    case = RoutingCase(question="q", agents=["tax"], history=["hi", "hello", "more"])
    assert [type(m) for m in case.messages()] == [HumanMessage, AIMessage, HumanMessage]


def test_labelled_set_loads():
    cases = load_cases()
    assert len(cases) >= 60
    assert sum(c.out_of_scope for c in cases) >= 5
    assert sum(len(c.agents) > 1 for c in cases) >= 5
    assert sum(bool(c.history) for c in cases) >= 4


def test_evaluate_reports_accuracy_extras_and_misses():
    cases = [
        RoutingCase(question="a", agents=["tax"]),
        RoutingCase(question="b", agents=["market"]),
        RoutingCase(question="c", out_of_scope=True),
    ]
    replies = {
        "a": decision("tax", "news"),
        "b": decision("news", source="keyword"),
        "c": decision(out_of_scope=True),
    }
    report = evaluate(lambda case: replies[case.question], cases, "llm")
    assert report.cases == 3
    assert report.accuracy == 2 / 3 and report.exact == 1 / 3
    assert report.extra_agents == 1 and report.keyword_fallbacks == 1
    assert report.misses == ["'b': wanted market, got news"]
    assert report.latency_ms_p50 >= 0


def test_evaluate_empty_and_out_of_scope_miss():
    assert evaluate(lambda case: decision(), [], "llm").accuracy == 0.0
    report = evaluate(
        lambda case: decision(out_of_scope=True), [RoutingCase(question="x", agents=["tax"])], "llm"
    )
    assert report.misses == ["'x': wanted tax, got out_of_scope"]


def test_llm_and_keyword_routers():
    case = RoutingCase(
        question="And taxes?", agents=["tax"], history=["TSLA?", "Up 1%."], has_portfolio=True
    )
    llm = FakeChatModel(structured_responses=[route("tax")])
    assert llm_router(llm, max_agents=3, min_confidence=0.5)(case).agents == ["tax"]
    assert "User: TSLA?" in llm.calls[0][1].content
    assert keyword_router(3)(case).agents == ["tax"]


def test_keyword_router_regression_floor():
    """The fallback router must stay above its floor on the labelled set (no API needed)."""
    report = evaluate(keyword_router(3), load_cases(), "keyword")
    assert report.accuracy >= KEYWORD_FLOOR, "\n".join(report.misses)
