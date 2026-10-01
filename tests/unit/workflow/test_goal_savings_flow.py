"""Asking how much of a saved portfolio counts toward a goal, end to end through the graph."""

import pytest

from src.core.models import Holding
from tests.unit.workflow.conftest import result, route

VTI = Holding(ticker="VTI", shares=10)  # $3,000.00 at the fake market's $300
QUESTION = "Am I on track to retire with $400,000 in 20 years?"
ASKED = "You have a saved portfolio worth $3,000.00. Should I count all of it, part of it, or none"


def retirement(**kwargs):
    return route("goal_planning", goal="retirement", **kwargs)


def savings_notes(agent):
    return [g for g in agent.requests[-1].guidance if "current_balance" in g]


def ask_then_answer(make_assistant, reply, routes=None):
    assistant, team = make_assistant(routes or [retirement()])
    first = assistant.ask(QUESTION, thread_id="t", portfolio=[VTI])
    assert first.status == "needs_input" and first.answer.startswith(ASKED)
    assert team["goal_planning"].requests == []  # nothing runs until the user answers
    second = assistant.ask(reply, thread_id="t")
    return assistant, team, second


@pytest.mark.parametrize(
    "reply, note, stored",
    [
        (
            "All of it",
            "whole portfolio, currently worth $3,000.00",
            {"choice": "all", "amount": None},
        ),
        ("none", "chose not to count their saved portfolio", {"choice": "none", "amount": None}),
        (
            "$1,500",
            "chose to count, from their saved portfolio, $1,500.00",
            {"choice": "amount", "amount": 1500.0},
        ),
    ],
)
def test_answer_resumes_the_goal_question(make_assistant, reply, note, stored):
    assistant, team, out = ask_then_answer(make_assistant, reply)

    assert out.status == "answered" and out.agents == ["goal_planning"]
    request = team["goal_planning"].requests[0]
    assert request.query == QUESTION  # the original question, not "all"
    [fact] = savings_notes(team["goal_planning"])
    assert note in fact
    assert assistant.context.fast_llm.structured_calls == 1  # the reply wasn't re-routed

    state = assistant.state("t")
    assert state["pending_savings"] is None
    assert state["goal_savings"]["retirement"] == stored | {
        "portfolio_value": 3000.0,
        "source": "asked",
    }
    assert [m.content.split(".")[0] for m in state["messages"]][:3] == [
        "Am I on track to retire with $400,000 in 20 years?",
        "You have a saved portfolio worth $3,000",
        reply.split(".")[0],
    ]


def test_follow_up_about_the_same_goal_does_not_ask_again(make_assistant):
    routes = [retirement(), retirement(standalone_query="Retire on track with $900/month?")]
    assistant, team, _ = ask_then_answer(make_assistant, "$1,500", routes)
    out = assistant.ask("What if I add $900 a month instead?", thread_id="t")

    assert out.status == "answered"
    assert len(team["goal_planning"].requests) == 2
    [fact] = savings_notes(team["goal_planning"])
    assert "$1,500.00" in fact
    router_context = assistant.context.fast_llm.calls[-1][1].content
    assert "Goal labels already used in this conversation: retirement." in router_context


def test_a_different_goal_asks_again(make_assistant):
    routes = [retirement(), route("goal_planning", goal="house down payment")]
    assistant, _, _ = ask_then_answer(make_assistant, "all", routes)
    out = assistant.ask("Could I save $60,000 for a house in 5 years?", thread_id="t")
    assert out.status == "needs_input" and out.answer.startswith(ASKED)


def test_all_uses_todays_value_on_later_turns(make_assistant):
    routes = [retirement(), retirement()]
    assistant, team, _ = ask_then_answer(make_assistant, "all", routes)
    assistant.ask("And in 25 years?", thread_id="t", portfolio=[Holding(ticker="VTI", shares=20)])
    [fact] = savings_notes(team["goal_planning"])
    assert "currently worth $6,000.00" in fact


@pytest.mark.parametrize(
    "routes, question",
    [
        ([retirement(current_savings=20000)], "With $20,000 saved, can I retire?"),
        ([retirement()], "I'm 30 with $20,000 saved. Can I retire with $400,000?"),  # regex
    ],
)
def test_savings_stated_in_question_are_used_without_asking(make_assistant, routes, question):
    assistant, team = make_assistant(routes)
    out = assistant.ask(question, thread_id="t", portfolio=[VTI])
    assert out.status == "answered"
    [fact] = savings_notes(team["goal_planning"])
    assert "said they have $20,000.00" in fact
    assert assistant.state("t")["goal_savings"]["retirement"]["source"] == "stated"


def test_unclear_short_reply_asks_again_then_accepts(make_assistant):
    assistant, team = make_assistant([retirement()])
    assistant.ask(QUESTION, thread_id="t", portfolio=[VTI])
    again = assistant.ask("hmm, not sure", thread_id="t")
    assert again.status == "needs_input" and again.answer.startswith("Sorry, I didn't catch that")
    assert assistant.state("t")["pending_savings"] is not None
    out = assistant.ask("none", thread_id="t")
    assert out.status == "answered" and len(team["goal_planning"].requests) == 1


def test_a_new_question_instead_of_an_answer_moves_on(make_assistant):
    assistant, team = make_assistant([retirement(), route("finance_qa")])
    assistant.ask(QUESTION, thread_id="t", portfolio=[VTI])
    out = assistant.ask(
        "Actually, can you first explain what an index fund is and how it works?", thread_id="t"
    )
    assert out.status == "answered" and out.agents == ["finance_qa"]
    assert assistant.state("t")["pending_savings"] is None
    assert team["goal_planning"].requests == []


def test_blocked_reply_keeps_the_question_pending(make_assistant):
    assistant, _ = make_assistant([retirement()])
    assistant.ask(QUESTION, thread_id="t", portfolio=[VTI])
    out = assistant.ask("Help me with insider trading so I can profit", thread_id="t")
    assert out.status == "blocked"
    assert assistant.state("t")["pending_savings"] is not None


def test_no_question_without_a_priced_portfolio(make_assistant):
    assistant, team = make_assistant([retirement(), retirement()])
    out = assistant.ask(QUESTION, thread_id="a")  # no saved portfolio
    assert out.status == "answered" and savings_notes(team["goal_planning"]) == []
    out = assistant.ask(QUESTION, thread_id="b", portfolio=[Holding(ticker="ZZZZ", shares=1)])
    assert out.status == "answered"  # no price for it: nothing to ask about


def test_non_goal_questions_never_ask(make_assistant):
    assistant, _ = make_assistant([route("portfolio")])
    assert assistant.ask("How diversified am I?", thread_id="t", portfolio=[VTI]).status == (
        "answered"
    )


def test_listed_holdings_count_without_asking_even_if_valuation_fails(make_assistant):
    assistant, team = make_assistant(
        [retirement()], agents={"portfolio": result("portfolio", error="RuntimeError: down")}
    )
    out = assistant.ask("I hold 10 VTI. Am I on track to retire?", thread_id="t")
    assert out.status == "answered" and out.agents == ["goal_planning"]
    [fact] = savings_notes(team["goal_planning"])
    assert "couldn't be fetched" in fact


def test_router_savings_not_in_the_question_still_asks(make_assistant):
    """Regression: the router model returned 0 savings for a question that stated none."""
    assistant, _ = make_assistant([retirement(current_savings=0)])
    out = assistant.ask(QUESTION, thread_id="t", portfolio=[VTI])
    assert out.status == "needs_input" and out.answer.startswith(ASKED)
