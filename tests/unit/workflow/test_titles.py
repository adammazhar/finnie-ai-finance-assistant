"""Conversation titles: written after the first answer, rewritten once after the third."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.workflow.titles import PROMPT, clean, fallback_title, write_title
from tests.fakes.llm import FakeChatModel
from tests.unit.workflow.conftest import route


def titled_assistant(make_assistant, titles):
    """An assistant whose fast model routes to finance_qa and writes the given titles."""
    assistant, team = make_assistant([route("finance_qa")] * 6, fast=titles)
    return assistant, team


def title_prompts(assistant):
    return [c[1].content for c in assistant.context.fast_llm.calls if c[0].content == PROMPT]


def test_first_title_comes_from_the_first_question_and_answer(make_assistant):
    assistant, _ = titled_assistant(make_assistant, ["ETF Basics"])
    assistant.ask("What is your name?", thread_id="t")
    assert assistant.update_title("t") == "ETF Basics"
    [prompt] = title_prompts(assistant)
    assert prompt == "User: What is your name?\nFinnie: finance_qa answer."
    state = assistant.state("t")
    assert (state["title"], state["title_final"]) == ("ETF Basics", False)


def test_title_is_rewritten_once_after_the_third_question_then_fixed(make_assistant):
    assistant, _ = titled_assistant(make_assistant, ["Getting Started", "Roth IRA Rules"])
    for question in ("What is your name?", "Can I save for retirement?", "How do Roth IRAs work?"):
        assistant.ask(question, thread_id="t")
        title = assistant.update_title("t")
    assert title == "Roth IRA Rules"
    first, second = title_prompts(assistant)
    assert first.startswith("User: What is your name?")
    # the rewrite sees the whole conversation so far, so the real topic shows up
    assert "User: How do Roth IRAs work?" in second and "User: Can I save" in second
    assert assistant.state("t")["title_final"] is True
    assistant.ask("And the income limits?", thread_id="t")
    assert assistant.update_title("t") == "Roth IRA Rules"
    assert len(title_prompts(assistant)) == 2  # no more model calls once fixed


def test_second_question_keeps_the_first_title(make_assistant):
    assistant, _ = titled_assistant(make_assistant, ["ETF Basics"])
    assistant.ask("What is an ETF?", thread_id="t")
    assistant.update_title("t")
    assistant.ask("And mutual funds?", thread_id="t")
    assert assistant.update_title("t") == "ETF Basics"
    assert len(title_prompts(assistant)) == 1


def test_model_failure_uses_a_topic_label_and_retries_the_rewrite(make_assistant):
    assistant, _ = titled_assistant(make_assistant, [RuntimeError("down")])
    assistant.ask("What is an ETF?", thread_id="t")
    assert assistant.update_title("t") == "Investing Basics"  # never the raw message
    for question in ("Q2?", "Q3?"):
        assistant.ask(question, thread_id="t")
    assert assistant.update_title("t") == "Investing Basics"
    assert assistant.state("t")["title_final"] is False  # try again next time


def test_no_questions_yet(make_assistant):
    assistant, _ = titled_assistant(make_assistant, ["x"])
    assert assistant.update_title("empty") is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('"Roth IRA Basics."', "Roth IRA Basics"),
        ("Title: Market Check", "Market Check"),
        ("**Savings Goal**\nextra line", "Savings Goal"),
        ("one two three four five six seven eight", "one two three four five six"),
        ("   ", None),
        (
            "Supercalifragilisticexpialidocious Antidisestablishmentarianism Plan",
            "Supercalifragilisticexpialidocious Antidisestab…",
        ),
    ],
)
def test_clean(raw, expected):
    assert clean(raw) == expected


def test_fallback_and_summary():
    assert fallback_title(["tax"]) == "Investment Taxes"
    assert fallback_title(["new_agent"]) == "New Agent Question"
    assert fallback_title([]) == "New Conversation"
    llm = FakeChatModel(responses=["Bond Funds"])
    messages = [HumanMessage(content="Bonds?"), AIMessage(content="Bonds are loans.")]
    assert write_title(llm, messages, summary="Talked about savings.") == "Bond Funds"
    assert llm.calls[0][1].content.startswith("Earlier: Talked about savings.\nUser: Bonds?")
