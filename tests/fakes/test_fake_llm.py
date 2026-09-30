import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from tests.fakes.llm import FakeChatModel


class Route(BaseModel):
    agent: str


def test_scripted_replies_repeat_last_and_record_calls():
    fake = FakeChatModel(
        responses=[
            "one",
            AIMessage(
                content="",
                tool_calls=[{"name": "get_quote", "args": {"ticker": "AAPL"}, "id": "1"}],
            ),
        ]
    )
    assert fake.invoke("a").content == "one"
    second = fake.invoke("b")
    assert second.tool_calls[0]["name"] == "get_quote"
    assert fake.invoke("c").tool_calls  # last response repeats
    assert len(fake.calls) == 3


def test_raises_scripted_exceptions():
    fake = FakeChatModel(responses=[TimeoutError("slow")])
    with pytest.raises(TimeoutError):
        fake.invoke("x")


def test_empty_script_is_a_test_bug():
    with pytest.raises(AssertionError):
        FakeChatModel(responses=[]).invoke("x")


def test_structured_output_validates_dicts_and_passes_objects():
    fake = FakeChatModel(structured_responses=[{"agent": "tax"}, Route(agent="news")])
    runnable = fake.with_structured_output(Route)
    assert runnable.invoke("q1") == Route(agent="tax")
    assert runnable.invoke("q2") == Route(agent="news")
    assert fake.structured_schemas == [Route]
    assert fake.with_structured_output(dict).invoke("q3") == {"agent": "tax"}


def test_bind_tools_records_tools():
    fake = FakeChatModel()
    assert fake.bind_tools(["t1"]) is fake
    assert fake.bound_tools == ["t1"]
