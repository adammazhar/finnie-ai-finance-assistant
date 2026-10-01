from src.workflow.nodes import TurnOutput
from src.workflow.progress import Progress, specialist
from tests.unit.workflow.conftest import result, route


def events_for(assistant, question, **kwargs):
    events = list(assistant.stream(question, thread_id="t", **kwargs))
    *progress, output = events
    assert all(isinstance(e, Progress) for e in progress)
    assert isinstance(output, TurnOutput)
    return progress, output


def test_multi_agent_turn_reports_each_step(make_assistant):
    assistant, _ = make_assistant(
        [route("market", "news")], agents={"news": result("news", error="TimeoutError")}
    )
    progress, output = events_for(assistant, "TSLA price and news?")

    assert progress[0].message == "Reading your question…"  # first feedback comes at once
    assert progress[1].message == "Choosing the right specialists…"
    started = {p.agent for p in progress if p.kind == "agent_started"}
    assert started == {"market", "news"}
    finished = {p.agent: p.ok for p in progress if p.kind == "agent_finished"}
    assert finished == {"market": True, "news": False}
    assert "Consulting the markets specialist…" in [p.message for p in progress]
    assert "News specialist ran into a problem" in [p.message for p in progress]
    assert progress[-1].message == "Checking the answer…"
    assert output.status == "answered" and output.agents == ["market"]


def test_merge_step_reported_only_for_several_answers(make_assistant):
    assistant, _ = make_assistant([route("market", "news"), route("tax")])
    progress, _ = events_for(assistant, "TSLA price and news?")
    assert "Combining the specialists' answers…" in [p.message for p in progress]
    progress, _ = events_for(assistant, "How are gains taxed?")
    assert "Combining the specialists' answers…" not in [p.message for p in progress]


def test_stream_matches_ask_and_saves_the_turn(make_assistant):
    assistant, _ = make_assistant([route("tax")])
    _, output = events_for(assistant, "How are gains taxed?")
    assert output.answer.startswith("tax answer.")
    assert assistant.state("t")["messages"][0].content == "How are gains taxed?"


def test_non_answers_skip_the_check(make_assistant):
    assistant, _ = make_assistant([route(out_of_scope=True)])
    progress, output = events_for(assistant, "Lasagna?")
    assert output.status == "out_of_scope"
    assert [p.message for p in progress] == [
        "Reading your question…",
        "Choosing the right specialists…",
    ]


def test_specialist_labels():
    assert specialist("goal_planning") == "goal planning"
    assert specialist("finance_qa") == "financial concepts"
    assert specialist("something_new") == "something new"
