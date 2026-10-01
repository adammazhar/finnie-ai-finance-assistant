from langchain_core.messages import AIMessage, HumanMessage

from src.workflow.nodes import _history, _latest_question, _team_note


def test_latest_question_and_history():
    messages = [
        HumanMessage(content=" first "),
        AIMessage(content="reply"),
        HumanMessage(content="second"),
    ]
    assert _latest_question({"messages": messages}) == "second"
    assert _latest_question({"messages": [AIMessage(content="only a reply")]}) == ""
    assert _latest_question({}) == ""
    assert [m.content for m in _history({"messages": messages}, 1)] == ["reply"]


def test_team_note_lists_other_agents_once():
    assert _team_note("tax", [["tax"]]) is None
    note = _team_note("tax", [["portfolio", "tax"], ["portfolio"]])
    assert note is not None and "(portfolio)" in note
