"""Saved data per browser: refresh, restart, rename, delete, and "Delete my data".

Each ``ui(...)`` call is a new app session against the same SQLite file, so a second
call is what a browser refresh looks like; with ``persistent=True`` the workflow's memory
is in that file too, so it's also what an app restart looks like.
"""

import logging

from src.core.models import Holding
from src.web_app import services
from tests.unit.web_app.conftest import BROWSER, ok, texts
from tests.unit.workflow.conftest import route

VTI = Holding(ticker="VTI", shares=10)


def ask(app, question):
    app.chat_input(key="chat_input").set_value(question).run()
    return ok(app)


def conversation_buttons(app):
    return [b for b in app.sidebar.button if b.key and b.key.startswith("conversation_")]


def onboard(app, level="advanced"):
    app.radio(key="profile_level").set_value(level)
    return ok(app.button(key="profile_save").click().run())


def test_refresh_and_restart_keep_profile_portfolio_and_conversations(ui):
    app, _, _ = ui(routes=[route("finance_qa")] * 4, onboarded=None, page=None, persistent=True)
    onboard(app)

    ask(app, "What is an ETF?")
    thread = app.session_state["finnie_thread_id"]
    services.store().save_portfolio(BROWSER, [VTI])  # as the Portfolio page would

    # a new session for the same browser: a refresh (and, with persistent memory, a restart)
    again, team2, _ = ui(
        routes=[route("finance_qa")] * 4, onboarded=None, page=None, persistent=True
    )
    assert "## Welcome to Finnie" not in texts(again.main.markdown)  # no onboarding again
    assert again.session_state["finnie_profile"].knowledge_level == "advanced"
    assert again.session_state["finnie_portfolio"] == [VTI]
    assert again.session_state["finnie_thread_id"] == thread  # back in the same chat
    assert again.chat_message[0].markdown[0].value == "What is an ETF?"
    assert len(conversation_buttons(again)) == 1
    # the workflow remembers the conversation: a follow-up sees the earlier turn
    ask(again, "And how do they differ from mutual funds?")
    history = [m.content for m in team2["finance_qa"].requests[0].history]
    assert history[:2] == ["What is an ETF?", "finance_qa answer."]


def test_a_new_browser_gets_a_cookie(ui):
    app, _, _ = ui(browser=None)
    [script] = [e.proto.srcdoc for e in app.get("iframe") if "finnie_id=" in e.proto.srcdoc]
    browser = app.session_state["finnie_browser_id"]
    assert f"finnie_id={browser}; path=/; max-age={400 * 24 * 3600}; SameSite=Lax" in script
    assert len(browser) == 32
    app.run()  # set once per session, not on every rerun
    assert not [e for e in app.get("iframe") if "finnie_id=" in e.proto.srcdoc]


def test_rename_inline_and_it_sticks(ui):
    app, _, context = ui(routes=[route("finance_qa")] * 4)
    context.fast_llm.responses = ["Auto Title", "Another Auto Title"]
    ask(app, "What is your name?")
    thread = app.session_state["finnie_thread_id"]
    assert conversation_buttons(app)[0].label == "Auto Title"
    ok(app.sidebar.button(key=f"rename_start_{thread}").click().run())
    box = app.sidebar.text_input(key=f"rename_{thread}")
    assert box.value == "Auto Title"
    box.set_value("  My   retirement plan ").run()  # Enter saves
    assert conversation_buttons(app)[0].label == "My retirement plan"
    # the third-question retitle never replaces a name the user chose
    ask(app, "Q2?")
    ask(app, "Q3?")
    assert conversation_buttons(app)[0].label == "My retirement plan"
    saved = services.store().load(BROWSER).conversations[0]
    assert (saved.title, saved.title_source) == ("My retirement plan", "user")


def test_rename_cancel_and_empty_name_keep_the_title(ui):
    app, _, context = ui(routes=[route("finance_qa")])
    context.fast_llm.responses = ["Auto Title"]
    ask(app, "What is an ETF?")
    thread = app.session_state["finnie_thread_id"]
    app.sidebar.button(key=f"rename_start_{thread}").click().run()
    ok(app.sidebar.button(key=f"rename_cancel_{thread}").click().run())
    assert conversation_buttons(app)[0].label == "Auto Title"
    app.sidebar.button(key=f"rename_start_{thread}").click().run()
    app.sidebar.text_input(key=f"rename_{thread}").set_value("   ")
    ok(app.sidebar.button(key=f"rename_save_{thread}").click().run())
    assert conversation_buttons(app)[0].label == "Auto Title"


def test_delete_a_conversation_with_confirmation(ui):
    app, _, context = ui(routes=[route("finance_qa")] * 2)
    context.fast_llm.responses = ["First Chat", "Second Chat"]
    ask(app, "What is an ETF?")
    first = app.session_state["finnie_thread_id"]
    app.sidebar.button(key="new_conversation").click().run()
    ask(app, "What is a bond?")
    second = app.session_state["finnie_thread_id"]
    assistant = services.assistant()
    assert assistant.state(second)["messages"]

    ok(app.sidebar.button(key=f"delete_start_{second}").click().run())
    assert any("Delete **Second Chat**?" in t for t in texts(app.sidebar.markdown))
    ok(app.sidebar.button(key=f"delete_cancel_{second}").click().run())
    assert len(conversation_buttons(app)) == 2  # cancel keeps it

    app.sidebar.button(key=f"delete_start_{second}").click().run()
    ok(app.sidebar.button(key=f"delete_confirm_{second}").click().run())
    assert [b.label for b in conversation_buttons(app)] == ["First Chat"]
    assert assistant.state(second) == {}  # its workflow memory is gone too
    assert [c.thread_id for c in services.store().load(BROWSER).conversations] == [first]
    assert app.session_state["finnie_thread_id"] not in (first, second)  # a fresh chat
    assert not app.chat_message

    # deleting a conversation that isn't open leaves the open one alone
    app.sidebar.button(key=f"conversation_{first}").click().run()
    app.sidebar.button(key="new_conversation").click().run()
    ask(app, "Third?")
    open_now = app.session_state["finnie_thread_id"]
    app.sidebar.button(key=f"delete_start_{first}").click().run()
    ok(app.sidebar.button(key=f"delete_confirm_{first}").click().run())
    assert app.session_state["finnie_thread_id"] == open_now


def test_delete_my_data_starts_fresh(ui):
    app, _, _ = ui(routes=[route("finance_qa")], onboarded=None, page=None)
    onboard(app)
    ask(app, "What is an ETF?")
    thread = app.session_state["finnie_thread_id"]
    ok(app.sidebar.button(key="open_profile").click().run())
    assert any(c.startswith("What Finnie stores:") for c in texts(app.caption))
    ok(app.button(key="delete_my_data").click().run())
    ok(app.button(key="delete_my_data_cancel").click().run())
    assert services.store().load(BROWSER).profile is not None  # cancel keeps everything
    app.button(key="delete_my_data").click().run()
    ok(app.button(key="delete_my_data_confirm").click().run())
    assert "## Welcome to Finnie" in texts(app.main.markdown)  # onboarding again
    saved = services.store().load(BROWSER)
    assert (saved.profile, saved.portfolio, saved.conversations) == (None, [], [])
    assert services.assistant().state(thread) == {}
    assert not conversation_buttons(app)


def test_message_content_never_reaches_the_logs(ui, caplog):
    caplog.set_level(logging.DEBUG)
    secret = "My salary is 123456 and my SSN-like code is ZQ-998877"
    app, _, _ = ui(routes=[route("finance_qa")])
    ask(app, secret)
    for record in caplog.records:
        text = record.getMessage() + " " + " ".join(str(v) for v in record.__dict__.values())
        assert "123456" not in text and "ZQ-998877" not in text, record.name


def test_a_title_lost_to_an_early_refresh_is_recovered(ui, tmp_path):
    import sqlite3

    app, _, context = ui(routes=[route("finance_qa")], persistent=True)
    context.fast_llm.responses = ["Exchange-Traded Funds"]
    ask(app, "What is an ETF?")
    with sqlite3.connect(tmp_path / "finnie.sqlite") as db:  # as if the page closed first
        db.execute("UPDATE conversations SET title = NULL, title_source = NULL")
    again, _, _ = ui(persistent=True)
    assert [b.label for b in conversation_buttons(again)] == ["Exchange-Traded Funds"]
    assert services.store().load(BROWSER).conversations[0].title == "Exchange-Traded Funds"


def test_an_untitled_conversation_without_a_workflow_title_stays_untitled(ui):
    ui()
    services.store().save_chat(
        BROWSER, "orphan", [{"role": "user", "content": "q", "output": None}]
    )
    again, _, _ = ui()
    assert "New conversation" in [b.label for b in conversation_buttons(again)]
