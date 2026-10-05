"""The app shell, onboarding and profile, the sidebar, and the Chat page."""

import logging

from src.core.guardrails import SHORT_DISCLAIMER
from src.core.models import Holding, Source
from src.core.monte_carlo import GoalInputs, simulate
from tests.unit.web_app.conftest import goto, ok, texts
from tests.unit.workflow.conftest import result, route


def ask(app, question):
    app.chat_input(key="chat_input").set_value(question).run()
    return ok(app)


def answer_with(agent, answer, sources=(), data=None):
    citations = {"kb": {str(i): s.article_id or s.url for i, s in enumerate(sources, 1)}}
    return result(
        agent, answer, sources=list(sources), data={"citations": citations, **(data or {})}
    )


# ---- shell ---------------------------------------------------------------------------------


def test_first_visit_shows_onboarding_then_the_app(ui):
    app, _, _ = ui(onboarded=False)
    assert "## Welcome to Finnie" in texts(app.main.markdown)
    assert not app.get("chat_input")
    app.radio(key="profile_level").set_value("intermediate")
    app.radio(key="profile_risk").set_value("aggressive")
    app.number_input(key="profile_age").set_value(30)
    ok(app.button(key="profile_save").click().run())
    profile = app.session_state["finnie_profile"]
    assert (profile.knowledge_level, profile.risk_tolerance, profile.age) == (
        "intermediate",
        "aggressive",
        30,
    )
    assert app.segmented_control(key="nav").value == "Chat"
    assert app.chat_input(key="chat_input")


def test_tab_bar_shows_one_page_and_pins_the_input_only_on_chat(ui):
    app, _, _ = ui()
    nav = app.segmented_control(key="nav")
    assert nav.options == ["Chat", "Portfolio", "Markets", "Goals", "Knowledge"]
    assert app.chat_input(key="chat_input")
    for page in ("Portfolio", "Markets", "Goals", "Knowledge"):
        goto(app, page)
        assert not app.get("chat_input"), page
    # clicking the open tab again keeps it open
    app.segmented_control(key="nav").set_value(None).run()
    assert app.session_state["finnie_page"] == "Knowledge"
    assert app.segmented_control(key="nav").value == "Knowledge"


def test_sidebar(ui):
    app, _, _ = ui()
    assert "## Finnie" in texts(app.sidebar.markdown)
    assert app.sidebar.button(key="new_conversation").label == "New conversation"
    assert app.sidebar.button(key="open_profile").label == "Profile: New to investing, moderate"
    footer = " ".join(texts(app.sidebar.caption))
    assert "educational information only" in footer
    status = " ".join(texts(app.sidebar.markdown))
    assert "**AI model:**" in status and "**yfinance:** on (fake)" in status


def test_profile_page_edits_and_cancels(ui):
    app, team, _ = ui(routes=[route("tax")])
    ok(app.sidebar.button(key="open_profile").click().run())
    assert "## Your profile" in texts(app.main.markdown)
    assert app.segmented_control(key="nav").value is None  # no tab for the profile page
    assert app.radio(key="profile_level").value == "beginner"
    app.radio(key="profile_level").set_value("advanced")
    ok(app.button(key="profile_save").click().run())
    assert app.sidebar.button(key="open_profile").label == "Profile: Experienced, moderate"
    assert app.button(key="starter_0").label == "What's my portfolio's Sharpe ratio and beta?"
    ask(app, "How are gains taxed?")
    assert team["tax"].requests[0].profile.knowledge_level == "advanced"
    app.sidebar.button(key="open_profile").click().run()
    ok(app.button(key="profile_cancel").click().run())
    assert app.session_state["finnie_page"] == "Chat"


def test_quiz_sets_risk_tolerance(ui):
    app, _, _ = ui(onboarded=False)
    app.button(key="quiz_score").click().run()
    assert "Answer all five questions to see a result." in texts(app.warning)
    for i in range(5):
        radio = app.radio(key=f"quiz_{i}")
        radio.set_value(radio.options[0])
    app.button(key="quiz_score").click().run()
    assert app.radio(key="profile_risk").value == "conservative"
    assert any("**conservative** risk tolerance" in s for s in texts(app.success))


# ---- chat ---------------------------------------------------------------------------------


def test_chat_shows_progress_answer_badges_and_sources(ui, monkeypatch):
    # The app reruns once after an answer (to refresh the sidebar), which clears the
    # progress lines; skip that here to see what the user sees while the answer arrives.
    monkeypatch.setattr("streamlit.rerun", lambda *a, **k: None)
    etf = Source(
        title="What Is an ETF?",
        kind="knowledge_base",
        url="https://www.investor.gov/etf",
        article_id="funds_etfs-002",
    )
    term = Source(
        title="Expense ratio",
        kind="knowledge_base",
        url="https://www.investor.gov/glossary",
        article_id="glossary:expense-ratio",
    )
    app, team, _ = ui(
        routes=[route("finance_qa")],
        agents={
            "finance_qa": answer_with(
                "finance_qa", "An ETF costs $5 to $10 a year [1][2][1].", [etf, term]
            )
        },
    )
    assert app.button(key="starter_0").label == "What is an ETF?"
    ask(app, "What is an ETF?")
    user, assistant = app.chat_message
    assert user.markdown[0].value == "What is an ETF?"
    body = texts(assistant.markdown)
    assert "✓ Financial concepts specialist finished" in body  # progress inside the status box
    answer = next(b for b in body if b.startswith("An ETF"))
    assert answer.startswith(r"An ETF costs \$5 to \$10 a year [1][2].")  # escaped, deduped
    assert SHORT_DISCLAIMER in answer
    assert ":blue-badge[Financial concepts]" in body
    assert (
        "**[1]** What Is an ETF?  \n:gray[Original source: [investor.gov](https://www.investor.gov/etf)]"
        in body
    )
    assert not app.get("button") or not [b for b in app.main.button if b.key.startswith("starter_")]
    assert team["finance_qa"].requests[0].profile.knowledge_level == "beginner"
    # sources open the article and the glossary term in Knowledge
    ok(app.button(key="turn_1_source_1").click().run())
    assert app.session_state["finnie_page"] == "Knowledge"
    assert app.selectbox(key="kb_article").value == "funds_etfs-002"
    goto(app, "Chat")
    assert len(app.chat_message) == 2  # the conversation survives navigation
    ok(app.button(key="turn_1_source_2").click().run())
    assert app.text_input(key="kb_term").value == "Expense ratio"


def test_news_sources_show_their_site(ui):
    from datetime import UTC, datetime

    story = Source(
        title="Stocks rise",
        kind="news",
        url="https://news.example.com/a",
        published_at=datetime(2026, 9, 30, tzinfo=UTC),
    )
    citations = {"news": {"1": story.url}}
    agents = {
        "news": result("news", "Stocks rose [N1].", sources=[story], data={"citations": citations})
    }
    app, _, _ = ui(routes=[route("news")], agents=agents)
    ask(app, "Any news?")
    expected = (
        "**[1]** [Stocks rise](https://news.example.com/a)  \n"
        ":gray[News · news.example.com · Sep 30, 2026]"
    )
    assert expected in texts(app.chat_message[1].markdown)


def test_starter_question_is_answered_and_starters_hide(ui):
    app, team, _ = ui(routes=[route("finance_qa")])
    ok(app.button(key="starter_0").click().run())
    assert team["finance_qa"].requests[0].query == "What is an ETF?"
    assert len(app.chat_message) == 2
    assert not [b for b in app.main.button if b.key and b.key.startswith("starter_")]


def test_goal_projection_and_allocation_render_as_charts(ui):
    projection = simulate(
        GoalInputs(
            current_balance=0,
            monthly_contribution=500,
            years=5,
            target_amount=40_000,
            expected_return=0.06,
            volatility=0.1,
            simulations=200,
            seed=1,
        )
    ).model_dump(mode="json")
    agents = {
        "goal_planning": answer_with(
            "goal_planning", "Projection.", data={"goal_projection": projection}
        ),
        "portfolio": answer_with(
            "portfolio",
            "Analysis.",
            data={"portfolio_analysis": {"asset_allocation": {"equity": 1.0}}},
        ),
    }
    app, _, _ = ui(
        routes=[route("portfolio", "goal_planning", current_savings=1000)], agents=agents
    )
    ask(app, "With $1,000 saved, how is my portfolio and am I on track for $40,000?")
    assert [c.key for c in app.get("plotly_chart")] == ["turn_1_fan", "turn_1_donut"]


def test_holdings_from_chat_become_the_saved_portfolio(ui):
    vti = Holding(ticker="VTI", shares=10)
    agents = {
        "portfolio": answer_with(
            "portfolio", "Saved.", data={"holdings": [vti.model_dump(mode="json")]}
        )
    }
    app, _, _ = ui(routes=[route("portfolio")], agents=agents)
    ask(app, "I hold 10 VTI. How diversified am I?")
    assert app.session_state["finnie_portfolio"] == [vti]
    goto(app, "Portfolio")
    assert any(m.label == "Total value" for m in app.metric)


def test_savings_question_in_chat(ui):
    app, team, _ = ui(
        routes=[route("goal_planning", goal="retirement")],
        session={"finnie_portfolio": [Holding(ticker="VTI", shares=10)]},
    )
    ask(app, "Am I on track to retire with $400,000?")
    reply = app.chat_message[1]
    assert any(r"You have a saved portfolio worth \$3,000.00" in t for t in texts(reply.markdown))
    assert not reply.get("feedback")  # no thumbs on a question
    ask(app, "none")
    assert len(team["goal_planning"].requests) == 1


def test_feedback_is_logged(ui, caplog):
    caplog.set_level(logging.INFO, logger="src.web_app.tabs.chat")
    app, _, _ = ui(routes=[route("finance_qa")])
    ask(app, "What is an ETF?")
    app.chat_message[1].get("feedback")[0].set_value(0).run()
    assert any(r.message == "Answer feedback" and r.helpful is False for r in caplog.records)
    assert "Thanks for the feedback!" in texts(app.chat_message[1].caption)


def test_failed_turn_shows_an_apology(ui):
    class Broken:
        def stream(self, *args, **kwargs):
            raise RuntimeError("graph exploded")
            yield  # pragma: no cover

    app, _, _ = ui(assistant=Broken())
    ask(app, "What is an ETF?")
    assert any(
        "Sorry, something went wrong on my side" in t for t in texts(app.chat_message[1].markdown)
    )


def test_conversations_are_listed_and_can_be_reopened(ui):
    app, _, context = ui(routes=[route("finance_qa"), route("tax")])
    context.fast_llm.responses = ["ETFs vs Mutual Funds", "Capital Gains Taxes"]
    ask(app, "What is an ETF and how is it different from a mutual fund exactly?")
    first = app.session_state["finnie_thread_id"]
    ok(app.sidebar.button(key="new_conversation").click().run())
    assert not app.chat_message
    ask(app, "How are gains taxed?")
    titles = [b.label for b in app.sidebar.button if b.key.startswith("conversation_")]
    assert titles == ["Capital Gains Taxes", "ETFs vs Mutual Funds"]  # written by the model
    current = app.sidebar.button(key=f"conversation_{app.session_state['finnie_thread_id']}")
    assert current.proto.type == "secondary"  # the open conversation is highlighted
    assert app.sidebar.button(key=f"conversation_{first}").proto.type == "tertiary"
    assert "Recent conversations" in texts(app.sidebar.caption)
    ok(app.sidebar.button(key=f"conversation_{first}").click().run())
    assert app.session_state["finnie_thread_id"] == first
    assert app.chat_message[0].markdown[0].value.startswith("What is an ETF")


# ---- voice: read aloud and speech to text --------------------------------------------------


def read_aloud_frames(app):
    return [e.proto.srcdoc for e in app.get("iframe") if 'id="speak"' in e.proto.srcdoc]


def test_each_answer_can_be_read_aloud_without_the_disclaimer(ui):
    reply = answer_with("finance_qa", "An **ETF** holds many stocks [1].")
    app, _, _ = ui(routes=[route("finance_qa")], agents={"finance_qa": reply})
    ask(app, "What is an ETF?")
    [frame] = read_aloud_frames(app)
    assert '"An ETF holds many stocks."' in frame  # the text to speak, as a JS string
    assert "educational information only" not in frame and "[1]" not in frame
    assert "speechSynthesis" in frame and "🔊 Read aloud" in frame


def test_an_answer_with_nothing_to_say_gets_no_read_aloud_button(monkeypatch):
    from src.core.guardrails import SHORT_DISCLAIMER
    from src.web_app.tabs import chat

    frames = []
    monkeypatch.setattr(chat.st, "iframe", lambda *a, **k: frames.append(a))
    chat._read_aloud(SHORT_DISCLAIMER, 0)
    assert frames == []


def test_the_mic_is_offered_only_with_an_openai_key(ui, monkeypatch):
    app, _, _ = ui()
    assert app.chat_input(key="chat_input").placeholder == "Message Finnie…"  # no key in tests
    from src.web_app import services

    monkeypatch.setattr(services, "voice_ready", lambda: True)
    app.run()
    assert app.chat_input(key="chat_input").placeholder == "Message Finnie… (or use the mic)"


def _submit_script(text):
    """What the chat page does with a submission from the chat box (text and/or audio)."""
    from types import SimpleNamespace

    import streamlit as st

    from src.web_app.tabs import chat

    audio = SimpleNamespace(getvalue=lambda: b"wav", name="recording.wav")
    plain = SimpleNamespace(text="typed only")  # a submission without audio
    st.session_state["plain"] = chat._submission(plain)
    st.session_state["string"] = chat._submission("a string")
    st.session_state["nothing"] = chat._submission(None)
    if not st.session_state.get("ran"):
        st.session_state["ran"] = True
        chat._submission(SimpleNamespace(text=text, audio=audio))  # reruns the script


def submit(monkeypatch, text="", transcript="What is an ETF?", error=None):
    """Run ``_submit_script`` with transcription replaced (undone after the test)."""
    from types import SimpleNamespace

    from streamlit.testing.v1 import AppTest

    from src.core.voice import VoiceError
    from src.web_app import services

    def fake_transcribe(audio, settings, filename):
        if error:
            raise VoiceError(error)
        return transcript

    monkeypatch.setattr("src.web_app.tabs.chat.transcribe", fake_transcribe)
    monkeypatch.setattr(services, "context", lambda: SimpleNamespace(settings=None))
    return AppTest.from_function(_submit_script, kwargs={"text": text}, default_timeout=60).run()


def test_a_recording_goes_into_the_box_to_check_not_straight_to_the_answer(monkeypatch):
    from src.web_app.tabs.chat import CHECK_TRANSCRIPT, NOTICE, PREFILL

    app = submit(monkeypatch, text="Quick one:")
    assert app.session_state[PREFILL] == "Quick one: What is an ETF?"
    assert app.session_state[NOTICE] == ("toast", CHECK_TRANSCRIPT)
    assert app.session_state["plain"] == "typed only"
    assert app.session_state["string"] == "a string" and app.session_state["nothing"] is None


def test_a_failed_transcription_keeps_the_typed_text_and_explains(monkeypatch):
    from src.web_app.tabs.chat import NOTICE, PREFILL

    app = submit(monkeypatch, text="My draft", error="Couldn't transcribe the recording just now.")
    assert app.session_state[PREFILL] == "My draft"
    assert app.session_state[NOTICE] == ("warning", "Couldn't transcribe the recording just now.")


def test_the_transcript_is_put_in_the_chat_box_with_a_note(ui):
    from src.web_app.tabs.chat import CHECK_TRANSCRIPT, NOTICE, PREFILL

    app, _, _ = ui(session={PREFILL: "What is an ETF?", NOTICE: ("toast", CHECK_TRANSCRIPT)})
    assert app.chat_input(key="chat_input").value == "What is an ETF?"
    assert [t.value for t in app.toast] == [CHECK_TRANSCRIPT]
    assert not app.chat_message  # nothing was sent
    warned, _, _ = ui(session={NOTICE: ("warning", "Couldn't transcribe.")})
    assert "Couldn't transcribe." in texts(warned.warning)
