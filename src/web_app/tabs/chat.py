"""Chat page: progress while specialists work, the streamed answer, sources, and charts.

``st.chat_input`` is called at the top level of the page, so Streamlit pins it to the
bottom of the window with the conversation scrolling above it.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from src.core.models import Holding, Source
from src.core.monte_carlo import SimulationResult
from src.web_app import charts, services, state
from src.web_app.formatting import agent_badges, domain, freshness_caption, md, stream_words
from src.workflow.nodes import TurnOutput
from src.workflow.progress import Progress

logger = logging.getLogger(__name__)

STARTERS = {
    "beginner": [
        "What is an ETF?",
        "How does compound interest work?",
        "What's the difference between stocks and bonds?",
        "How is the stock market doing today?",
    ],
    "intermediate": [
        "How diversified is my portfolio?",
        "How are long-term capital gains taxed?",
        "What does a high P/E ratio tell me?",
        "Any news on Nvidia this week?",
    ],
    "advanced": [
        "What's my portfolio's Sharpe ratio and beta?",
        "How does tax-loss harvesting interact with the wash sale rule?",
        "Is SPY above its 200-day moving average?",
        "How does duration affect bond fund prices when rates rise?",
    ],
}
ERROR_REPLY = "Sorry, something went wrong on my side. Please try again in a moment."
LOGGED_FEEDBACK = "chat_feedback_logged"
ANCHOR = "finnie-answer-"
SCROLL_AFTER_RERUN = "finnie_scroll_after_rerun"
DRAWN = "finnie_question_drawn"


def _run_turn(prompt: str) -> TurnOutput:
    """Run one turn, showing each step as the specialists work."""
    output: TurnOutput | None = None
    with st.status("Reading your question…", expanded=False) as box:
        try:
            for event in services.assistant().stream(
                prompt,
                thread_id=state.thread_id(),
                profile=state.profile(),
                portfolio=state.portfolio() or None,
            ):
                if isinstance(event, Progress):
                    box.update(label=event.message)
                    if event.kind == "agent_finished":
                        box.write(("✓ " if event.ok else "⚠ ") + event.message)
                else:
                    output = event
        except Exception:
            logger.exception("Chat turn failed")
        if output is None:
            box.update(label="Something went wrong", state="error")
            return TurnOutput(answer=ERROR_REPLY, status="fallback")
        label = {
            "needs_input": "One quick question first",
            "blocked": "Finnie can't help with that",
            "out_of_scope": "Outside Finnie's topics",
        }.get(output.status, "Done")
        box.update(label=label, state="complete")
    _sync_portfolio()
    return output


def _sync_portfolio() -> None:
    """Holdings the user typed in chat become the saved portfolio on every page."""
    saved = services.assistant().state(state.thread_id()).get("portfolio") or []
    holdings = [Holding.model_validate(h) for h in saved]
    if holdings and holdings != state.portfolio():
        state.set_portfolio(holdings)


def _charts(output: TurnOutput, key: str) -> None:
    projection = output.data.get("goal_planning", {}).get("goal_projection")
    if projection:
        result = SimulationResult.model_validate(projection)
        st.plotly_chart(charts.fan_chart(result), width="stretch", key=f"{key}_fan")
    analysis = output.data.get("portfolio", {}).get("portfolio_analysis")
    if analysis and analysis.get("asset_allocation"):
        figure = charts.allocation_donut(analysis["asset_allocation"])
        st.plotly_chart(figure, width="stretch", key=f"{key}_donut")


def _source(number: int, source: Source, key: str) -> None:
    """A knowledge base article (opens in Knowledge) and its original source, separately."""
    site = domain(source.url)
    original = f"[{site}]({source.url})" if site else None
    if source.kind == "news":
        title = f"[{md(source.title)}]({source.url})" if source.url else md(source.title)
        when = f" · {source.published_at:%b %d, %Y}" if source.published_at else ""
        st.markdown(f"**[{number}]** {title}  \n:gray[News · {site or 'unknown site'}{when}]")
        return
    article = source.article_id or ""
    text, button = st.columns([5, 2], vertical_alignment="center")
    detail = f"Original source: {original}" if original else "Finnie knowledge base"
    text.markdown(f"**[{number}]** {md(source.title)}  \n:gray[{detail}]")
    if article.startswith("glossary:"):
        button.button(
            "Open in Glossary",
            key=f"{key}_source_{number}",
            on_click=state.open_glossary,
            args=(source.title,),
            width="stretch",
        )
    elif article:
        button.button(
            "Read article",
            key=f"{key}_source_{number}",
            on_click=state.open_article,
            args=(article, article.rsplit("-", 1)[0]),
            width="stretch",
        )


def _feedback(index: int) -> None:
    rating = st.feedback("thumbs", key=f"feedback_{index}")
    logged: set[int] = st.session_state.setdefault(LOGGED_FEEDBACK, set())
    if rating is not None and index not in logged:
        logged.add(index)
        logger.info(
            "Answer feedback",
            extra={"thread_id": state.thread_id(), "turn": index, "helpful": rating == 1},
        )
        st.caption("Thanks for the feedback!")


def _extras(output: TurnOutput, index: int) -> None:
    if output.agents:
        st.markdown(agent_badges(output.agents))
    caption = freshness_caption(output.freshness)
    if caption:
        st.caption(caption)
    key = f"turn_{index}"
    _charts(output, key)
    if output.sources:
        with st.expander(f"Sources ({len(output.sources)})"):
            for number, source in enumerate(output.sources, 1):
                _source(number, source, key)
    if output.status == "answered":
        _feedback(index)


def _anchor(index: int) -> None:
    """Marks where an answer starts, so the page can scroll to it."""
    st.markdown(f'<div id="{ANCHOR}{index}"></div>', unsafe_allow_html=True)


def _scroll_to(index: int) -> None:
    """Scroll so the answer's first line sits just under the tab bar.

    Retries for a moment, because the page is still laying out when the script runs.
    """
    # A fixed script with only an integer filled in: no user or model text goes in here.
    st.iframe(
        "<script>"
        f"const target = {ANCHOR!r} + {index};"
        "let tries = 8;"
        "function go() {"
        "  const el = window.parent.document.getElementById(target);"
        "  if (el) el.scrollIntoView({block: 'start'});"
        "  if (--tries > 0) setTimeout(go, 150);"
        "}"
        "setTimeout(go, 100);"
        "</script>",
        height=1,
    )


def _history() -> None:
    for index, entry in enumerate(state.chat()):
        with st.chat_message(entry["role"]):
            if entry["role"] == "assistant":
                _anchor(index)
            st.markdown(md(entry["content"]))
            if entry.get("output"):
                _extras(TurnOutput.model_validate(entry["output"]), index)


def _intro() -> None:
    with st.container(key="chat_intro"):
        st.markdown("### Hi, I'm Finnie. What would you like to learn about?")
        st.caption(
            "Ask about investing basics, your portfolio, markets, savings goals, or how "
            "investments are taxed."
        )
    questions = STARTERS[state.profile().knowledge_level]
    with st.container(key="starters"):
        columns = st.columns(2)
        for i, question in enumerate(questions):
            columns[i % 2].button(
                question,
                key=f"starter_{i}",
                on_click=state.queue_prompt,
                args=(question,),
                width="stretch",
            )


def _answer(prompt: str, scroll: bool) -> None:
    state.add_chat("user", prompt)
    with st.chat_message("user"):
        st.markdown(md(prompt))
    with st.chat_message("assistant"):
        index = len(state.chat())
        _anchor(index)
        if scroll:
            _scroll_to(index)
        output = _run_turn(prompt)
        st.write_stream(stream_words(md(output.answer)))
        _extras(output, index)
    payload: dict[str, Any] = output.model_dump(mode="json")
    state.add_chat("assistant", output.answer, payload)
    thread = state.thread_id()
    try:
        state.set_title(thread, services.assistant().update_title(thread))
    except Exception:  # a title is cosmetic; never lose the answer over it
        logger.exception("Titling the conversation failed")
    if scroll:
        st.session_state[SCROLL_AFTER_RERUN] = index
    st.rerun()  # redraw once so the sidebar lists this conversation with its title


def render() -> None:
    typed = st.chat_input("Message Finnie…", key="chat_input")
    waiting = None if typed else state.peek_prompt()
    if waiting and not st.session_state.pop(DRAWN, False):
        # A button sent this question from another page. Draw the chat with the question
        # first, so the previous page doesn't linger (faded) while the answer is written.
        _history()
        with st.chat_message("user"):
            st.markdown(md(waiting))
        st.session_state[DRAWN] = True
        st.rerun()
    queued = None if typed else state.take_prompt()
    scroll = state.take_scroll()
    prompt = typed or queued
    if not state.chat() and not prompt:
        _intro()
    _history()
    after_rerun = st.session_state.pop(SCROLL_AFTER_RERUN, None)
    if after_rerun is not None:
        _scroll_to(after_rerun)
    if prompt:
        _answer(prompt, scroll=bool(queued) and scroll)
