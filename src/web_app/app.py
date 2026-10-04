"""Finnie's Streamlit app: streamlit run src/web_app/app.py

The tab bar is a segmented control rather than ``st.tabs``: the app then knows which page
is open, renders only that page, and can place ``st.chat_input`` at the top level on the
Chat page, where Streamlit pins it to the bottom of the window.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = str(Path(__file__).resolve().parent)
# `streamlit run` puts this folder first on sys.path, so a module here could shadow a
# standard-library one (a "profile.py" once broke torch's imports). Import everything as
# src.web_app.* from the project root instead.
sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) != HERE]
if str(ROOT) not in sys.path:  # pragma: no cover - only under `streamlit run`
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.core.llm import LLMConfigurationError  # noqa: E402
from src.web_app import profile_page, services, sidebar, state, theme  # noqa: E402
from src.web_app.tabs import chat, goals, knowledge, markets, portfolio  # noqa: E402

logger = logging.getLogger(__name__)

PAGES: dict[str, Callable[[], None]] = {
    "Chat": chat.render,
    "Portfolio": portfolio.render,
    "Markets": markets.render,
    "Goals": goals.render,
    "Knowledge": knowledge.render,
    state.PROFILE_PAGE: lambda: profile_page.render(first_visit=False),
}
ICONS = {
    "Chat": ":material/chat:",
    "Portfolio": ":material/pie_chart:",
    "Markets": ":material/monitoring:",
    "Goals": ":material/flag:",
    "Knowledge": ":material/menu_book:",
}


@st.cache_resource(show_spinner=False)
def _configure_logging() -> bool:
    from src.utils.logging import configure_logging

    app = services.context().settings.app
    configure_logging(app.log_level, app.log_format)
    return True


def _safe(render: Callable[[], None], name: str) -> None:
    """One page failing shows a message instead of a stack trace."""
    try:
        render()
    except Exception:
        logger.exception("Page failed", extra={"page": name})
        st.error("This page couldn't load just now. Please try again in a moment.")


def _tab_bar() -> None:
    if state.NAV not in st.session_state:
        st.session_state[state.NAV] = state.page()
    st.segmented_control(
        "Navigation",
        state.PAGES,
        format_func=lambda name: f"{ICONS[name]} {name}",
        key=state.NAV,
        on_change=state.on_nav,
        label_visibility="collapsed",
    )


def _remember_browser() -> None:
    """Store a new browser's random ID in a cookie, so its data loads on the next visit.

    The ID is 32 hex characters generated here (never user input); the script only sets
    a first-party cookie on this app's own page.
    """
    new_id = state.take_new_cookie()
    if new_id is None:
        return
    max_age = state.COOKIE_DAYS * 24 * 3600
    st.iframe(
        "<script>"
        "const secure = window.parent.location.protocol === 'https:' ? '; Secure' : '';"
        f"window.parent.document.cookie = '{state.COOKIE}={new_id}; path=/; "
        f"max-age={max_age}; SameSite=Lax' + secure;"
        "</script>",
        height=1,
    )


def _setup_problem() -> str | None:
    """Why Finnie can't start (no usable LLM API key), or None when it can."""
    try:
        services.context()
    except LLMConfigurationError as exc:
        return str(exc)
    return None


def _setup_page(problem: str) -> None:
    """Shown instead of a stack trace when the app isn't configured yet."""
    st.markdown("## Finnie needs an API key to start")
    st.error(problem)
    st.markdown(
        "1. Copy `.env.example` to `.env` in the project folder.\n"
        "2. Set `OPENAI_API_KEY`, or set `ANTHROPIC_API_KEY` and `LLM_PROVIDER=anthropic`.\n"
        "3. Restart Finnie: `python -m src.web_app`, or `docker compose up` with Docker "
        "(it reads the same `.env`)."
    )
    st.caption("Keys stay on your machine in `.env`, which is never committed.")


def main() -> None:
    """One script run: setup page if no API key, onboarding on a first visit, otherwise
    the sidebar, the tab bar, and the selected page."""
    st.set_page_config(
        page_title="Finnie: financial education assistant",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="auto",  # open on desktop, closed on phones (it would cover the page)
    )
    problem = _setup_problem()
    if problem is not None:
        theme.apply("Onboarding")
        _setup_page(problem)
        return
    _configure_logging()
    state.start()
    _remember_browser()
    if not state.onboarded():
        theme.apply("Onboarding")
        _safe(lambda: profile_page.render(first_visit=True), "Onboarding")
        return
    page = state.page()
    theme.apply(page)
    sidebar.render()
    _tab_bar()
    _safe(PAGES[page], page)


main()
