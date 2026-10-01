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


def main() -> None:
    st.set_page_config(
        page_title="Finnie: financial education assistant",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _configure_logging()
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
