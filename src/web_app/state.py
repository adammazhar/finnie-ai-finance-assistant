"""Typed accessors for ``st.session_state``.

A browser session holds several conversations (each its own workflow thread), the user's
profile and portfolio, and which page is open. Navigation helpers are written to be used
as widget callbacks, because a widget's value can only be set before it is drawn.
"""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from src.core.models import Holding, UserProfile

PAGES = ["Chat", "Portfolio", "Markets", "Goals", "Knowledge"]
PROFILE_PAGE = "Profile"

NAV = "nav"  # the segmented control's key
PAGE = "finnie_page"
THREAD = "finnie_thread_id"
PROFILE = "finnie_profile"
ONBOARDED = "finnie_onboarded"
PORTFOLIO = "finnie_portfolio"
CHAT = "finnie_chat"
CONVERSATIONS = "finnie_conversations"
PENDING = "finnie_pending_prompt"
ACTIVITY = "finnie_activity"  # increases with every message; orders the conversation list
TITLE_CHARS = 42


# ---- pages --------------------------------------------------------------------------------


def page() -> str:
    return str(st.session_state.get(PAGE) or "Chat")


def go(name: str) -> None:
    """Open a page (a callback). The profile page has no tab, so the tab bar is cleared."""
    st.session_state[PAGE] = name
    st.session_state[NAV] = name if name in PAGES else None


def on_nav() -> None:
    """The tab bar changed. Clicking the open tab deselects it, so keep that page open."""
    choice = st.session_state.get(NAV)
    if choice:
        st.session_state[PAGE] = choice
    else:  # only an open tab can be deselected, so the open page is a tab
        st.session_state[NAV] = page()


def open_article(article_id: str, category: str) -> None:
    """Show an article in Knowledge > Browse (a callback)."""
    st.session_state["kb_view"] = "Browse"
    st.session_state["kb_category"] = category
    st.session_state["kb_article"] = article_id
    go("Knowledge")


def open_glossary(term: str) -> None:
    """Show a term in Knowledge > Glossary (a callback)."""
    st.session_state["kb_view"] = "Glossary"
    st.session_state["kb_term"] = term
    go("Knowledge")


# ---- profile ------------------------------------------------------------------------------


def profile() -> UserProfile:
    value = st.session_state.get(PROFILE)
    return value if isinstance(value, UserProfile) else UserProfile()


def set_profile(value: UserProfile) -> None:
    st.session_state[PROFILE] = value
    st.session_state[ONBOARDED] = True


def onboarded() -> bool:
    return bool(st.session_state.get(ONBOARDED))


# ---- portfolio ----------------------------------------------------------------------------


def portfolio() -> list[Holding]:
    return list(st.session_state.get(PORTFOLIO) or [])


def set_portfolio(holdings: list[Holding]) -> None:
    st.session_state[PORTFOLIO] = list(holdings)


# ---- conversations ------------------------------------------------------------------------


def thread_id() -> str:
    if THREAD not in st.session_state:
        st.session_state[THREAD] = uuid.uuid4().hex
    return str(st.session_state[THREAD])


def chat() -> list[dict[str, Any]]:
    if CHAT not in st.session_state:
        st.session_state[CHAT] = []
    return list(st.session_state[CHAT])


def _store() -> dict[str, dict[str, Any]]:
    store: dict[str, dict[str, Any]] = st.session_state.setdefault(CONVERSATIONS, {})
    return store


def add_chat(role: str, content: str, output: dict[str, Any] | None = None) -> None:
    entries = chat()
    entries.append({"role": role, "content": content, "output": output})
    st.session_state[CHAT] = entries
    store = _store()
    first = next((e["content"] for e in entries if e["role"] == "user"), "New conversation")
    title = first if len(first) <= TITLE_CHARS else first[: TITLE_CHARS - 1].rstrip() + "…"
    st.session_state[ACTIVITY] = st.session_state.get(ACTIVITY, 0) + 1
    store[thread_id()] = {"title": title, "chat": entries, "order": st.session_state[ACTIVITY]}


def conversations() -> list[tuple[str, str]]:
    """(thread id, title), most recently active first."""
    store = _store()
    ranked = sorted(store.items(), key=lambda kv: kv[1]["order"], reverse=True)
    return [(cid, c["title"]) for cid, c in ranked]


def new_conversation() -> None:
    """A fresh thread (a callback). Earlier ones stay in the sidebar list."""
    st.session_state[THREAD] = uuid.uuid4().hex
    st.session_state[CHAT] = []
    st.session_state.pop(PENDING, None)
    go("Chat")


def switch_conversation(conversation_id: str) -> None:
    saved = _store()[conversation_id]  # the sidebar lists only stored conversations
    st.session_state[THREAD] = conversation_id
    st.session_state[CHAT] = list(saved["chat"])
    go("Chat")


def queue_prompt(text: str) -> None:
    """Ask a question in the chat from another page (a callback; answered on the rerun)."""
    st.session_state[PENDING] = text
    go("Chat")


def take_prompt() -> str | None:
    value = st.session_state.pop(PENDING, None)
    return str(value) if value else None
