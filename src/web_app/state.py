"""Typed accessors for ``st.session_state``, saved per browser.

A browser session holds several conversations (each its own workflow thread), the user's
profile and portfolio, and which page is open. Session state is the working copy; profile,
portfolio, and conversations are also written to the per-browser store (storage.py) on
every change and loaded once when a session starts, so they survive a refresh or restart.

Navigation helpers are written to be used as widget callbacks, because a widget's value
can only be set before it is drawn.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

import streamlit as st

from src.core.models import Holding, UserProfile
from src.web_app import services

logger = logging.getLogger(__name__)

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
SCROLL = "finnie_scroll_to_answer"
ACTIVITY = "finnie_activity"  # increases with every message; orders the conversation list
UNTITLED = "New conversation"
BROWSER = "finnie_browser_id"
STARTED = "finnie_started"
NEW_COOKIE = "finnie_new_cookie"  # set the cookie once, on a browser's first visit
RENAMING = "finnie_renaming"
DELETING = "finnie_deleting"
COOKIE = "finnie_id"
COOKIE_DAYS = 400  # browsers cap cookie lifetimes at about 400 days
BROWSER_ID = re.compile(r"^[0-9a-f]{32}$")


# ---- browser identity and loading -------------------------------------------------------


def _cookie_id() -> str | None:
    value = st.context.cookies.get(COOKIE)
    return value if isinstance(value, str) and BROWSER_ID.match(value) else None


def browser_id() -> str:
    return str(st.session_state[BROWSER])


def start() -> None:
    """Once per session: find this browser's ID (cookie, or a new random one) and load
    what's saved for it. Values already in the session (tests set some) win."""
    if st.session_state.get(STARTED):
        return
    found = st.session_state.get(BROWSER) or _cookie_id()
    if not found:
        found = uuid.uuid4().hex
        st.session_state[NEW_COOKIE] = True
    st.session_state[BROWSER] = found
    saved = services.store().load(found)
    if saved.profile is not None and PROFILE not in st.session_state:
        st.session_state[PROFILE] = saved.profile
        st.session_state[ONBOARDED] = True
    st.session_state.setdefault(PORTFOLIO, saved.portfolio)
    count = len(saved.conversations)
    for conversation in saved.conversations:
        if conversation.title is None:  # e.g. a refresh landed before the title was saved
            conversation.title = _recover_title(found, conversation.thread_id)
    st.session_state.setdefault(
        CONVERSATIONS,
        {
            c.thread_id: {
                "title": c.title or UNTITLED,
                "title_source": c.title_source,
                "chat": c.chat,
                "order": count - i,
            }
            for i, c in enumerate(saved.conversations)
        },
    )
    st.session_state.setdefault(ACTIVITY, count)
    current = saved.current_thread if saved.current_thread in _store() else None
    if current and THREAD not in st.session_state:
        st.session_state[THREAD] = current
        st.session_state[CHAT] = list(_store()[current]["chat"])
    st.session_state[STARTED] = True


def _recover_title(browser: str, thread: str) -> str | None:
    """The title the workflow wrote, if the page closed before it reached the store."""
    title = services.assistant().state(thread).get("title")
    if title:
        services.store().set_title(browser, thread, title, "auto")
    return title


def take_new_cookie() -> str | None:
    """The ID to store in a cookie, if this browser didn't have one yet."""
    return browser_id() if st.session_state.pop(NEW_COOKIE, False) else None


def delete_my_data() -> None:
    """Remove everything saved for this browser, then start over (a callback)."""
    threads = set(services.store().delete_browser(browser_id()))
    threads.add(thread_id())
    for thread in threads:
        services.assistant().forget(thread)
    keep = {BROWSER, STARTED}
    for key in [k for k in st.session_state if k not in keep]:
        del st.session_state[key]
    logger.info("Browser data deleted", extra={"conversations": len(threads)})


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
    services.store().save_profile(browser_id(), value)


def onboarded() -> bool:
    return bool(st.session_state.get(ONBOARDED))


# ---- portfolio ----------------------------------------------------------------------------


def portfolio() -> list[Holding]:
    return list(st.session_state.get(PORTFOLIO) or [])


def set_portfolio(holdings: list[Holding]) -> None:
    st.session_state[PORTFOLIO] = list(holdings)
    services.store().save_portfolio(browser_id(), list(holdings))


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
    saved = store.get(thread_id(), {})
    st.session_state[ACTIVITY] = st.session_state.get(ACTIVITY, 0) + 1
    store[thread_id()] = {
        "title": saved.get("title", UNTITLED),  # the model writes the real one (set_title)
        "title_source": saved.get("title_source"),
        "chat": entries,
        "order": st.session_state[ACTIVITY],
    }
    services.store().save_chat(browser_id(), thread_id(), entries)


def set_title(conversation_id: str, title: str) -> None:
    """An automatic title. It never replaces a name the user chose."""
    kept = services.store().set_title(browser_id(), conversation_id, title, "auto")
    _store()[conversation_id]["title"] = kept or title


# ---- rename and delete (sidebar callbacks) -------------------------------------------


def renaming() -> str | None:
    return st.session_state.get(RENAMING)


def deleting() -> str | None:
    return st.session_state.get(DELETING)


def start_rename(conversation_id: str) -> None:
    st.session_state.pop(DELETING, None)
    st.session_state[RENAMING] = conversation_id
    st.session_state[f"rename_{conversation_id}"] = _store()[conversation_id]["title"]


def cancel_edit() -> None:
    st.session_state.pop(RENAMING, None)
    st.session_state.pop(DELETING, None)


def rename_conversation(conversation_id: str) -> None:
    """Save the name typed in the sidebar. It overrides automatic titles for good."""
    title = " ".join(str(st.session_state.get(f"rename_{conversation_id}") or "").split())[:80]
    st.session_state.pop(RENAMING, None)
    if not title:
        return  # an empty name keeps the old one
    services.store().set_title(browser_id(), conversation_id, title, "user")
    services.assistant().lock_title(conversation_id, title)
    saved = _store()[conversation_id]
    saved["title"], saved["title_source"] = title, "user"


def ask_delete(conversation_id: str) -> None:
    st.session_state.pop(RENAMING, None)
    st.session_state[DELETING] = conversation_id


def delete_conversation(conversation_id: str) -> None:
    """Remove the conversation and its workflow memory (a callback)."""
    st.session_state.pop(DELETING, None)
    services.store().delete_conversation(browser_id(), conversation_id)
    services.assistant().forget(conversation_id)
    _store().pop(conversation_id, None)
    if st.session_state.get(THREAD) == conversation_id:
        new_conversation()


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
    services.store().set_current(browser_id(), None)
    go("Chat")


def switch_conversation(conversation_id: str) -> None:
    saved = _store()[conversation_id]  # the sidebar lists only stored conversations
    st.session_state[THREAD] = conversation_id
    st.session_state[CHAT] = list(saved["chat"])
    services.store().set_current(browser_id(), conversation_id)
    go("Chat")


def queue_prompt(text: str) -> None:
    """Ask a question in the chat from a button (a callback; answered on the rerun).

    The chat then scrolls to the start of the new answer, not the bottom of the page.
    """
    st.session_state[PENDING] = text
    st.session_state[SCROLL] = True
    go("Chat")


def take_scroll() -> bool:
    return bool(st.session_state.pop(SCROLL, False))


def peek_prompt() -> str | None:
    value = st.session_state.get(PENDING)
    return str(value) if value else None


def take_prompt() -> str | None:
    value = st.session_state.pop(PENDING, None)
    return str(value) if value else None
