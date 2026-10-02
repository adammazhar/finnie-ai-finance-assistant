"""Sidebar, modeled on Claude.ai's: name, New conversation, recent conversations, and a
profile button with a status icon at the bottom. The disclaimer is a small footer."""

from __future__ import annotations

import streamlit as st

from src.core.llm import describe_llm
from src.web_app import profile_page, services, state
from src.web_app.formatting import md

RECENT_SHOWN = 12


def _status() -> tuple[str, list[str]]:
    settings = services.context().settings
    info = describe_llm(settings)
    model = f"{info.provider} · {info.main_model}" if info.primary_ready else "not configured"
    backup = (
        f"{info.fallback_provider} ({'ready' if info.fallback_ready else 'no key'})"
        if info.fallback_provider
        else "none"
    )
    lines = [f"**AI model:** {model}", f"**Backup model:** {backup}"]
    providers = services.provider_status()
    for provider in providers:
        lines.append(
            f"**{provider.name}:** {'on' if provider.enabled else 'off'} ({md(provider.detail)})"
        )
    healthy = info.primary_ready and all(p.enabled for p in providers if p.name == "yfinance")
    return ("🟢" if healthy else "🟠"), lines


def _conversations() -> None:
    recent = state.conversations()
    if not recent:
        return
    st.caption("Recent conversations")
    current = state.thread_id()
    with st.container(key="conversations"):
        for conversation_id, title in recent[:RECENT_SHOWN]:
            if state.renaming() == conversation_id:
                _rename_row(conversation_id)
            elif state.deleting() == conversation_id:
                _delete_row(conversation_id, title)
            else:
                _row(conversation_id, title, conversation_id == current)


def _row(conversation_id: str, title: str, current: bool) -> None:
    name, menu = st.columns([6, 1], gap="small", vertical_alignment="center")
    name.button(
        md(title),
        key=f"conversation_{conversation_id}",
        on_click=state.switch_conversation,
        args=(conversation_id,),
        type="secondary" if current else "tertiary",
        width="stretch",
    )
    with menu.popover("", icon=":material/more_horiz:", help="Rename or delete"):
        st.button(
            "Rename",
            icon=":material/edit:",
            key=f"rename_start_{conversation_id}",
            on_click=state.start_rename,
            args=(conversation_id,),
            type="tertiary",
        )
        st.button(
            "Delete",
            icon=":material/delete:",
            key=f"delete_start_{conversation_id}",
            on_click=state.ask_delete,
            args=(conversation_id,),
            type="tertiary",
        )


def _rename_row(conversation_id: str) -> None:
    """Edit the name in place; Enter or Save keeps it, Cancel or an empty name doesn't."""
    st.text_input(
        "Conversation name",
        key=f"rename_{conversation_id}",
        max_chars=80,
        label_visibility="collapsed",
        on_change=state.rename_conversation,
        args=(conversation_id,),
    )
    save, cancel = st.columns(2)
    save.button(
        "Save",
        key=f"rename_save_{conversation_id}",
        on_click=state.rename_conversation,
        args=(conversation_id,),
        type="primary",
        width="stretch",
    )
    cancel.button(
        "Cancel",
        key=f"rename_cancel_{conversation_id}",
        on_click=state.cancel_edit,
        width="stretch",
    )


def _delete_row(conversation_id: str, title: str) -> None:
    with st.container(border=True):
        st.markdown(md(f"Delete **{title}**? This can't be undone."))
        confirm, cancel = st.columns(2)
        confirm.button(
            "Delete",
            key=f"delete_confirm_{conversation_id}",
            on_click=state.delete_conversation,
            args=(conversation_id,),
            type="primary",
            width="stretch",
        )
        cancel.button(
            "Cancel",
            key=f"delete_cancel_{conversation_id}",
            on_click=state.cancel_edit,
            width="stretch",
        )


def render() -> None:
    with st.sidebar:
        with st.container(key="brand"):
            st.markdown("## Finnie")
            st.caption("Your financial education assistant")
        st.button(
            "New conversation",
            icon=":material/add:",
            type="primary",
            on_click=state.new_conversation,
            key="new_conversation",
            width="stretch",
        )
        _conversations()
        st.divider()
        you = state.profile()
        with st.container(key="sidebar_bottom"):
            st.button(
                f"Profile: {profile_page.LEVEL_LABELS[you.knowledge_level]}, "
                f"{profile_page.RISK_LABELS[you.risk_tolerance].lower()}",
                icon=":material/account_circle:",
                on_click=profile_page.open_profile,
                key="open_profile",
                help="Edit your profile",
                width="stretch",
            )
            icon, lines = _status()
            with st.popover(
                f"{icon} System status", help="Models and market data", width="stretch"
            ):
                st.markdown("  \n".join(lines))
        with st.container(key="sidebar_footer"):
            st.caption(md(services.context().settings.app.disclaimer))
