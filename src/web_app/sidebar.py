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
            st.button(
                md(title),
                key=f"conversation_{conversation_id}",
                on_click=state.switch_conversation,
                args=(conversation_id,),
                type="secondary" if conversation_id == current else "tertiary",
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
