"""Scoped CSS for what the theme in .streamlit/config.toml can't set.

Selectors target keyed elements (Streamlit adds an ``st-key-<key>`` class) or stable
``data-testid`` attributes, so the styles don't leak onto unrelated widgets.
"""

from __future__ import annotations

import streamlit as st

NAVY = "#0B2545"
GREEN = "#12A26F"
MUTED = "#5B6B7F"
LINE = "#E3E8EF"
CHAT_WIDTH = "760px"

BASE = f"""
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ color: {NAVY}; letter-spacing: -0.01em; }}
[data-testid="stCaptionContainer"] {{ color: {MUTED}; }}

/* metric cards */
[data-testid="stMetric"] {{
  background: #FFFFFF; border: 1px solid {LINE}; border-radius: 12px;
  padding: 14px 16px; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.85rem; font-weight: 500; }}
[data-testid="stMetricValue"] {{ color: {NAVY}; }}

/* tab bar: a segmented control styled as underlined tabs */
.st-key-nav {{ border-bottom: 1px solid {LINE}; margin-bottom: 0.6rem; }}
.st-key-nav [data-testid="stButtonGroup"] {{ gap: 0.25rem; flex-wrap: wrap; }}
.st-key-nav button {{
  background: transparent !important; border: none !important; border-radius: 0 !important;
  border-bottom: 3px solid transparent !important; box-shadow: none !important;
  color: {MUTED} !important; font-weight: 600; padding: 0.55rem 1rem !important;
}}
.st-key-nav button:hover {{ color: {NAVY} !important; }}
.st-key-nav button[aria-checked="true"] {{
  color: {NAVY} !important; border-bottom-color: {GREEN} !important;
}}

/* sidebar */
.st-key-brand h2 {{ margin: 0; font-size: 1.45rem; }}
.st-key-conversations button {{
  justify-content: flex-start; text-align: left; border: none;
  color: #1B2433; font-weight: 400; padding: 0.35rem 0.6rem; min-height: 0;
}}
.st-key-conversations button > div {{ justify-content: flex-start; width: 100%; }}
.st-key-conversations button p {{ text-align: left; }}
.st-key-sidebar_bottom button {{ justify-content: flex-start; }}
.st-key-sidebar_bottom button > div {{ justify-content: flex-start; width: 100%; }}
.st-key-sidebar_bottom [data-testid="stPopoverButton"] {{
  border: none; background: transparent; color: {MUTED}; font-size: 0.85rem;
  justify-content: flex-start;
}}
.st-key-conversations button:hover {{ background: #EDF1F7; }}
.st-key-conversations button[kind="secondary"] p {{
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.st-key-sidebar_footer p {{ font-size: 0.72rem; line-height: 1.35; color: {MUTED}; }}
"""

CHAT = f"""
/* Claude.ai-style chat: one centered reading column, input pinned at the bottom */
[data-testid="stChatMessage"] {{
  max-width: {CHAT_WIDTH}; margin-left: auto; margin-right: auto;
  background: transparent; padding: 0.4rem 0;
}}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {{ display: none; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
  background: #F1F4F9; border-radius: 18px; padding: 0.55rem 1rem;
  max-width: calc({CHAT_WIDTH} * 0.8); margin-right: max(0px, calc((100% - {CHAT_WIDTH}) / 2));
}}
.st-key-chat_intro, .st-key-starters {{ max-width: {CHAT_WIDTH}; margin: 0 auto; }}
[data-testid="stBottomBlockContainer"] {{ max-width: calc({CHAT_WIDTH} + 2rem); margin: 0 auto; }}
[data-testid="stChatInput"] {{
  border-radius: 24px !important; border: 1px solid #D5DCE6 !important;
  box-shadow: 0 2px 10px rgba(16, 24, 40, 0.07);
}}
[data-testid="stChatInput"] textarea {{ font-size: 1rem; }}
"""


def apply(page: str) -> None:
    css = BASE + (CHAT if page == "Chat" else "")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
