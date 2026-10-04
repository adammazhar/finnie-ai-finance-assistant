"""Scoped CSS for what the theme in .streamlit/config.toml can't set.

Selectors target keyed elements (Streamlit adds an ``st-key-<key>`` class) or stable
``data-testid`` attributes, so the styles don't leak onto unrelated widgets.

Contrast rules (WCAG AA, 4.5:1 for text):
- Colors come from a palette for the active theme (``st.context.theme``): the configured
  light theme, or Streamlit's dark theme if the config isn't loaded (for example when
  the app is started from another folder). Every text color here was checked against
  the backgrounds it sits on; the ratios are in the comments.
- Any rule that sets a text color on a component also sets that component's background,
  so a component stays readable even if the theme is misdetected for a moment.
- Green is an accent for lines and bars only: as text it reaches just 3:1 on white.
"""

from __future__ import annotations

from typing import Literal

import streamlit as st

CHAT_WIDTH = "760px"

PALETTES: dict[str, dict[str, str]] = {
    "light": {
        "page": "#FFFFFF",
        "text": "#1B2433",  # 15.6:1 on white, 12.9:1 on the selected row
        "muted": "#4A5A6E",  # 7.1:1 on white, 6.2:1 on the sidebar hover, 5.8:1 selected
        "heading": "#0B2545",  # 15.4:1 on white
        "line": "#D5DCE6",
        "subtle": "#F4F6FA",
        "sidebar": "#F7F9FC",
        "hover": "#EDF1F7",
        "bubble": "#F1F4F9",
        "selected": "#E3EAF4",
        "selected_text": "#0B2545",  # 12.7:1
        "accent": "#12A26F",  # lines only
        "primary": "#0B2545",  # white text on it: 15.4:1
        "primary_hover": "#163B66",  # 11.3:1
        "on_primary": "#FFFFFF",
        "shadow": "rgba(16, 24, 40, 0.06)",
    },
    "dark": {
        "page": "#0E1117",
        "text": "#FAFAFA",  # 18.1:1 on the page, 11.0:1 on the selected row
        "muted": "#B6BFCC",  # 10.2:1 on the page, 6.2:1 on the selected row
        "heading": "#FAFAFA",
        "line": "#3A3F4B",
        "subtle": "#1A1D24",
        "sidebar": "#262730",
        "hover": "#30333F",
        "bubble": "#1E2533",
        "selected": "#1F3A5F",
        "selected_text": "#FAFAFA",
        "accent": "#5FD3A0",
        "primary": "#2563A8",  # white text on it: 6.1:1 (Streamlit's default red is 3.3:1)
        "primary_hover": "#1F5A99",  # 7.1:1
        "on_primary": "#FFFFFF",
        "shadow": "rgba(0, 0, 0, 0.4)",
    },
}


def theme_type() -> Literal["light", "dark"]:
    """The active theme. Streamlit infers it from the background; it can lag on a
    session's very first load, which the self-contained component colors cover."""
    detected = getattr(getattr(st.context, "theme", None), "type", None)
    return "dark" if detected == "dark" else "light"


def base_css(c: dict[str, str]) -> str:
    """CSS for every page in the palette ``c``: layout, text colors, buttons, and sidebar."""
    return f"""
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ color: {c["heading"]}; letter-spacing: -0.01em; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: {c["muted"]}; }}
input::placeholder, textarea::placeholder {{ color: {c["muted"]} !important; opacity: 1; }}
[data-testid="stSliderTickBar"], [data-testid="stSliderTickBar"] * {{ color: {c["muted"]}; }}
[data-testid="stSliderThumbValue"] {{ color: {c["heading"]}; }}

/* icon-only buttons (feedback thumbs, sidebar collapse/expand): full text color, not faded */
[class*="st-key-feedback_"] button, [data-testid="stSidebarHeader"] button,
[data-testid="stSidebarHeader"] [data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] button {{
  color: {c["text"]} !important;
}}

/* primary buttons: palette colors in both themes */
button[kind="primary"], button[data-testid="stBaseButton-primary"] {{
  background: {c["primary"]} !important; border-color: {c["primary"]} !important;
  color: {c["on_primary"]} !important;
}}
button[kind="primary"]:hover, button[kind="primary"]:focus-visible,
button[data-testid="stBaseButton-primary"]:hover {{
  background: {c["primary_hover"]} !important; border-color: {c["primary_hover"]} !important;
  color: {c["on_primary"]} !important;
}}
button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p {{
  color: {c["on_primary"]} !important;
}}

/* rows of metric cards wrap instead of truncating values (e.g. 4 index cards at tablet
   width with the sidebar open become 2 x 2) */
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) {{
  flex-wrap: wrap;
}}
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"])
  > [data-testid="stColumn"] {{
  flex: 1 1 11rem !important; min-width: 11rem; width: auto !important;
}}

/* metric cards */
[data-testid="stMetric"] {{
  background: {c["page"]}; border: 1px solid {c["line"]}; border-radius: 12px;
  padding: 14px 16px; box-shadow: 0 1px 2px {c["shadow"]};
}}
[data-testid="stMetricLabel"] p {{ color: {c["muted"]}; font-size: 0.85rem; font-weight: 500; }}
[data-testid="stMetricValue"] {{ color: {c["heading"]}; }}

/* tab bar: a segmented control styled as underlined tabs, pinned while scrolling */
.st-key-nav {{
  position: sticky; top: 3.75rem; z-index: 999; width: 100% !important;
  background: {c["page"]}; border-bottom: 1px solid {c["line"]}; margin-bottom: 0.6rem;
}}
/* the header strip above the tab bar is transparent by default; content scrolling
   under it would show above the pinned tabs */
[data-testid="stHeader"] {{ background: {c["page"]}; }}
[id^="finnie-answer-"] {{ scroll-margin-top: 7.5rem; }}
.st-key-nav [data-testid="stButtonGroup"] {{ gap: 0.25rem; flex-wrap: wrap; }}
.st-key-nav button {{
  background: {c["page"]} !important; border: none !important; border-radius: 0 !important;
  border-bottom: 3px solid transparent !important; box-shadow: none !important;
  color: {c["muted"]} !important; font-weight: 600; padding: 0.55rem 1rem !important;
}}
.st-key-nav button:hover, .st-key-nav button:focus-visible {{
  color: {c["heading"]} !important; background: {c["hover"]} !important;
}}
.st-key-nav button[aria-checked="true"] {{
  color: {c["heading"]} !important; border-bottom-color: {c["accent"]} !important;
}}

/* sidebar: conversation rows, their "..." menus, and the bottom profile/status area.
   Row styles are scoped to the row buttons (key "conversation_<id>"), so the Save, Delete,
   and Cancel buttons of an inline rename or delete keep the navy primary style. */
.st-key-brand h2 {{ margin: 0; font-size: 1.45rem; }}
.st-key-conversations [data-testid="stHorizontalBlock"] {{ gap: 0.15rem; }}
.st-key-conversations [class*="st-key-conversation_"] button,
.st-key-conversations [data-testid="stPopoverButton"],
.st-key-sidebar_bottom [data-testid="stPopoverButton"] {{
  background: {c["sidebar"]} !important; color: {c["text"]} !important;
  border: none !important; box-shadow: none !important;
}}
.st-key-conversations [class*="st-key-conversation_"] button {{
  justify-content: flex-start; text-align: left;
  font-weight: 400; padding: 0.35rem 0.6rem; min-height: 0;
}}
.st-key-conversations [data-testid="stPopoverButton"] {{ padding: 0.2rem 0.35rem; }}
.st-key-conversations [class*="st-key-conversation_"] button > div {{
  justify-content: flex-start; width: 100%;
}}
.st-key-conversations [class*="st-key-conversation_"] button p {{
  text-align: left; color: inherit;
}}
.st-key-conversations [class*="st-key-conversation_"] button:hover,
.st-key-conversations [class*="st-key-conversation_"] button:focus-visible,
.st-key-conversations [data-testid="stPopoverButton"]:hover,
.st-key-sidebar_bottom [data-testid="stPopoverButton"]:hover {{
  background: {c["hover"]} !important; color: {c["text"]} !important;
}}
.st-key-conversations [class*="st-key-conversation_"] button[kind="secondary"] {{
  background: {c["selected"]} !important; color: {c["selected_text"]} !important;
  font-weight: 600; box-shadow: inset 3px 0 0 {c["accent"]} !important;
}}
.st-key-conversations [class*="st-key-conversation_"] button[kind="secondary"] p {{
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.st-key-sidebar_bottom button {{ justify-content: flex-start; }}
.st-key-sidebar_bottom button > div {{ justify-content: flex-start; width: 100%; }}
.st-key-sidebar_bottom [data-testid="stPopoverButton"] {{ font-size: 0.85rem; }}
.st-key-sidebar_footer p {{ font-size: 0.72rem; line-height: 1.35; color: {c["muted"]}; }}

/* popover menus (rendered outside the sidebar): the menu and its items */
[data-testid="stPopoverBody"] {{ background: {c["page"]} !important; color: {c["text"]}; }}
[data-testid="stPopoverBody"] button {{
  background: {c["page"]} !important; color: {c["text"]} !important;
}}
[data-testid="stPopoverBody"] button:hover, [data-testid="stPopoverBody"] button:focus-visible {{
  background: {c["hover"]} !important; color: {c["text"]} !important;
}}
"""


def chat_css(c: dict[str, str]) -> str:
    """Extra CSS for the Chat page: a centered reading column and user message bubbles."""
    return f"""
/* Claude.ai-style chat: one centered reading column, input pinned at the bottom */
[data-testid="stChatMessage"] {{
  max-width: {CHAT_WIDTH}; margin-left: auto; margin-right: auto;
  background: transparent; padding: 0.4rem 0;
}}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {{ display: none; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
  background: {c["bubble"]}; color: {c["text"]}; border-radius: 18px; padding: 0.55rem 1rem;
  max-width: calc({CHAT_WIDTH} * 0.8); margin-right: max(0px, calc((100% - {CHAT_WIDTH}) / 2));
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {{
  color: {c["text"]};
}}
/* headings inside answers stay modest: a model's "# Title" shouldn't outsize the page */
[data-testid="stChatMessage"] h1 {{ font-size: 1.3rem; }}
[data-testid="stChatMessage"] h2 {{ font-size: 1.15rem; }}
[data-testid="stChatMessage"] h3, [data-testid="stChatMessage"] h4 {{ font-size: 1.05rem; }}
.st-key-chat_intro, .st-key-starters {{ max-width: {CHAT_WIDTH}; margin: 0 auto; }}
[data-testid="stBottomBlockContainer"] {{ max-width: calc({CHAT_WIDTH} + 2rem); margin: 0 auto; }}
[data-testid="stChatInput"] {{
  border-radius: 24px !important; border: 1px solid {c["line"]} !important;
  box-shadow: 0 2px 10px {c["shadow"]};
}}
[data-testid="stChatInput"]:focus-within {{ border-color: {c["primary"]} !important; }}
[data-testid="stChatInput"] textarea {{ font-size: 1rem; }}
"""


def apply(page: str) -> None:
    """Inject the CSS for ``page`` in the active theme's palette."""
    colors = PALETTES[theme_type()]
    css = base_css(colors) + (chat_css(colors) if page == "Chat" else "")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
