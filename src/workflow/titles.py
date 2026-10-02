"""Short conversation titles for the sidebar, written by the fast model (like Claude.ai).

A title is written after the first answer, from the question and the answer. After the
third question it is rewritten once from the conversation so far, because openers are
often vague ("What is your name?") and the real topic shows up later. Then it stays fixed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.agents.base import message_text
from src.workflow.progress import specialist

logger = logging.getLogger(__name__)

FIRST_TITLE_AFTER = 1  # questions
FINAL_TITLE_AFTER = 3
MAX_WORDS = 6
MAX_CHARS = 48
PROMPT = (
    "Write a title of 2 to 6 words for this conversation with Finnie, a financial "
    "education assistant. Name the topic the user cares about, not their exact words, for "
    'example "Roth vs Traditional IRA" or "Retirement Savings Projection". Title case, no '
    "quotes, no final punctuation. Reply with the title only."
)
TOPICS = {
    "finance_qa": "Investing Basics",
    "portfolio": "Portfolio Review",
    "market": "Market Check",
    "goal_planning": "Savings Goal Plan",
    "news": "Market News",
    "tax": "Investment Taxes",
}


def clean(text: str) -> str | None:
    """One line, no quotes or trailing punctuation, at most a few words."""
    line = (text.strip().splitlines() or [""])[0]
    line = re.sub(r"^(title:\s*)", "", line, flags=re.IGNORECASE)
    line = line.strip().strip("\"'*`#").strip().rstrip(".!?:;,")
    words = line.split()
    if not words:
        return None
    title = " ".join(words[:MAX_WORDS])
    return title if len(title) <= MAX_CHARS else title[: MAX_CHARS - 1].rstrip() + "…"


def fallback_title(agents: Sequence[str]) -> str:
    """A topic label when the model can't write a title (never the user's raw message)."""
    if agents:
        return TOPICS.get(agents[0], f"{specialist(agents[0]).title()} Question")
    return "New Conversation"


def write_title(
    llm: Any, messages: Sequence[BaseMessage], summary: str | None = None
) -> str | None:
    lines = [f"Earlier: {summary}"] if summary else []
    for message in messages:
        role = "User" if isinstance(message, HumanMessage) else "Finnie"
        lines.append(f"{role}: {message_text(message)[:600]}")
    try:
        reply = llm.invoke([SystemMessage(content=PROMPT), HumanMessage(content="\n".join(lines))])
    except Exception:
        logger.exception("Writing a conversation title failed")
        return None
    return clean(message_text(reply))
