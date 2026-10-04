"""Small text helpers shared by the pages (pure, no Streamlit)."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator, Sequence
from urllib.parse import urlparse

from src.core.models import Freshness
from src.workflow.progress import specialist

STREAM_DELAY_S = 0.012  # pause between words when streaming an answer into the chat
SNIPPET_CHARS = 320
SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
PROVIDERS = {
    "alpha_vantage": "Alpha Vantage",
    "yfinance": "Yahoo Finance",
    "tavily": "Tavily",
    "mock": "demo data",
}


def md(text: str | None) -> str:
    """Escape dollar signs so Streamlit's markdown doesn't render "$5 ... $10" as math.

    Use on every piece of text rendered as markdown: answers, article bodies, snippets,
    titles, and captions. Already-escaped dollars stay as they are.
    """
    return (text or "").replace("\\$", "$").replace("$", "\\$")


def money(value: float | None, cents: bool = False) -> str:
    """Dollars with thousands separators, whole by default or with cents; "n/a" for None."""
    if value is None:
        return "n/a"
    return f"${value:,.2f}" if cents else f"${value:,.0f}"


def big_money(value: float | None) -> str:
    """Market caps and other large amounts: $2.9T, $415.2B, $87.0M."""
    if value is None:
        return "n/a"
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= size:
            return f"${value / size:,.1f}{suffix}"
    return money(value)


def percent(value: float | None, digits: int = 1, signed: bool = False) -> str:
    """A fraction as a percentage (0.123 -> "12.3%"), optionally signed; "n/a" for None."""
    if value is None:
        return "n/a"
    return f"{value:+.{digits}%}" if signed else f"{value:.{digits}%}"


def freshness_caption(freshness: Sequence[Freshness]) -> str | None:
    """One caption per kind of data: "Market data: Live · 3 min ago · News fetched ..."."""
    parts = []
    for kind, label in (("market", "Market data"), ("news", None)):
        items = [f for f in freshness if f.kind == kind]
        if not items:
            continue
        if any(f.is_mock for f in items):
            parts.append(f"{label or 'News'}: demo data (live feed unavailable)")
            continue
        oldest = min(items, key=lambda f: f.fetched_at)
        parts.append(f"{label}: {oldest.label()}" if label else oldest.label())
    return " · ".join(parts) or None


def provider_name(freshness: Freshness) -> str:
    """Display name of the provider behind a piece of data, e.g. "Yahoo Finance"."""
    source = freshness.origin or freshness.source
    return PROVIDERS.get(source, source.replace("_", " ").title())


def domain(url: str | None) -> str | None:
    """The host of a URL without ``www.``, for source labels; None if there is none."""
    if not url:
        return None
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.") or None


def first_sentence(text: str | None, limit: int = 300) -> str | None:
    """The first sentence of ``text``, cut at a word with "…" if longer than ``limit``."""
    if not text or not text.strip():
        return None
    match = SENTENCE_END.search(text.strip())
    sentence = text.strip()[: match.end()] if match else text.strip()
    return sentence if len(sentence) <= limit else sentence[:limit].rsplit(" ", 1)[0] + "…"


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Shorten to whole sentences within ``limit`` characters (one sentence at least).

    Markdown list items become "•"-separated, so they don't run together as one item.
    """
    text = re.sub(r"(?m)^\s*(?:[-*]|\d+\.)\s+", "• ", text)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    ends = [m.end() for m in SENTENCE_END.finditer(text) if m.end() <= limit]
    if ends:
        return text[: ends[-1]]
    return first_sentence(text, limit) or text[:limit]


def match_label(score: float) -> str:
    """Plain-language relevance for a search result (scores are cosine similarities)."""
    if score >= 0.6:
        return "Strong match"
    if score >= 0.5:
        return "Good match"
    return "Related"


def agent_badges(agents: Sequence[str]) -> str:
    """Markdown badges naming the specialists that worked on an answer."""
    return " ".join(f":blue-badge[{specialist(a).capitalize()}]" for a in agents)


def stream_words(
    text: str, delay: float = STREAM_DELAY_S, sleep: Callable[[float], None] = time.sleep
) -> Iterator[str]:
    """Yield an answer a few words at a time, for ``st.write_stream``.

    The answer streams after the output guardrail has approved it, so nothing is shown
    that the guardrail might still rewrite.
    """
    words = text.split(" ")
    for i in range(0, len(words), 3):
        yield " ".join(words[i : i + 3]) + (" " if i + 3 < len(words) else "")
        if delay:
            sleep(delay)
