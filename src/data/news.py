"""Tavily news search (second news provider, after yfinance)."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

import requests
from pydantic import SecretStr

from src.data.errors import ProviderError
from src.data.http import request_json
from src.data.models import NewsArticle

TAVILY_URL = "https://api.tavily.com/search"
NAME = "tavily"
SUMMARY_CHARS = 500


# Common English function words, plus headline vocabulary (headlines often drop "the").
ENGLISH_WORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "as",
        "at",
        "by",
        "from",
        "is",
        "are",
        "was",
        "be",
        "its",
        "it",
        "this",
        "that",
        "after",
        "over",
        "into",
        "than",
        "up",
        "down",
        "new",
        "says",
        "said",
        "will",
        "has",
        "have",
        "how",
        "why",
        "what",
        "stock",
        "stocks",
        "shares",
        "market",
        "markets",
        "rates",
        "rally",
        "rise",
        "rises",
        "falls",
        "fall",
        "earnings",
        "price",
        "prices",
        "holds",
        "sales",
        "investors",
        "report",
        "beats",
        "misses",
        "deal",
        "growth",
    ]
)


def is_english(text: str) -> bool:
    """A cheap check that a headline (plus summary) is in English.

    Most letters must be plain ASCII, and a text of four or more words must contain at
    least one common English word. Good enough to drop the German, Spanish, or Japanese
    articles some feeds mix in, without a language-detection dependency.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) < 0.9:
        return False
    words = re.findall(r"[a-z']+", text.lower())
    return len(words) < 4 or any(w in ENGLISH_WORDS for w in words)


# Search results that are price pages rather than articles, e.g. Yahoo's
# "SPY Sep 2026 665.000 call (SPY260929C00665000) Stock Price, News, Quote & History"
OPTION_SYMBOL = re.compile(r"\b[A-Z]{1,6}\d{6}[CP]\d{8}\b")
QUOTE_PAGE_TITLE = re.compile(r"stock price, news, quote|quote & history|quote and history", re.I)
QUOTE_PAGE_PATH = re.compile(r"/quote/|/options?/", re.I)


def is_article(title: str, url: str | None) -> bool:
    """False for quote, option-chain, and other price pages that news searches sometimes
    return; True for anything that looks like an actual article."""
    if OPTION_SYMBOL.search(title) or QUOTE_PAGE_TITLE.search(title):
        return False
    return not (url and QUOTE_PAGE_PATH.search(url))


class TavilyNewsProvider:
    """News search through Tavily's ``news`` topic, limited to the last ``days`` days."""

    name = NAME

    def __init__(
        self,
        api_key: SecretStr,
        *,
        session: requests.Session | None = None,
        timeout_s: float = 10,
        days: int = 7,
    ) -> None:
        self._api_key = api_key
        self._session = session or requests.Session()
        self._timeout = timeout_s
        self._days = days

    def get_news(
        self, ticker: str | None = None, query: str | None = None, limit: int = 5
    ) -> list[NewsArticle]:
        """Search Tavily for ``query``, or "<ticker> stock news"; ``[]`` when given neither.

        Summaries are cut to 500 characters. Raises ``ProviderError`` on an unexpected response
        shape.
        """
        search = query or (f"{ticker} stock news" if ticker else None)
        if not search:
            return []
        data = request_json(
            self._session,
            "POST",
            TAVILY_URL,
            provider=NAME,
            timeout_s=self._timeout,
            json={"query": search, "topic": "news", "max_results": limit, "days": self._days},
            headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
        )
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise ProviderError(f"{NAME}: unexpected response shape")
        articles = []
        for item in data["results"][:limit]:
            if not item.get("title"):
                continue
            url = item.get("url")
            articles.append(
                NewsArticle(
                    title=item["title"],
                    url=url,
                    source=urlparse(url).netloc.removeprefix("www.") if url else None,
                    published_at=parse_published(item.get("published_date")),
                    summary=(item.get("content") or "")[:SUMMARY_CHARS] or None,
                    tickers=[ticker] if ticker else [],
                )
            )
        return articles


def parse_published(value: Any) -> datetime | None:
    """Tavily dates arrive as RFC 2822 (``Mon, 29 Sep 2026 14:00:00 GMT``) or ISO 8601."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
