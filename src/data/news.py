"""Tavily news search (second news provider, after yfinance)."""

from __future__ import annotations

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


class TavilyNewsProvider:
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
