"""Alpha Vantage client (primary price provider).

Free-tier quirks handled here:

- Rate-limited responses come back as HTTP 200 with a ``Note`` or ``Information`` key
  instead of data. These are raised as ``RateLimitError``, never parsed as data.
- The free tier allows about 25 requests/day and 5/minute. A local limiter and a daily
  budget stop us before Alpha Vantage refuses, so the chain moves to yfinance at once.
- ``TIME_SERIES_DAILY`` with ``outputsize=compact`` returns 100 bars; longer histories
  are left to yfinance.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import requests
from pydantic import SecretStr

from src.core.models import Freshness
from src.data.errors import ProviderError, RateLimitError, SymbolNotFoundError
from src.data.http import request_json
from src.data.models import (
    CompanyOverview,
    NewsArticle,
    PriceBar,
    PriceHistory,
    Quote,
    session_close_utc,
)
from src.data.rate_limit import DailyBudget, SlidingWindowRateLimiter
from src.utils.clock import Clock, utcnow

BASE_URL = "https://www.alphavantage.co/query"
COMPACT_BARS = 100
NAME = "alpha_vantage"


def _num(value: Any) -> float | None:
    if value in (None, "", "None", "-", "N/A"):
        return None
    try:
        return float(str(value).rstrip("%"))
    except ValueError:
        return None


def _int(value: Any) -> int | None:
    number = _num(value)
    return int(number) if number is not None else None


class AlphaVantageProvider:
    name = NAME

    def __init__(
        self,
        api_key: SecretStr,
        *,
        session: requests.Session | None = None,
        timeout_s: float = 10,
        limiter: SlidingWindowRateLimiter | None = None,
        budget: DailyBudget | None = None,
        clock: Clock = utcnow,
    ) -> None:
        self._api_key = api_key
        self._session = session or requests.Session()
        self._timeout = timeout_s
        self._limiter = limiter
        self._budget = budget
        self._clock = clock

    def _request(self, **params: str) -> dict[str, Any]:
        if self._limiter and not self._limiter.try_acquire():
            raise RateLimitError(f"{NAME}: local per-minute limit reached")
        if self._budget and not self._budget.try_consume():
            raise RateLimitError(f"{NAME}: daily request budget exhausted")
        data = request_json(
            self._session,
            "GET",
            BASE_URL,
            provider=NAME,
            timeout_s=self._timeout,
            params={**params, "apikey": self._api_key.get_secret_value()},
        )
        if not isinstance(data, dict):
            raise ProviderError(f"{NAME}: unexpected response shape")
        notice = data.get("Note") or data.get("Information")
        if notice:
            # Rate-limit notices also mention premium plans, so only the explicit
            # "premium endpoint" wording means the endpoint itself is unavailable.
            if "premium endpoint" in str(notice).lower():
                raise ProviderError(f"{NAME}: endpoint requires a premium plan")
            raise RateLimitError(f"{NAME}: rate limit notice returned")
        if "Error Message" in data:
            raise SymbolNotFoundError(f"{NAME}: {params.get('symbol', 'request')} not recognized")
        return data

    def _freshness(self, as_of: datetime) -> Freshness:
        return Freshness(source=NAME, as_of=as_of, fetched_at=self._clock())

    def get_quote(self, ticker: str) -> Quote:
        data = self._request(function="GLOBAL_QUOTE", symbol=ticker)
        raw = data.get("Global Quote") or {}
        price = _num(raw.get("05. price"))
        if not price:
            raise SymbolNotFoundError(f"{NAME}: no quote for {ticker}")
        day = raw.get("07. latest trading day")
        now = self._clock()
        as_of = session_close_utc(date.fromisoformat(day), now) if day else now
        return Quote(
            ticker=ticker,
            price=price,
            previous_close=_num(raw.get("08. previous close")),
            change=_num(raw.get("09. change")),
            change_percent=_num(raw.get("10. change percent")),
            open=_num(raw.get("02. open")),
            high=_num(raw.get("03. high")),
            low=_num(raw.get("04. low")),
            volume=_int(raw.get("06. volume")),
            freshness=self._freshness(as_of),
        )

    def get_daily_history(self, ticker: str, days: int) -> PriceHistory:
        if days > COMPACT_BARS:
            raise ProviderError(f"{NAME}: free tier returns at most {COMPACT_BARS} daily bars")
        data = self._request(function="TIME_SERIES_DAILY", symbol=ticker, outputsize="compact")
        series = data.get("Time Series (Daily)") or {}
        if not series:
            raise SymbolNotFoundError(f"{NAME}: no history for {ticker}")
        try:
            bars = [
                PriceBar(
                    date=date.fromisoformat(day),
                    open=float(row["1. open"]),
                    high=float(row["2. high"]),
                    low=float(row["3. low"]),
                    close=float(row["4. close"]),
                    volume=_int(row.get("5. volume")),
                )
                for day, row in series.items()
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(f"{NAME}: malformed time series ({type(exc).__name__})") from exc
        bars.sort(key=lambda bar: bar.date)
        bars = bars[-days:]
        as_of = session_close_utc(bars[-1].date, self._clock())
        return PriceHistory(
            ticker=ticker, bars=bars, adjusted=False, freshness=self._freshness(as_of)
        )

    def get_company_overview(self, ticker: str) -> CompanyOverview:
        data = self._request(function="OVERVIEW", symbol=ticker)
        if not data.get("Symbol") or not data.get("Name"):
            raise SymbolNotFoundError(f"{NAME}: no overview for {ticker}")
        return CompanyOverview(
            ticker=ticker,
            name=data["Name"],
            asset_type=data.get("AssetType") or None,
            sector=(data.get("Sector") or "").title() or None,
            industry=(data.get("Industry") or "").title() or None,
            description=data.get("Description") or None,
            market_cap=_num(data.get("MarketCapitalization")),
            pe_ratio=_num(data.get("PERatio")),
            dividend_yield=_num(data.get("DividendYield")),
            beta=_num(data.get("Beta")),
            week_52_high=_num(data.get("52WeekHigh")),
            week_52_low=_num(data.get("52WeekLow")),
            freshness=self._freshness(self._clock()),
        )

    def get_news(
        self, ticker: str | None = None, query: str | None = None, limit: int = 5
    ) -> list[NewsArticle]:
        """News with sentiment. Alpha Vantage only supports ticker lookups here."""
        if not ticker:
            return []
        data = self._request(function="NEWS_SENTIMENT", tickers=ticker, limit=str(limit))
        articles = []
        for item in (data.get("feed") or [])[:limit]:
            if not item.get("title"):
                continue
            articles.append(
                NewsArticle(
                    title=item["title"],
                    url=item.get("url"),
                    source=item.get("source"),
                    published_at=_parse_av_time(item.get("time_published")),
                    summary=item.get("summary"),
                    tickers=[
                        t["ticker"] for t in item.get("ticker_sentiment") or [] if "ticker" in t
                    ],
                    sentiment_score=_num(item.get("overall_sentiment_score")),
                    sentiment_label=item.get("overall_sentiment_label"),
                )
            )
        return articles


def _parse_av_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None
