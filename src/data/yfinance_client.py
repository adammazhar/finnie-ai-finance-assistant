"""yfinance client (fallback price provider, first news provider).

yfinance scrapes Yahoo Finance, so its failures are varied. Every call goes through
``_guard``, which turns library and transport exceptions into market data errors.
The ``yfinance`` module is injectable so tests never touch the network.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

import requests

from src.core.models import Freshness
from src.data.errors import (
    MarketDataError,
    ProviderError,
    RateLimitError,
    SymbolNotFoundError,
    TransientProviderError,
)
from src.data.models import (
    CompanyOverview,
    NewsArticle,
    PriceBar,
    PriceHistory,
    Quote,
    session_close_utc,
)
from src.utils.clock import Clock, utcnow

NAME = "yfinance"
T = TypeVar("T")

_NOT_FOUND = {"YFTickerMissingError", "YFPricesMissingError", "YFInvalidPeriodError"}
_ASSET_TYPES = {"EQUITY": "Equity", "ETF": "ETF", "MUTUALFUND": "Mutual Fund", "INDEX": "Index"}


def _clean(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _drop_missing_closes(frame: Any) -> Any:
    """Rows with a close price, or ``None`` when there are none (including empty frames)."""
    if frame is None or frame.empty or "Close" not in frame.columns:
        return None
    frame = frame.dropna(subset=["Close"])
    return None if frame.empty else frame


def _translate(exc: Exception) -> MarketDataError:
    if isinstance(exc, MarketDataError):
        return exc
    name = type(exc).__name__
    if name == "YFRateLimitError":
        return RateLimitError(f"{NAME}: rate limited")
    if name in _NOT_FOUND:
        return SymbolNotFoundError(f"{NAME}: {exc}")
    if isinstance(exc, (requests.Timeout, requests.ConnectionError, TimeoutError)):
        return TransientProviderError(f"{NAME}: {name}")
    return ProviderError(f"{NAME}: {name}")


class YFinanceProvider:
    """Yahoo Finance via the ``yfinance`` library, with errors translated to market data errors."""

    name = NAME

    def __init__(self, yf_module: Any = None, clock: Clock = utcnow) -> None:
        self._yf = yf_module or importlib.import_module("yfinance")
        self._clock = clock

    def _guard(self, fn: Callable[[], T]) -> T:
        try:
            return fn()
        except Exception as exc:
            raise _translate(exc) from exc

    def _freshness(self, as_of: datetime) -> Freshness:
        return Freshness(source=NAME, as_of=as_of, fetched_at=self._clock())

    def _quote_from_frame(self, ticker: str, frame: Any) -> Quote:
        frame = _drop_missing_closes(frame)
        if frame is None:
            raise SymbolNotFoundError(f"{NAME}: no prices for {ticker}")
        last = frame.iloc[-1]
        previous = _clean(frame.iloc[-2]["Close"]) if len(frame) > 1 else None
        volume = _clean(last.get("Volume"))
        return Quote(
            ticker=ticker,
            price=float(last["Close"]),
            previous_close=previous,
            open=_clean(last.get("Open")),
            high=_clean(last.get("High")),
            low=_clean(last.get("Low")),
            volume=int(volume) if volume is not None else None,
            freshness=self._freshness(session_close_utc(frame.index[-1].date(), self._clock())),
        )

    def get_quote(self, ticker: str) -> Quote:
        """Latest unadjusted close from the last five daily bars, with the prior close.

        Raises ``SymbolNotFoundError`` when there are no prices.
        """

        def fetch() -> Quote:
            frame = self._yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
            return self._quote_from_frame(ticker, frame)

        return self._guard(fetch)

    def get_quotes(self, tickers: Iterable[str]) -> dict[str, Quote]:
        """One batched download. Tickers missing from the result are simply absent."""
        tickers = list(tickers)

        def fetch() -> dict[str, Quote]:
            frame = self._yf.download(
                tickers,
                period="5d",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
                multi_level_index=True,
            )
            quotes: dict[str, Quote] = {}
            if frame is None or frame.empty:
                return quotes
            available = set(frame.columns.get_level_values(0))
            for ticker in tickers:
                if ticker not in available:
                    continue
                try:
                    quotes[ticker] = self._quote_from_frame(ticker, frame[ticker])
                except SymbolNotFoundError:
                    continue
            return quotes

        return self._guard(fetch)

    def get_daily_history(self, ticker: str, days: int) -> PriceHistory:
        """The last ``days`` split- and dividend-adjusted daily bars.

        Raises ``SymbolNotFoundError`` when there is no history.
        """

        def fetch() -> PriceHistory:
            start = (self._clock() - timedelta(days=int(days * 1.5) + 10)).date()
            frame = self._yf.Ticker(ticker).history(
                start=start.isoformat(), interval="1d", auto_adjust=True
            )
            frame = _drop_missing_closes(frame)
            if frame is None:
                raise SymbolNotFoundError(f"{NAME}: no history for {ticker}")
            frame = frame.tail(days)
            bars = [
                PriceBar(
                    date=index.date(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]) if _clean(row.get("Volume")) is not None else None,
                )
                for index, row in frame.iterrows()
            ]
            as_of = session_close_utc(bars[-1].date, self._clock())
            return PriceHistory(
                ticker=ticker, bars=bars, adjusted=True, freshness=self._freshness(as_of)
            )

        return self._guard(fetch)

    def get_company_overview(self, ticker: str) -> CompanyOverview:
        """Profile and fundamentals from ``Ticker.info``.

        Raises ``SymbolNotFoundError`` when Yahoo has no name for the ticker.
        """

        def fetch() -> CompanyOverview:
            info = self._yf.Ticker(ticker).info or {}
            name = info.get("longName") or info.get("shortName")
            if not name:
                raise SymbolNotFoundError(f"{NAME}: no overview for {ticker}")
            quote_type = info.get("quoteType")
            return CompanyOverview(
                ticker=ticker,
                name=name,
                asset_type=_ASSET_TYPES.get(str(quote_type), quote_type),
                sector=info.get("sector"),
                industry=info.get("industry"),
                description=info.get("longBusinessSummary"),
                market_cap=_clean(info.get("marketCap")),
                pe_ratio=_clean(info.get("trailingPE")),
                # ETFs report distribution yield as "yield"; both are fractions
                dividend_yield=_clean(info.get("trailingAnnualDividendYield", info.get("yield"))),
                beta=_clean(info.get("beta")),
                week_52_high=_clean(info.get("fiftyTwoWeekHigh")),
                week_52_low=_clean(info.get("fiftyTwoWeekLow")),
                freshness=self._freshness(self._clock()),
            )

        return self._guard(fetch)

    def get_news(
        self, ticker: str | None = None, query: str | None = None, limit: int = 5
    ) -> list[NewsArticle]:
        """Yahoo news for ``ticker``, else a Yahoo search for ``query``; ``[]`` with neither."""

        def fetch() -> list[NewsArticle]:
            if ticker:
                items = self._yf.Ticker(ticker).get_news(count=limit)
            elif query:
                items = self._yf.Search(query, news_count=limit, max_results=1).news
            else:
                return []
            parsed = (parse_yahoo_news(item, ticker) for item in items or [])
            return [article for article in parsed if article][:limit]

        return self._guard(fetch)


def parse_yahoo_news(item: dict[str, Any], ticker: str | None = None) -> NewsArticle | None:
    """Parse both Yahoo news shapes: the current nested ``content`` form and the legacy flat one."""
    content = item.get("content")
    if isinstance(content, dict):
        title = content.get("title")
        url = (content.get("canonicalUrl") or {}).get("url") or (
            content.get("clickThroughUrl") or {}
        ).get("url")
        source = (content.get("provider") or {}).get("displayName")
        published = _parse_iso(content.get("pubDate"))
        summary = content.get("summary")
        tickers = [ticker] if ticker else []
    else:
        title = item.get("title")
        url = item.get("link")
        source = item.get("publisher")
        stamp = item.get("providerPublishTime")
        published = datetime.fromtimestamp(stamp, UTC) if isinstance(stamp, int | float) else None
        summary = item.get("summary")
        tickers = list(item.get("relatedTickers") or ([ticker] if ticker else []))
    if not title:
        return None
    return NewsArticle(
        title=title,
        url=url,
        source=source,
        published_at=published,
        summary=summary,
        tickers=tickers,
    )


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
