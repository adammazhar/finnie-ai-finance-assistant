"""A deterministic stand-in for MarketDataService used by agent and tool tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.core.models import Freshness
from src.data.errors import DataUnavailableError, SymbolNotFoundError
from src.data.models import (
    BatchQuotes,
    CompanyOverview,
    NewsArticle,
    NewsFeed,
    PriceBar,
    PriceHistory,
    Quote,
)
from src.data.service import ProviderStatus

NOW = datetime(2026, 9, 30, 18, 0, tzinfo=UTC)

PRICES = {
    "VTI": 300.0,
    "BND": 70.0,
    "AAPL": 230.0,
    "TSLA": 250.0,
    "SPY": 600.0,
    "QQQ": 520.0,
    "DIA": 450.0,
    "IWM": 230.0,
    "XLK": 240.0,
    "XLF": 50.0,
    "XLV": 150.0,
    "XLE": 90.0,
    "XLY": 200.0,
    "XLP": 80.0,
    "XLI": 140.0,
    "XLB": 90.0,
    "XLU": 80.0,
    "XLRE": 42.0,
    "XLC": 100.0,
}


def fresh(source: str = "yfinance", *, mock: bool = False, stale: bool = False) -> Freshness:
    return Freshness(
        source=source,
        as_of=NOW,
        fetched_at=NOW,
        is_mock=mock,  # type: ignore[arg-type]
        is_stale=stale,
    )


class FakeMarketService:
    def __init__(
        self,
        *,
        mock: bool = False,
        news: list[NewsArticle] | None = None,
        fail: set[str] | None = None,
    ) -> None:
        self.mock = mock
        self.fail = fail or set()
        self.news = (
            news
            if news is not None
            else [
                NewsArticle(
                    title="Stocks rise on rate hopes",
                    url="https://www.sec.gov/news/1",
                    source="Example Wire",
                    published_at=NOW - timedelta(hours=3),
                    summary="Major indexes rose as investors weighed interest rates.",
                ),
            ]
        )
        self.calls: list[tuple[str, tuple]] = []

    def _check(self, name: str, *args: object) -> None:
        self.calls.append((name, args))
        if name in self.fail:
            raise DataUnavailableError(f"{name} unavailable")

    def get_quotes(self, tickers):
        tickers = [t.upper() for t in tickers]
        self._check("get_quotes", tickers)
        quotes = {
            t: Quote(
                ticker=t,
                price=PRICES[t],
                previous_close=PRICES[t] * 0.99,
                freshness=fresh(mock=self.mock),
            )
            for t in tickers
            if t in PRICES
        }
        return BatchQuotes(
            quotes=quotes, errors={t: "not found" for t in tickers if t not in PRICES}
        )

    def get_daily_history(self, ticker, days=252):
        self._check("get_daily_history", ticker, days)
        ticker = ticker.upper()
        if ticker not in PRICES:
            raise SymbolNotFoundError(ticker)
        start = NOW.date() - timedelta(days=days * 2)
        bars = []
        day = start
        i = 0
        while len(bars) < days:
            if day.weekday() < 5:
                close = PRICES[ticker] * (0.8 + 0.2 * i / days) * (1 + 0.01 * ((i % 7) - 3) / 3)
                bars.append(
                    PriceBar(
                        date=day,
                        open=close,
                        high=close * 1.01,
                        low=close * 0.99,
                        close=round(close, 2),
                    )
                )
                i += 1
            day += timedelta(days=1)
        return PriceHistory(
            ticker=ticker, bars=bars, adjusted=True, freshness=fresh(mock=self.mock)
        )

    def get_company_overview(self, ticker):
        self._check("get_company_overview", ticker)
        if ticker.upper() not in PRICES:
            raise SymbolNotFoundError(ticker)
        return CompanyOverview(
            ticker=ticker.upper(),
            name=f"{ticker.upper()} Inc.",
            asset_type="Equity",
            sector="Technology",
            market_cap=3.4e12,
            pe_ratio=33.2,
            dividend_yield=0.0044,
            beta=1.2,
            freshness=fresh(),
        )

    def get_treasury_bill_yield(self):
        self._check("get_treasury_bill_yield")
        return Quote(ticker="^IRX", price=4.03, freshness=fresh())

    def provider_status(self):
        return [ProviderStatus(name="yfinance", enabled=True, detail="fake")]

    def get_news(self, ticker=None, query=None, limit=5):
        self._check("get_news", ticker, query, limit)
        return NewsFeed(
            query=ticker or query or "", articles=self.news[:limit], freshness=fresh(mock=self.mock)
        )
