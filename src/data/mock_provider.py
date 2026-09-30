"""Demo data used only when every live provider (and the cache) has failed.

Values come from ``data/reference/mock_market.json`` and are always flagged ``is_mock``,
so the UI shows a "Demo data" badge. Only tickers in that file are served: an unknown
ticker raises ``SymbolNotFoundError`` rather than returning invented prices.
Price history is a deterministic random walk ending at the demo price.
"""

from __future__ import annotations

import json
import math
import random
import zlib
from datetime import date, timedelta
from functools import cached_property
from pathlib import Path
from typing import Any

from src.core.config import PROJECT_ROOT
from src.core.models import Freshness
from src.data.errors import SymbolNotFoundError
from src.data.models import CompanyOverview, NewsArticle, PriceBar, PriceHistory, Quote
from src.utils.clock import Clock, utcnow

DEFAULT_MOCK_PATH = PROJECT_ROOT / "data" / "reference" / "mock_market.json"
TRADING_DAYS_PER_YEAR = 252
NAME = "mock"


class MockMarketDataProvider:
    name = NAME

    def __init__(self, path: Path = DEFAULT_MOCK_PATH, clock: Clock = utcnow) -> None:
        self._path = path
        self._clock = clock

    @cached_property
    def _data(self) -> dict[str, Any]:
        return json.loads(self._path.read_text(encoding="utf-8"))

    @property
    def tickers(self) -> frozenset[str]:
        return frozenset(self._data["securities"])

    def _security(self, ticker: str) -> dict[str, Any]:
        try:
            return self._data["securities"][ticker]
        except KeyError:
            raise SymbolNotFoundError(f"{NAME}: no demo data for {ticker}") from None

    def _freshness(self) -> Freshness:
        now = self._clock()
        return Freshness(source=NAME, as_of=now, fetched_at=now, is_mock=True)

    def get_quote(self, ticker: str) -> Quote:
        sec = self._security(ticker)
        return Quote(
            ticker=ticker,
            price=sec["price"],
            previous_close=sec.get("previous_close"),
            name=sec["name"],
            freshness=self._freshness(),
        )

    def get_daily_history(self, ticker: str, days: int) -> PriceHistory:
        sec = self._security(ticker)
        rng = random.Random(zlib.crc32(ticker.encode()))
        daily_vol = sec.get("annual_volatility", 0.2) / math.sqrt(TRADING_DAYS_PER_YEAR)
        drift = 0.07 / TRADING_DAYS_PER_YEAR

        # Walk backwards from today's demo price so the series ends exactly there.
        closes = [float(sec["price"])]
        for _ in range(days - 1):
            closes.append(closes[-1] / math.exp(drift + rng.gauss(0, daily_vol)))
        closes.reverse()

        bars = []
        for day, close in zip(_weekdays_ending(self._clock().date(), days), closes, strict=True):
            spread = close * daily_vol * 0.5
            open_ = close * math.exp(rng.gauss(0, daily_vol / 2))
            bars.append(
                PriceBar(
                    date=day,
                    open=round(open_, 2),
                    high=round(max(open_, close) + spread, 2),
                    low=round(min(open_, close) - spread, 2),
                    close=round(close, 2),
                    volume=None,
                )
            )
        return PriceHistory(ticker=ticker, bars=bars, adjusted=True, freshness=self._freshness())

    def get_company_overview(self, ticker: str) -> CompanyOverview:
        sec = self._security(ticker)
        return CompanyOverview(
            ticker=ticker,
            name=sec["name"],
            asset_type=sec.get("asset_type"),
            sector=sec.get("sector"),
            pe_ratio=sec.get("pe_ratio"),
            dividend_yield=sec.get("dividend_yield"),
            beta=sec.get("beta"),
            freshness=self._freshness(),
        )

    def get_news(
        self, ticker: str | None = None, query: str | None = None, limit: int = 5
    ) -> list[NewsArticle]:
        return [
            NewsArticle(title=item["title"], summary=item["summary"], source="Finnie demo data")
            for item in self._data["news"][:limit]
        ]

    def freshness(self) -> Freshness:
        return self._freshness()


def _weekdays_ending(end: date, count: int) -> list[date]:
    days: list[date] = []
    current = end
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current -= timedelta(days=1)
    days.reverse()
    return days
