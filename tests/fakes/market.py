"""Test doubles for market data: a mutable clock, scripted providers, and a fake yfinance."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from src.core.models import Freshness
from src.data.models import CompanyOverview, NewsArticle, PriceBar, PriceHistory, Quote

START = datetime(2026, 9, 30, 18, 0, tzinfo=UTC)  # a Wednesday, during US market hours


class FakeClock:
    def __init__(self, now: datetime = START) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


def fresh(source: str, clock: FakeClock) -> Freshness:
    return Freshness(source=source, as_of=clock(), fetched_at=clock())  # type: ignore[arg-type]


class ScriptedProvider:
    """Each method pops the next scripted item: a value to return or an exception to raise.

    Items may also be callables taking the call arguments. Unscripted methods return
    sensible default data.
    """

    def __init__(self, name: str, clock: FakeClock, **scripts: list[Any]) -> None:
        self.name = name
        self.clock = clock
        self.scripts = scripts
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def _run(self, method: str, default: Any, *args: Any) -> Any:
        self.calls.append((method, args))
        script = self.scripts.get(method)
        item = script.pop(0) if script else default
        if isinstance(item, BaseException):
            raise item
        return item(*args) if callable(item) else item

    def calls_to(self, method: str) -> int:
        return sum(1 for name, _ in self.calls if name == method)

    def quote(self, ticker: str, price: float = 100.0) -> Quote:
        return Quote(
            ticker=ticker,
            price=price,
            previous_close=price - 1,
            freshness=fresh(self.name, self.clock),
        )

    def get_quote(self, ticker: str) -> Quote:
        return self._run("get_quote", lambda t: self.quote(t), ticker)

    def get_daily_history(self, ticker: str, days: int) -> PriceHistory:
        def default(t: str, d: int) -> PriceHistory:
            start = self.clock().date() - timedelta(days=d)
            bars = [
                PriceBar(date=start + timedelta(days=i), open=1, high=1, low=1, close=1)
                for i in range(d)
            ]
            return PriceHistory(
                ticker=t, bars=bars, adjusted=True, freshness=fresh(self.name, self.clock)
            )

        return self._run("get_daily_history", default, ticker, days)

    def get_company_overview(self, ticker: str) -> CompanyOverview:
        return self._run(
            "get_company_overview",
            lambda t: CompanyOverview(
                ticker=t, name=f"{t} Inc.", freshness=fresh(self.name, self.clock)
            ),
            ticker,
        )

    def get_news(
        self, ticker: str | None = None, query: str | None = None, limit: int = 5
    ) -> list[NewsArticle]:
        return self._run(
            "get_news",
            lambda *a: [
                NewsArticle(title=f"{self.name} story", url=f"https://{self.name}.example/1")
            ],
            ticker,
            query,
            limit,
        )


class BatchScriptedProvider(ScriptedProvider):
    def get_quotes(self, tickers: list[str]) -> dict[str, Quote]:
        return self._run("get_quotes", lambda ts: {t: self.quote(t) for t in ts}, tickers)


def price_frame(closes: list[float], end: str = "2026-09-30", volume: bool = True) -> pd.DataFrame:
    index = pd.bdate_range(end=end, periods=len(closes), tz="America/New_York")
    data: dict[str, Any] = {
        "Open": [c - 1 for c in closes],
        "High": [c + 1 for c in closes],
        "Low": [c - 2 for c in closes],
        "Close": closes,
    }
    if volume:
        data["Volume"] = [1000 + i for i in range(len(closes))]
    return pd.DataFrame(data, index=index)


class FakeTicker:
    def __init__(self, module: FakeYF, symbol: str) -> None:
        self.module = module
        self.symbol = symbol

    def history(self, **kwargs: Any) -> pd.DataFrame:
        self.module.calls.append(("history", self.symbol, kwargs))
        return self.module._resolve(self.module.history.get(self.symbol, pd.DataFrame()))

    @property
    def info(self) -> dict[str, Any]:
        return self.module._resolve(self.module.info.get(self.symbol, {}))

    def get_news(self, count: int = 10) -> list[dict[str, Any]]:
        return self.module._resolve(self.module.news.get(self.symbol, []))


class FakeSearch:
    def __init__(self, module: FakeYF, query: str, **kwargs: Any) -> None:
        self.news = module._resolve(module.search_news.get(query, []))


class FakeYF:
    """Stands in for the ``yfinance`` module."""

    def __init__(self) -> None:
        self.history: dict[str, Any] = {}
        self.info: dict[str, Any] = {}
        self.news: dict[str, Any] = {}
        self.search_news: dict[str, Any] = {}
        self.download_result: Any = pd.DataFrame()
        self.calls: list[Any] = []

    @staticmethod
    def _resolve(value: Any) -> Any:
        if isinstance(value, BaseException):
            raise value
        return value

    def Ticker(self, symbol: str) -> FakeTicker:  # noqa: N802 - mirrors yfinance
        return FakeTicker(self, symbol)

    def Search(self, query: str, **kwargs: Any) -> FakeSearch:  # noqa: N802
        return FakeSearch(self, query, **kwargs)

    def download(self, tickers: list[str], **kwargs: Any) -> pd.DataFrame:
        self.calls.append(("download", tickers, kwargs))
        return self._resolve(self.download_result)
