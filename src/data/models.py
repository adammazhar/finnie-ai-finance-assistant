"""Market data types returned by providers and the market data service."""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, model_validator

from src.core.models import Freshness

if TYPE_CHECKING:
    import pandas as pd

NEW_YORK = ZoneInfo("America/New_York")


def session_close_utc(day: date, now: datetime) -> datetime:
    """The US market close (16:00 New York) for ``day``, capped at ``now`` for today's session."""
    close = datetime.combine(day, time(16, 0), tzinfo=NEW_YORK).astimezone(UTC)
    return min(close, now)


class Quote(BaseModel):
    """A latest price quote, as returned by providers and ``MarketDataService.get_quote``.

    ``change`` and ``change_percent`` are derived from ``previous_close`` when a provider omits
    them.
    """

    ticker: str
    price: float = Field(gt=0)
    previous_close: float | None = None
    change: float | None = None
    change_percent: float | None = Field(default=None, description="Percent, e.g. 1.25 = +1.25%")
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: int | None = None
    currency: str = "USD"
    name: str | None = None
    freshness: Freshness

    @model_validator(mode="after")
    def _derive_change(self) -> Quote:
        if self.previous_close:
            if self.change is None:
                self.change = round(self.price - self.previous_close, 4)
            if self.change_percent is None:
                self.change_percent = round(self.change / self.previous_close * 100, 4)
        return self


class PriceBar(BaseModel):
    """One daily open/high/low/close bar."""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int | None = None


class PriceHistory(BaseModel):
    """Daily bars for one ticker, kept sorted oldest first."""

    ticker: str
    bars: list[PriceBar]
    adjusted: bool = Field(description="True when closes are split/dividend adjusted")
    freshness: Freshness

    @model_validator(mode="after")
    def _sort(self) -> PriceHistory:
        self.bars.sort(key=lambda bar: bar.date)
        return self

    @property
    def closes(self) -> list[float]:
        """Closing prices, oldest first."""
        return [bar.close for bar in self.bars]

    def to_frame(self) -> pd.DataFrame:
        """The bars as a pandas DataFrame indexed by date (empty when there are no bars)."""
        import pandas as pd

        frame = pd.DataFrame([bar.model_dump() for bar in self.bars])
        if frame.empty:
            return frame
        return frame.set_index(pd.to_datetime(frame.pop("date")))


class CompanyOverview(BaseModel):
    """Company profile and headline fundamentals for one ticker."""

    ticker: str
    name: str
    asset_type: str | None = None
    sector: str | None = None
    industry: str | None = None
    description: str | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    dividend_yield: float | None = Field(default=None, description="Fraction, e.g. 0.015 = 1.5%")
    beta: float | None = None
    week_52_high: float | None = None
    week_52_low: float | None = None
    freshness: Freshness


class NewsArticle(BaseModel):
    """One news item, with sentiment when the provider supplies it (only Alpha Vantage does)."""

    title: str
    url: str | None = None
    source: str | None = None
    published_at: datetime | None = None
    summary: str | None = None
    tickers: list[str] = Field(default_factory=list)
    sentiment_score: float | None = None
    sentiment_label: str | None = None


class NewsFeed(BaseModel):
    """News articles for a ticker or topic, as returned by ``MarketDataService.get_news``."""

    query: str
    articles: list[NewsArticle]
    freshness: Freshness

    @model_validator(mode="after")
    def _news_freshness(self) -> NewsFeed:
        """News is labelled "News fetched ..." rather than as market prices."""
        if self.freshness.kind != "news":
            self.freshness = self.freshness.model_copy(update={"kind": "news"})
        return self


class BatchQuotes(BaseModel):
    """Result of ``MarketDataService.get_quotes``: quotes by ticker and errors by ticker."""

    quotes: dict[str, Quote]
    errors: dict[str, str] = Field(default_factory=dict)
