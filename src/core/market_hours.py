"""U.S. stock market hours: is the market open now, and when was the last close?

Uses the NYSE calendar in data/reference/market_calendar.yaml (regular hours, holidays,
and 1:00 p.m. early closes). Used to label prices ("Market open", "Last close, Oct 1,
4:00 PM ET") and to decide when a cached quote is too old to show.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel

from src.core.reference import REFERENCE_DIR

ET = ZoneInfo("America/New_York")
MarketState = Literal["open", "pre_market", "after_hours", "weekend", "holiday"]


class MarketCalendar(BaseModel):
    source_url: str
    verified_on: date
    open: time
    close: time
    early_close: time
    holidays: frozenset[date]
    early_closes: frozenset[date]

    def is_trading_day(self, day: date) -> bool:
        return day.weekday() < 5 and day not in self.holidays

    def session(self, day: date) -> tuple[datetime, datetime]:
        """Open and close of a trading day, in Eastern time."""
        close = self.early_close if day in self.early_closes else self.close
        return datetime.combine(day, self.open, ET), datetime.combine(day, close, ET)


class MarketStatus(BaseModel):
    state: MarketState
    is_open: bool
    last_close: datetime  # Eastern time
    label: str  # "Market open", "Market closed (after hours)", ...


@lru_cache(maxsize=1)
def get_calendar(path: Path = REFERENCE_DIR / "market_calendar.yaml") -> MarketCalendar:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return MarketCalendar.model_validate(data)


def _previous_trading_day(day: date, calendar: MarketCalendar) -> date:
    day -= timedelta(days=1)
    while not calendar.is_trading_day(day):
        day -= timedelta(days=1)
    return day


def market_status(
    now: datetime | None = None, calendar: MarketCalendar | None = None
) -> MarketStatus:
    calendar = calendar or get_calendar()
    local = (now or datetime.now(UTC)).astimezone(ET)
    today = local.date()
    if calendar.is_trading_day(today):
        opens, closes = calendar.session(today)
        if opens <= local < closes:
            last = calendar.session(_previous_trading_day(today, calendar))[1]
            return MarketStatus(state="open", is_open=True, last_close=last, label="Market open")
        if local >= closes:
            return MarketStatus(
                state="after_hours",
                is_open=False,
                last_close=closes,
                label="Market closed (after hours)",
            )
        last = calendar.session(_previous_trading_day(today, calendar))[1]
        return MarketStatus(
            state="pre_market", is_open=False, last_close=last, label="Market closed (pre-market)"
        )
    last = calendar.session(_previous_trading_day(today, calendar))[1]
    if today.weekday() >= 5:
        return MarketStatus(
            state="weekend", is_open=False, last_close=last, label="Market closed (weekend)"
        )
    return MarketStatus(
        state="holiday", is_open=False, last_close=last, label="Market closed (holiday)"
    )


def quote_max_age(
    now: datetime,
    live_ttl: timedelta,
    closed_ttl: timedelta,
    settle: timedelta = timedelta(minutes=15),
) -> timedelta:
    """How old a cached quote may be and still be the latest price.

    While the market is open, prices move, so only ``live_ttl``. While it's closed, a quote
    fetched before the last close (plus ``settle`` for the final prints) is out of date;
    one fetched after it is current, refreshed every ``closed_ttl`` for assets that trade
    around the clock.
    """
    status = market_status(now)
    if status.is_open:
        return live_ttl
    since_close = now - (status.last_close + settle)
    return max(timedelta(0), min(since_close, closed_ttl))


def price_time_label(as_of: datetime, now: datetime | None = None) -> str:
    """When a price is from, and whether it's live, delayed, or the last close.

    E.g. "Live · Oct 2, 2026, 10:41 AM ET · Market open",
    "Delayed about 15 min · Oct 2, 2026, 10:26 AM ET · Market open", or
    "Last close · Oct 1, 2026, 4:00 PM ET · Market closed (after hours)".
    """
    now = now or datetime.now(UTC)
    status = market_status(now)
    stamp = as_of.astimezone(ET)
    when = f"{stamp:%b} {stamp.day}, {stamp.year}, {stamp.hour % 12 or 12}:{stamp:%M %p} ET"
    if not status.is_open:
        kind = (
            "Last close" if stamp >= status.last_close - timedelta(minutes=5) else "Earlier price"
        )
    elif stamp < calendar_open(now):
        kind = "Previous close"  # e.g. an end-of-day quote from a fallback provider
    else:
        delay = (now - as_of).total_seconds() / 60
        kind = "Live" if delay < 2 else f"Delayed about {round(delay)} min"
    return f"{kind} · {when} · {status.label}"


def calendar_open(now: datetime) -> datetime:
    """Today's opening time (call only when the market is open)."""
    return get_calendar().session(now.astimezone(ET).date())[0]
