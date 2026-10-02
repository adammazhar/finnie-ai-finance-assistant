from datetime import UTC, datetime, timedelta

import pytest

from src.core.market_hours import (
    ET,
    get_calendar,
    market_status,
    price_time_label,
    quote_max_age,
)


def et(*args: int) -> datetime:
    return datetime(*args, tzinfo=ET)


def test_calendar_is_sourced():
    calendar = get_calendar()
    assert calendar.source_url == "https://www.nyse.com/markets/hours-calendars"
    assert et(2026, 11, 26).date() in calendar.holidays  # Thanksgiving
    assert et(2026, 11, 27).date() in calendar.early_closes


@pytest.mark.parametrize(
    "now, state, last_close",
    [
        (et(2026, 10, 1, 10, 0), "open", et(2026, 9, 30, 16, 0)),
        (et(2026, 10, 1, 9, 0), "pre_market", et(2026, 9, 30, 16, 0)),
        (et(2026, 10, 1, 17, 30), "after_hours", et(2026, 10, 1, 16, 0)),
        (et(2026, 10, 3, 12, 0), "weekend", et(2026, 10, 2, 16, 0)),  # Saturday
        (et(2026, 11, 26, 12, 0), "holiday", et(2026, 11, 25, 16, 0)),  # Thanksgiving
        (et(2026, 11, 27, 14, 0), "after_hours", et(2026, 11, 27, 13, 0)),  # early close
        (et(2026, 10, 5, 9, 0), "pre_market", et(2026, 10, 2, 16, 0)),  # Monday morning
    ],
)
def test_market_status(now, state, last_close):
    status = market_status(now)
    assert (status.state, status.last_close) == (state, last_close)
    assert status.is_open == (state == "open")


def test_status_labels_and_default_clock():
    assert market_status(et(2026, 10, 3, 12, 0)).label == "Market closed (weekend)"
    assert market_status(et(2026, 10, 1, 10, 0)).label == "Market open"
    assert market_status().state in {"open", "pre_market", "after_hours", "weekend", "holiday"}


def test_quote_max_age():
    live, closed = timedelta(seconds=60), timedelta(minutes=30)
    assert quote_max_age(et(2026, 10, 1, 10, 0), live, closed) == live
    # 4:10 PM: the close plus settling time hasn't passed, so nothing cached is current
    assert quote_max_age(et(2026, 10, 1, 16, 10), live, closed) == timedelta(0)
    assert quote_max_age(et(2026, 10, 1, 16, 25), live, closed) == timedelta(minutes=10)
    assert quote_max_age(et(2026, 10, 3, 12, 0), live, closed) == closed


def test_price_time_labels():
    open_now = et(2026, 10, 1, 10, 41)
    assert price_time_label(et(2026, 10, 1, 10, 40, 30), open_now) == (
        "Live · Oct 1, 2026, 10:40 AM ET · Market open"
    )
    assert price_time_label(et(2026, 10, 1, 10, 26), open_now) == (
        "Delayed about 15 min · Oct 1, 2026, 10:26 AM ET · Market open"
    )
    assert price_time_label(et(2026, 9, 30, 16, 0), open_now) == (
        "Previous close · Sep 30, 2026, 4:00 PM ET · Market open"
    )
    weekend = et(2026, 10, 3, 12, 0)
    assert price_time_label(et(2026, 10, 2, 16, 0).astimezone(UTC), weekend) == (
        "Last close · Oct 2, 2026, 4:00 PM ET · Market closed (weekend)"
    )
    assert price_time_label(et(2026, 10, 1, 15, 0), weekend).startswith("Earlier price · Oct 1")
    assert price_time_label(datetime.now(UTC)).endswith(market_status().label)
