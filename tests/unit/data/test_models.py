from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from src.core.models import Freshness
from src.data.models import PriceBar, PriceHistory, Quote, session_close_utc

NOW = datetime(2026, 9, 30, 18, 0, tzinfo=UTC)
FRESH = Freshness(source="yfinance", as_of=NOW, fetched_at=NOW)


def test_session_close_handles_daylight_saving():
    assert session_close_utc(date(2026, 7, 1), NOW) == datetime(2026, 7, 1, 20, tzinfo=UTC)
    assert session_close_utc(date(2026, 1, 5), NOW) == datetime(2026, 1, 5, 21, tzinfo=UTC)
    assert session_close_utc(date(2026, 9, 30), NOW) == NOW  # session still open


def test_quote_derives_change_only_when_missing():
    q = Quote(ticker="X", price=110, previous_close=100, freshness=FRESH)
    assert (q.change, q.change_percent) == (10, 10)
    given = Quote(
        ticker="X", price=110, previous_close=100, change=9, change_percent=9.1, freshness=FRESH
    )
    assert (given.change, given.change_percent) == (9, 9.1)
    assert Quote(ticker="X", price=1, freshness=FRESH).change is None
    with pytest.raises(ValidationError):
        Quote(ticker="X", price=0, freshness=FRESH)


def test_history_sorted_and_frame():
    bars = [
        PriceBar(date=date(2026, 9, d), open=1, high=2, low=0.5, close=d, volume=None)
        for d in (29, 25, 26)
    ]
    hist = PriceHistory(ticker="X", bars=bars, adjusted=True, freshness=FRESH)
    assert hist.closes == [25, 26, 29]
    frame = hist.to_frame()
    assert list(frame["close"]) == [25, 26, 29]
    assert str(frame.index[0].date()) == "2026-09-25"
    empty = PriceHistory(ticker="X", bars=[], adjusted=True, freshness=FRESH)
    assert empty.to_frame().empty
