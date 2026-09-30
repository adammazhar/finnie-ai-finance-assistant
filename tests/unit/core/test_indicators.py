from datetime import timedelta

import pandas as pd
import pytest

from src.core.indicators import (
    INDEX_PROXIES,
    SECTOR_ETFS,
    MoverSummary,
    build_market_overview,
    latest_cross,
    market_mood,
    realized_volatility,
    rsi,
    sma,
    technical_snapshot,
)
from src.core.models import Freshness
from src.data.errors import DataUnavailableError
from src.data.models import BatchQuotes, PriceBar, PriceHistory, Quote
from tests.fakes.market import START

FRESH = Freshness(source="yfinance", as_of=START, fetched_at=START)


def closes(values):
    return pd.Series(values, index=pd.bdate_range("2025-01-01", periods=len(values)), dtype=float)


def history(ticker, values):
    s = closes(values)
    bars = [PriceBar(date=d.date(), open=v, high=v, low=v, close=v) for d, v in s.items()]
    return PriceHistory(ticker=ticker, bars=bars, adjusted=True, freshness=FRESH)


def test_sma():
    result = sma(closes([1, 2, 3, 4]), 2)
    assert result.isna().iloc[0] and list(result.iloc[1:]) == [1.5, 2.5, 3.5]


def test_rsi_hand_computed_with_wilder_smoothing():
    # deltas: +1, -0.5, +1 ; alpha = 1/2
    # avg gain: 1 -> 0.5 -> 0.75 ; avg loss: 0 -> 0.25 -> 0.125 (first value hidden: min_periods)
    result = rsi(closes([10, 11, 10.5, 11.5]), period=2)
    assert result.isna().iloc[:2].all()
    assert result.iloc[2] == pytest.approx(100 - 100 / (1 + 0.5 / 0.25))
    assert result.iloc[3] == pytest.approx(100 - 100 / (1 + 0.75 / 0.125))


def test_rsi_extremes():
    assert rsi(closes(range(1, 30)), 14).iloc[-1] == 100
    assert rsi(closes(range(30, 1, -1)), 14).iloc[-1] == pytest.approx(0)
    assert rsi(closes([5.0] * 30), 14).iloc[-1] == 50


def test_realized_volatility():
    assert realized_volatility(closes([100 * 1.01**i for i in range(40)])) == pytest.approx(
        0, abs=1e-9
    )
    alternating = closes([100, 101, 100, 101] * 10)
    assert realized_volatility(alternating) > 0.1
    assert realized_volatility(closes([1, 2, 3])) is None


def v_shape(down=150, up=150):
    return [200 - i * 0.5 for i in range(down)] + [125 + i * 1.5 for i in range(up)]


def test_golden_cross_detected():
    s = closes(v_shape())
    event = latest_cross(s, lookback=120)
    assert event.kind == "golden_cross"
    assert s.index[-120].date() <= event.date <= s.index[-1].date()


def test_death_cross_and_no_cross():
    s = closes([100 + i for i in range(200)] + [300 - i * 3 for i in range(100)])
    assert latest_cross(s, lookback=150).kind == "death_cross"
    assert latest_cross(closes(range(1, 300))) is None  # steady rise: no crossover
    assert latest_cross(closes(range(1, 150))) is None  # not enough data for a 200-day average


def test_snapshot_uptrend():
    snap = technical_snapshot(history("SPY", [100 + i * 0.5 for i in range(260)]))
    assert snap.trend == "uptrend" and snap.rsi_14 == 100 and snap.range_position == 1.0
    assert snap.price == 229.5 and snap.high_52w == 229.5 and snap.sma_50 < snap.price
    text = " ".join(snap.notes)
    assert "uptrend" in text and "overbought" in text and "100% of the way" in text
    assert "volatility" in text


def test_snapshot_downtrend():
    snap = technical_snapshot(history("X", [300 - i for i in range(260)]))
    assert snap.trend == "downtrend" and snap.range_position == 0.0
    assert "oversold" in " ".join(snap.notes)


def test_snapshot_with_cross_and_mixed_trend():
    snap = technical_snapshot(history("X", v_shape(150, 110)))
    assert snap.cross is not None
    assert any("cross" in n for n in snap.notes)


def test_snapshot_short_and_flat_history():
    short = technical_snapshot(history("X", [10.0] * 30))
    assert short.sma_200 is None and short.trend == "unknown"
    assert short.range_position is None and short.rsi_14 == 50
    flat = technical_snapshot(history("X", [10.0] * 210))
    assert flat.trend == "mixed"
    with pytest.raises(ValueError, match="No price history"):
        technical_snapshot(PriceHistory(ticker="X", bars=[], adjusted=True, freshness=FRESH))


def mover(ticker, change):
    return MoverSummary(
        ticker=ticker,
        name=SECTOR_ETFS.get(ticker, ticker),
        price=10,
        change_percent=change,
        freshness=FRESH,
    )


@pytest.mark.parametrize(
    ("changes", "label"),
    [
        ([1, 2, 3, 0.5], "broadly higher"),
        ([-1, -2, -3, 0.5], "broadly lower"),
        ([1, -1, 2, -2], "mixed"),
        ([], "unknown"),
    ],
)
def test_market_mood_labels(changes, label):
    sectors = [mover(t, c) for t, c in zip(SECTOR_ETFS, changes, strict=False)]
    mood = market_mood(sectors, None)
    assert mood.label == label and mood.volatility_regime == "unknown"
    if changes:
        assert mood.sectors_total == len(changes)
        assert "Strongest" in mood.summary[1]
    if label == "broadly lower":
        assert mood.summary[0].startswith("1 of 4 sectors is up")
    if label == "broadly higher":
        assert mood.summary[0].startswith("4 of 4 sectors are up")


@pytest.mark.parametrize(
    ("vol", "regime"), [(0.08, "calm"), (0.15, "normal"), (0.25, "elevated"), (0.45, "high")]
)
def test_market_mood_volatility_regimes(vol, regime):
    snap = technical_snapshot(history("SPY", [100 + i * 0.5 for i in range(260)]))
    snap = snap.model_copy(update={"volatility_30d": vol})
    mood = market_mood([mover("XLK", 1.0)], snap)
    assert mood.volatility_regime == regime
    assert any("uptrend" in s for s in mood.summary)


class FakeMarket:
    def __init__(self, missing=(), spy=True):
        self.missing = set(missing)
        self.spy = spy

    def get_quotes(self, tickers):
        quotes = {
            t: Quote(ticker=t, price=100, previous_close=100 - (i - 5) * 0.5, freshness=FRESH)
            for i, t in enumerate(tickers)
            if t not in self.missing
        }
        return BatchQuotes(quotes=quotes, errors={t: "unavailable" for t in self.missing})

    def get_daily_history(self, ticker, days=252):
        if not self.spy:
            raise DataUnavailableError("down")
        return history(ticker, [100 + i * 0.1 + (i % 3) for i in range(260)])


def test_build_market_overview():
    overview = build_market_overview(FakeMarket())
    assert [m.ticker for m in overview.indices] == list(INDEX_PROXIES)
    assert len(overview.sectors) == 11
    changes = [s.change_percent for s in overview.sectors]
    assert changes == sorted(changes, reverse=True)
    assert overview.benchmark.ticker == "SPY" and overview.errors == {}
    assert overview.mood.sectors_total == 11


def test_build_market_overview_partial_failures():
    overview = build_market_overview(FakeMarket(missing={"QQQ", "XLE"}, spy=False))
    assert "QQQ" not in [m.ticker for m in overview.indices] and len(overview.sectors) == 10
    assert overview.benchmark is None
    assert set(overview.errors) == {"QQQ", "XLE", "SPY history"}


def test_cross_dates_are_dates():
    event = latest_cross(closes(v_shape()), lookback=120)
    assert event.date - timedelta(days=0) == event.date


def test_snapshot_with_too_little_data_for_volatility():
    snap = technical_snapshot(history("X", [10.0, 11.0, 12.0, 11.5, 12.5]))
    assert snap.volatility_30d is None and snap.rsi_14 is None
    assert not any("volatility" in n for n in snap.notes)
