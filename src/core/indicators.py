"""Technical indicators and a plain-English market overview.

Indicators are computed locally from daily closes rather than requested from an API, to
save rate-limited quota. They describe what prices have done; they don't predict them.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Literal, Protocol

import pandas as pd
from pydantic import BaseModel

from src.core.models import Freshness
from src.data.errors import MarketDataError
from src.data.models import BatchQuotes, PriceHistory

TRADING_DAYS = 252

INDEX_PROXIES: dict[str, str] = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq-100",
    "DIA": "Dow Jones Industrial Average",
    "IWM": "Russell 2000 (small companies)",
}
# The index itself, shown next to the ETF that tracks it (indexes can't be bought).
INDEX_LEVELS: dict[str, str] = {"SPY": "^GSPC", "QQQ": "^NDX", "DIA": "^DJI", "IWM": "^RUT"}
# Names people (and models) use for indexes, mapped to the Yahoo symbols the data uses.
# Without this, "SPX" finds nothing and the S&P 500 looks unavailable.
INDEX_ALIASES: dict[str, str] = {
    "SPX": "^GSPC",
    "GSPC": "^GSPC",
    "INX": "^GSPC",
    "DJI": "^DJI",
    "DJIA": "^DJI",
    "NDX": "^NDX",
    "COMP": "^IXIC",
    "IXIC": "^IXIC",
    "RUT": "^RUT",
    "VIX": "^VIX",
}


def index_symbol(ticker: str) -> str:
    """The Yahoo symbol for an index alias (``SPX`` -> ``^GSPC``); other tickers unchanged."""
    return INDEX_ALIASES.get(ticker.strip().upper().lstrip("$"), ticker)


SECTOR_ETFS: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

Trend = Literal["uptrend", "downtrend", "mixed", "unknown"]


def sma(closes: pd.Series, window: int) -> pd.Series:
    """Simple moving average; NaN until ``window`` observations exist."""
    return closes.rolling(window=window, min_periods=window).mean()


def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing (0-100)."""
    delta = closes.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    with_losses = 100 - 100 / (1 + avg_gain / avg_loss)
    # No losses in the window: RSI is 100 (or undefined/50 if there were no moves at all).
    result = with_losses.where(avg_loss > 0, 100.0)
    result = result.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return result.where(avg_gain.notna())


def realized_volatility(closes: pd.Series, window: int = 30) -> float | None:
    """Annualized standard deviation of the last ``window`` daily returns."""
    returns = closes.pct_change().dropna().tail(window)
    if len(returns) < min(window, 10):
        return None
    return float(returns.std(ddof=1)) * math.sqrt(TRADING_DAYS)


class CrossEvent(BaseModel):
    """A 50/200-day moving-average crossover and the date it happened."""

    kind: Literal["golden_cross", "death_cross"]
    date: date


def latest_cross(closes: pd.Series, lookback: int = 60) -> CrossEvent | None:
    """Most recent 50/200-day moving-average crossover within ``lookback`` trading days."""
    fast, slow = sma(closes, 50), sma(closes, 200)
    above = (fast > slow)[slow.notna()].tail(lookback + 1)
    if len(above) < 2:
        return None
    flipped = above.ne(above.shift()).iloc[1:]
    flip_dates = flipped[flipped].index
    if flip_dates.empty:
        return None
    when = flip_dates[-1]
    kind: Literal["golden_cross", "death_cross"] = "golden_cross" if above[when] else "death_cross"
    return CrossEvent(kind=kind, date=when.date())


class TechnicalSnapshot(BaseModel):
    """Moving averages, RSI, volatility, and 52-week range for one ticker, with plain-English notes.

    ``volatility_30d`` is annualized, as a fraction. ``range_position`` is 0 at the
    52-week low and 1 at the high. Values are ``None`` when there isn't enough history.
    """

    ticker: str
    price: float
    as_of: date
    sma_50: float | None
    sma_200: float | None
    rsi_14: float | None
    volatility_30d: float | None
    high_52w: float
    low_52w: float
    range_position: float | None  # 0 = at 52-week low, 1 = at 52-week high
    trend: Trend
    cross: CrossEvent | None
    notes: list[str]


def _last(series: pd.Series) -> float | None:
    value = series.iloc[-1] if len(series) else float("nan")
    return None if pd.isna(value) else round(float(value), 4)


def technical_snapshot(history: PriceHistory) -> TechnicalSnapshot:
    """Compute a :class:`TechnicalSnapshot` from daily closes.

    The trend is "uptrend" when price > 50-day > 200-day average, "downtrend" when
    reversed, and "unknown" without 200 days of history. Raises ``ValueError`` when the
    history is empty.
    """
    frame = history.to_frame()
    if frame.empty:
        raise ValueError(f"No price history for {history.ticker}")
    closes = frame["close"]
    price = float(closes.iloc[-1])
    sma50, sma200 = _last(sma(closes, 50)), _last(sma(closes, 200))
    rsi14 = _last(rsi(closes, 14))
    year = closes.tail(TRADING_DAYS)
    high, low = float(year.max()), float(year.min())
    position = (price - low) / (high - low) if high > low else None

    if sma50 is None or sma200 is None:
        trend: Trend = "unknown"
    elif price > sma50 > sma200:
        trend = "uptrend"
    elif price < sma50 < sma200:
        trend = "downtrend"
    else:
        trend = "mixed"

    cross = latest_cross(closes)
    vol = realized_volatility(closes)
    return TechnicalSnapshot(
        ticker=history.ticker,
        price=price,
        as_of=history.bars[-1].date,
        sma_50=sma50,
        sma_200=sma200,
        rsi_14=rsi14,
        volatility_30d=round(vol, 4) if vol is not None else None,
        high_52w=high,
        low_52w=low,
        range_position=round(position, 4) if position is not None else None,
        trend=trend,
        cross=cross,
        notes=_snapshot_notes(history.ticker, trend, rsi14, position, cross, vol),
    )


def _snapshot_notes(
    ticker: str,
    trend: Trend,
    rsi14: float | None,
    position: float | None,
    cross: CrossEvent | None,
    vol: float | None,
) -> list[str]:
    notes = []
    if trend == "uptrend":
        notes.append(
            f"{ticker} is above its 50-day and 200-day moving averages, a pattern often "
            "described as an uptrend."
        )
    elif trend == "downtrend":
        notes.append(
            f"{ticker} is below its 50-day and 200-day moving averages, a pattern often "
            "described as a downtrend."
        )
    if rsi14 is not None and rsi14 >= 70:
        notes.append(
            f"RSI is {rsi14:.0f}. Readings above 70 are often called 'overbought', meaning "
            "prices rose quickly; it isn't a reliable signal on its own."
        )
    elif rsi14 is not None and rsi14 <= 30:
        notes.append(
            f"RSI is {rsi14:.0f}. Readings below 30 are often called 'oversold', meaning "
            "prices fell quickly; it isn't a reliable signal on its own."
        )
    if position is not None:
        notes.append(f"The price is {position:.0%} of the way from its 52-week low to its high.")
    if cross:
        label = "golden cross" if cross.kind == "golden_cross" else "death cross"
        notes.append(
            f"A {label} (the 50-day average crossing the 200-day) happened on {cross.date:%b %d}."
        )
    if vol is not None:
        notes.append(f"Recent volatility is about {vol:.0%} a year, annualized from 30 days.")
    return notes


class MoverSummary(BaseModel):
    """One index, index ETF, or sector ETF's price and daily change (``change_percent`` is %)."""

    ticker: str
    name: str
    price: float
    change_percent: float | None
    freshness: Freshness


class MarketMood(BaseModel):
    """A plain-English read of the day: sector breadth and the S&P 500's volatility regime.

    ``average_sector_change`` is in percent (1.2 means +1.2%).
    """

    label: Literal["broadly higher", "broadly lower", "mixed", "unknown"]
    sectors_up: int
    sectors_total: int
    average_sector_change: float | None
    volatility_regime: Literal["calm", "normal", "elevated", "high", "unknown"]
    summary: list[str]


class MarketOverview(BaseModel):
    """Indices, sectors, an S&P 500 technical snapshot, and the market mood.

    ``errors`` maps each ticker (or ``SPY history``) that couldn't be fetched to its error.
    """

    indices: list[MoverSummary]
    levels: dict[str, MoverSummary] = {}  # ETF ticker -> the index it tracks (e.g. ^GSPC)
    sectors: list[MoverSummary]
    benchmark: TechnicalSnapshot | None
    mood: MarketMood
    errors: dict[str, str]


def market_mood(sectors: list[MoverSummary], benchmark: TechnicalSnapshot | None) -> MarketMood:
    """Describe sector breadth and the benchmark's volatility regime.

    "Broadly higher" means at least 70% of sectors are up and "broadly lower" at most
    30%. The regime comes from the benchmark's 30-day annualized volatility: calm under
    12%, normal under 20%, elevated under 30%, else high.
    """
    changes = [s.change_percent for s in sectors if s.change_percent is not None]
    up = sum(1 for c in changes if c > 0)
    average = round(sum(changes) / len(changes), 3) if changes else None
    if not changes:
        label: Literal["broadly higher", "broadly lower", "mixed", "unknown"] = "unknown"
    elif up / len(changes) >= 0.7:
        label = "broadly higher"
    elif up / len(changes) <= 0.3:
        label = "broadly lower"
    else:
        label = "mixed"

    vol = benchmark.volatility_30d if benchmark else None
    if vol is None:
        regime: Literal["calm", "normal", "elevated", "high", "unknown"] = "unknown"
    elif vol < 0.12:
        regime = "calm"
    elif vol < 0.20:
        regime = "normal"
    elif vol < 0.30:
        regime = "elevated"
    else:
        regime = "high"

    summary = []
    if changes:
        verb = "is" if up == 1 else "are"
        summary.append(
            f"{up} of {len(changes)} sectors {verb} up, with an average move of {average:+.2f}%."
        )
        best = max(sectors, key=lambda s: s.change_percent or -math.inf)
        worst = min(sectors, key=lambda s: s.change_percent or math.inf)
        summary.append(
            f"Strongest: {best.name} ({best.change_percent:+.2f}%). "
            f"Weakest: {worst.name} ({worst.change_percent:+.2f}%)."
        )
    if vol is not None:
        summary.append(
            f"S&P 500 volatility over the last 30 days is about {vol:.0%} annualized ({regime})."
        )
    if benchmark and benchmark.trend in ("uptrend", "downtrend"):
        summary.append(f"The S&P 500 is in a longer-term {benchmark.trend} by moving averages.")
    return MarketMood(
        label=label,
        sectors_up=up,
        sectors_total=len(changes),
        average_sector_change=average,
        volatility_regime=regime,
        summary=summary,
    )


class MarketData(Protocol):
    """The market data calls :func:`build_market_overview` needs."""

    def get_quotes(self, tickers: list[str]) -> BatchQuotes:
        """Latest quotes for ``tickers``, with per-ticker errors for any that failed."""

    def get_daily_history(self, ticker: str, days: int = ...) -> PriceHistory:
        """The last ``days`` daily bars for ``ticker``."""


def build_market_overview(market: MarketData, history_days: int = TRADING_DAYS) -> MarketOverview:
    """Index and sector snapshot plus an S&P 500 technical read, from one batch of quotes."""
    names = INDEX_PROXIES | SECTOR_ETFS
    batch = market.get_quotes([*names, *INDEX_LEVELS.values()])

    def summarize(tickers: dict[str, str]) -> list[MoverSummary]:
        return [
            MoverSummary(
                ticker=t,
                name=name,
                price=batch.quotes[t].price,
                change_percent=batch.quotes[t].change_percent,
                freshness=batch.quotes[t].freshness,
            )
            for t, name in tickers.items()
            if t in batch.quotes
        ]

    errors = dict(batch.errors)
    try:
        benchmark = technical_snapshot(market.get_daily_history("SPY", history_days))
    except (MarketDataError, ValueError) as exc:
        benchmark = None
        errors["SPY history"] = str(exc)
    sectors = summarize(SECTOR_ETFS)
    levels = {
        etf: summary
        for etf, index in INDEX_LEVELS.items()
        for summary in summarize({index: INDEX_PROXIES[etf]})
    }
    return MarketOverview(
        indices=summarize(INDEX_PROXIES),
        levels=levels,
        sectors=sorted(sectors, key=lambda s: s.change_percent or 0.0, reverse=True),
        benchmark=benchmark,
        mood=market_mood(sectors, benchmark),
        errors=errors,
    )
