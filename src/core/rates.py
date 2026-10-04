"""Risk-free rate for Sharpe ratios: the live 13-week Treasury bill yield, with a fallback.

The yield comes from yfinance's ``^IRX`` quote, whose "price" is the annualized yield in
percent (4.21 means 4.21%). It is cached for the daily-data TTL. If it can't be fetched,
or the value is implausible, the configured ``analytics.risk_free_rate`` is used instead,
and the result says so, so a Sharpe ratio is never shown without its rate and source.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Protocol

from pydantic import BaseModel

from src.core.models import Freshness
from src.data.errors import MarketDataError
from src.data.models import Quote

logger = logging.getLogger(__name__)

LIVE_SOURCE = "13-week U.S. Treasury bill yield (^IRX via yfinance)"
FALLBACK_SOURCE = "configured default (analytics.risk_free_rate)"
PLAUSIBLE_RANGE = (-0.01, 0.25)


class RiskFreeRate(BaseModel):
    """The risk-free rate used for a Sharpe ratio (an annual fraction) and where it came from."""

    rate: float  # annual, as a fraction
    source: str
    as_of: date | None
    is_fallback: bool
    freshness: Freshness | None = None

    def label(self) -> str:
        """E.g. ``4.21% (13-week U.S. Treasury bill yield ..., as of Sep 30, 2026)``."""
        if self.is_fallback or self.as_of is None:
            return f"{self.rate:.2%} ({self.source}; live T-bill yield unavailable)"
        when = f"{self.as_of:%b} {self.as_of.day}, {self.as_of.year}"
        return f"{self.rate:.2%} ({self.source}, as of {when})"


class TreasuryYieldSource(Protocol):
    """Anything that can fetch the 13-week T-bill yield."""

    def get_treasury_bill_yield(self) -> Quote:
        """The 13-week T-bill yield; the quote's ``price`` is in percent."""


def fallback_risk_free_rate(rate: float) -> RiskFreeRate:
    """Wrap the configured rate (an annual fraction), labelled as the fallback."""
    return RiskFreeRate(rate=rate, source=FALLBACK_SOURCE, as_of=None, is_fallback=True)


def fetch_risk_free_rate(market: TreasuryYieldSource, fallback: float) -> RiskFreeRate:
    """The live T-bill yield as a fraction, or ``fallback`` if unavailable or implausible.

    Market data errors don't propagate: they're logged as a warning and the fallback is
    used. Plausible means between -1% and 25%.
    """
    try:
        quote = market.get_treasury_bill_yield()
    except MarketDataError as exc:
        logger.warning("T-bill yield unavailable (%s); using the configured risk-free rate", exc)
        return fallback_risk_free_rate(fallback)
    rate = quote.price / 100
    low, high = PLAUSIBLE_RANGE
    if not low <= rate <= high:
        logger.warning("Implausible T-bill yield %.4f; using the configured risk-free rate", rate)
        return fallback_risk_free_rate(fallback)
    return RiskFreeRate(
        rate=round(rate, 6),
        source=LIVE_SOURCE,
        as_of=quote.freshness.as_of.date(),
        is_fallback=False,
        freshness=quote.freshness,
    )
