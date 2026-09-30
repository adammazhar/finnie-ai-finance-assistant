from datetime import UTC, date, datetime

import pytest

from src.core.config import load_settings
from src.core.models import Freshness
from src.core.rates import (
    FALLBACK_SOURCE,
    LIVE_SOURCE,
    fallback_risk_free_rate,
    fetch_risk_free_rate,
)
from src.data.errors import DataUnavailableError
from src.data.models import Quote

CLOSE = datetime(2026, 9, 29, 20, 0, tzinfo=UTC)


class TBill:
    def __init__(self, value):
        self.value = value

    def get_treasury_bill_yield(self):
        if isinstance(self.value, Exception):
            raise self.value
        freshness = Freshness(source="yfinance", as_of=CLOSE, fetched_at=CLOSE)
        return Quote(ticker="^IRX", price=self.value, freshness=freshness)


def test_live_yield_is_converted_from_percent():
    rate = fetch_risk_free_rate(TBill(4.215), fallback=0.042)
    assert rate.rate == pytest.approx(0.04215) and not rate.is_fallback
    assert rate.source == LIVE_SOURCE and rate.as_of == date(2026, 9, 29)
    assert rate.freshness.source == "yfinance"
    assert rate.label() == (
        "4.21% (13-week U.S. Treasury bill yield (^IRX via yfinance), as of Sep 29, 2026)"
    )


def test_fallback_when_unavailable(caplog):
    rate = fetch_risk_free_rate(TBill(DataUnavailableError("down")), fallback=0.042)
    assert (rate.rate, rate.is_fallback, rate.as_of) == (0.042, True, None)
    assert rate.source == FALLBACK_SOURCE
    assert rate.label() == (
        "4.20% (configured default (analytics.risk_free_rate); live T-bill yield unavailable)"
    )
    assert "configured risk-free rate" in caplog.text


@pytest.mark.parametrize("percent", [55.0, 30.0])
def test_implausible_values_fall_back(percent, caplog):
    rate = fetch_risk_free_rate(TBill(percent), fallback=0.042)
    assert rate.is_fallback and rate.rate == 0.042
    assert "Implausible" in caplog.text


def test_config_fallback_is_4_2_percent():
    assert load_settings().analytics.risk_free_rate == 0.042
    assert fallback_risk_free_rate(0.042).rate == 0.042
