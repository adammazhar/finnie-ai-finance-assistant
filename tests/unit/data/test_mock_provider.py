import json

import pytest

from src.data.errors import SymbolNotFoundError
from src.data.mock_provider import DEFAULT_MOCK_PATH, MockMarketDataProvider


@pytest.fixture
def mock(clock):
    return MockMarketDataProvider(clock=clock)


def test_reference_file_is_complete():
    data = json.loads(DEFAULT_MOCK_PATH.read_text(encoding="utf-8"))
    assert "Not real market data" in data["_notice"]
    securities = data["securities"]
    for ticker in ["SPY", "QQQ", "DIA", "IWM", "VTI", "BND", "AAPL", "TSLA"]:
        assert ticker in securities
    sectors = [t for t in securities if t.startswith("XL")]
    assert len(sectors) == 11  # all SPDR sector ETFs for the Markets tab
    for ticker, sec in securities.items():
        assert sec["price"] > 0 and sec["previous_close"] > 0, ticker
        assert {"name", "asset_type", "sector", "annual_volatility"} <= set(sec), ticker


def test_quote_is_flagged_mock(mock):
    q = mock.get_quote("AAPL")
    assert q.name == "Apple Inc." and q.change is not None
    assert q.freshness.is_mock and q.freshness.status == "mock"
    assert "AAPL" in mock.tickers


def test_unknown_ticker_never_invents_prices(mock):
    for call in (mock.get_quote, mock.get_company_overview):
        with pytest.raises(SymbolNotFoundError):
            call("ZZZZ")
    with pytest.raises(SymbolNotFoundError):
        mock.get_daily_history("ZZZZ", 5)


def test_history_is_deterministic_weekdays_ending_at_demo_price(mock, clock):
    first = mock.get_daily_history("VTI", 30)
    second = mock.get_daily_history("VTI", 30)
    assert first.closes == second.closes
    assert len(first.bars) == 30 and first.closes[-1] == 275.0
    assert first.bars[-1].date == clock().date()
    assert all(bar.date.weekday() < 5 for bar in first.bars)
    assert all(
        bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high
        for bar in first.bars
    )
    assert mock.get_daily_history("AAPL", 30).closes != first.closes


def test_overview_and_news(mock):
    o = mock.get_company_overview("KO")
    assert o.sector == "Consumer Staples" and o.dividend_yield == 0.028
    news = mock.get_news(limit=2)
    assert len(news) == 2 and all(n.title.startswith("Demo:") for n in news)
    assert mock.freshness().is_mock
