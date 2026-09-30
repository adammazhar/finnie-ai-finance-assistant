from datetime import UTC, date, datetime

import pandas as pd
import pytest
import requests

from src.data.errors import (
    ProviderError,
    RateLimitError,
    SymbolNotFoundError,
    TransientProviderError,
)
from src.data.yfinance_client import YFinanceProvider, parse_yahoo_news
from tests.fakes.market import FakeYF, price_frame


class YFRateLimitError(Exception):
    pass


class YFTickerMissingError(Exception):
    pass


@pytest.fixture
def yf():
    return FakeYF()


@pytest.fixture
def provider(yf, clock):
    return YFinanceProvider(yf_module=yf, clock=clock)


def test_quote_from_recent_daily_bars(provider, yf):
    yf.history["VTI"] = price_frame([270.0, 272.5, 275.0], end="2026-09-29")
    q = provider.get_quote("VTI")
    assert q.price == 275.0 and q.previous_close == 272.5
    assert q.change == 2.5 and round(q.change_percent, 4) == 0.9174
    assert (q.open, q.high, q.low, q.volume) == (274.0, 276.0, 273.0, 1002)
    assert q.freshness.as_of == datetime(2026, 9, 29, 20, 0, tzinfo=UTC)
    assert q.freshness.source == "yfinance"
    assert yf.calls[0][2]["auto_adjust"] is False


def test_single_bar_quote_has_no_change(provider, yf):
    yf.history["X"] = price_frame([10.0], volume=False)
    q = provider.get_quote("X")
    assert q.previous_close is None and q.change is None and q.volume is None


def test_trailing_nan_rows_are_ignored(provider, yf):
    frame = price_frame([10.0, 11.0, 12.0])
    frame.iloc[-1, frame.columns.get_loc("Close")] = float("nan")
    yf.history["X"] = frame
    assert provider.get_quote("X").price == 11.0


def test_empty_history_is_not_found(provider, yf):
    with pytest.raises(SymbolNotFoundError):
        provider.get_quote("NOPE")
    yf.history["NONE"] = None
    with pytest.raises(SymbolNotFoundError):
        provider.get_quote("NONE")


@pytest.mark.parametrize(
    ("exc", "error"),
    [
        (YFRateLimitError("Too Many Requests"), RateLimitError),
        (YFTickerMissingError("delisted"), SymbolNotFoundError),
        (requests.ConnectionError("down"), TransientProviderError),
        (TimeoutError(), TransientProviderError),
        (KeyError("chart"), ProviderError),
    ],
)
def test_exception_translation(provider, yf, exc, error):
    yf.history["AAPL"] = exc
    with pytest.raises(error):
        provider.get_quote("AAPL")


def test_daily_history_is_adjusted_and_trimmed(provider, yf, clock):
    yf.history["VOO"] = price_frame([float(i) for i in range(1, 11)], end="2026-09-29")
    hist = provider.get_daily_history("VOO", 5)
    assert hist.closes == [6.0, 7.0, 8.0, 9.0, 10.0] and hist.adjusted
    assert hist.bars[-1].date == date(2026, 9, 29)
    kwargs = yf.calls[0][2]
    assert kwargs["auto_adjust"] is True and kwargs["start"] == "2026-09-13"


def test_daily_history_without_volume_and_empty(provider, yf):
    yf.history["IDX"] = price_frame([1.0, 2.0], volume=False)
    assert provider.get_daily_history("IDX", 5).bars[0].volume is None
    with pytest.raises(SymbolNotFoundError):
        provider.get_daily_history("NONE", 5)


def test_batch_quotes(provider, yf):
    frames = {
        "AAPL": price_frame([1.0, 2.0]),
        "MSFT": price_frame([3.0, 4.0]),
        "GONE": price_frame([float("nan"), float("nan")]),
    }
    yf.download_result = pd.concat(frames, axis=1)
    quotes = provider.get_quotes(["AAPL", "MSFT", "GONE", "MISSING"])
    assert {t: q.price for t, q in quotes.items()} == {"AAPL": 2.0, "MSFT": 4.0}
    _, tickers, kwargs = yf.calls[0]
    assert tickers == ["AAPL", "MSFT", "GONE", "MISSING"] and kwargs["group_by"] == "ticker"


def test_batch_quotes_empty_and_failing(provider, yf):
    assert provider.get_quotes(["AAPL"]) == {}
    yf.download_result = None
    assert provider.get_quotes(["AAPL"]) == {}
    yf.download_result = requests.Timeout()
    with pytest.raises(TransientProviderError):
        provider.get_quotes(["AAPL"])


def test_company_overview(provider, yf):
    yf.info["VTI"] = {
        "longName": "Vanguard Total Stock Market Index Fund ETF Shares",
        "quoteType": "ETF",
        "trailingAnnualDividendYield": 0.0125,
        "beta": float("nan"),
        "fiftyTwoWeekHigh": 280,
        "fiftyTwoWeekLow": "n/a",
    }
    o = provider.get_company_overview("VTI")
    assert o.asset_type == "ETF" and o.dividend_yield == 0.0125
    assert o.beta is None and o.week_52_high == 280 and o.week_52_low is None


def test_company_overview_unknown_quote_type_and_missing(provider, yf):
    yf.info["X"] = {"shortName": "X Corp", "quoteType": "FUTURE", "yield": 0.031}
    x = provider.get_company_overview("X")
    assert x.asset_type == "FUTURE" and x.dividend_yield == 0.031
    yf.info["EMPTY"] = None
    with pytest.raises(SymbolNotFoundError):
        provider.get_company_overview("EMPTY")


NEW_STYLE = {
    "id": "abc",
    "content": {
        "title": "Markets rally",
        "summary": "Stocks rose.",
        "pubDate": "2026-09-29T14:00:00Z",
        "canonicalUrl": {"url": "https://finance.yahoo.com/news/markets-rally"},
        "provider": {"displayName": "Yahoo Finance"},
    },
}
OLD_STYLE = {
    "title": "Old format story",
    "link": "https://example.com/old",
    "publisher": "Wire",
    "providerPublishTime": 1790000000,
    "relatedTickers": ["AAPL", "MSFT"],
}


def test_ticker_news_both_formats(provider, yf):
    yf.news["AAPL"] = [NEW_STYLE, OLD_STYLE, {"content": {"title": ""}}]
    news = provider.get_news(ticker="AAPL", limit=5)
    assert [n.title for n in news] == ["Markets rally", "Old format story"]
    assert news[0].url.endswith("markets-rally") and news[0].source == "Yahoo Finance"
    assert news[0].published_at == datetime(2026, 9, 29, 14, tzinfo=UTC)
    assert news[0].tickers == ["AAPL"] and news[1].tickers == ["AAPL", "MSFT"]


def test_topic_news_uses_search_and_limit(provider, yf):
    yf.search_news["inflation"] = [OLD_STYLE, OLD_STYLE | {"title": "Second"}]
    assert [n.title for n in provider.get_news(query="inflation", limit=1)] == ["Old format story"]
    assert provider.get_news() == []


def test_parse_yahoo_news_edge_cases():
    click_through = {
        "content": {
            "title": "T",
            "clickThroughUrl": {"url": "https://x.test"},
            "pubDate": "2026-09-29T14:00:00",
        }
    }
    article = parse_yahoo_news(click_through)
    assert article.url == "https://x.test" and article.published_at.tzinfo is UTC
    assert parse_yahoo_news({"content": {"title": "T", "pubDate": "garbage"}}).published_at is None
    assert parse_yahoo_news({"content": {"title": "T", "pubDate": 5}}).published_at is None
    legacy = parse_yahoo_news({"title": "T", "providerPublishTime": "bad"}, ticker="VTI")
    assert legacy.published_at is None and legacy.tickers == ["VTI"]


def test_default_module_is_real_yfinance(clock):
    import yfinance

    assert YFinanceProvider(clock=clock)._yf is yfinance
