import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import requests
import responses
from pydantic import SecretStr

from src.data.alpha_vantage import BASE_URL, AlphaVantageProvider
from src.data.errors import (
    ProviderError,
    RateLimitError,
    SymbolNotFoundError,
    TransientProviderError,
)
from src.data.rate_limit import DailyBudget, SlidingWindowRateLimiter

FIXTURES = Path(__file__).parents[2] / "fixtures" / "alpha_vantage"
KEY = "FAKEAVKEY1234567"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture
def av(clock):
    return AlphaVantageProvider(SecretStr(KEY), clock=clock)


def stub(rsps, body=None, status=200, **match_params):
    matchers = [responses.matchers.query_param_matcher(match_params, strict_match=False)]
    kwargs = {"json": body} if body is not None else {}
    rsps.add(responses.GET, BASE_URL, status=status, match=matchers, **kwargs)


@responses.activate
def test_get_quote_parses_global_quote(av):
    stub(responses, fixture("global_quote"), function="GLOBAL_QUOTE", symbol="AAPL", apikey=KEY)
    q = av.get_quote("AAPL")
    assert q.price == 225.77 and q.previous_close == 223.95
    assert q.change == 1.82 and q.change_percent == 0.8127
    assert (q.open, q.high, q.low, q.volume) == (224.1, 226.5, 223.2, 41234567)
    # as_of is the 16:00 New York close of the latest trading day (EDT = UTC-4)
    assert q.freshness.as_of == datetime(2026, 9, 29, 20, 0, tzinfo=UTC)
    assert q.freshness.source == "alpha_vantage" and q.freshness.status == "live"


@responses.activate
def test_quote_for_today_is_capped_at_now(av, clock):
    body = fixture("global_quote")
    body["Global Quote"]["07. latest trading day"] = "2026-09-30"  # clock is 18:00 UTC
    stub(responses, body)
    assert av.get_quote("AAPL").freshness.as_of == clock()


@responses.activate
def test_quote_without_trading_day_uses_now(av, clock):
    body = fixture("global_quote")
    del body["Global Quote"]["07. latest trading day"]
    stub(responses, body)
    assert av.get_quote("AAPL").freshness.as_of == clock()


@responses.activate
def test_empty_quote_is_not_found(av):
    stub(responses, fixture("global_quote_empty"))
    with pytest.raises(SymbolNotFoundError):
        av.get_quote("ZZZZ")


@pytest.mark.parametrize("name", ["rate_limit_note", "rate_limit_information"])
@responses.activate
def test_http_200_rate_limit_notice_is_an_error_not_data(av, name):
    stub(responses, fixture(name))
    with pytest.raises(RateLimitError):
        av.get_quote("AAPL")


@responses.activate
def test_premium_endpoint_notice(av):
    stub(responses, fixture("premium_endpoint"))
    with pytest.raises(ProviderError, match="premium") as info:
        av.get_quote("AAPL")
    assert not isinstance(info.value, RateLimitError)


@responses.activate
def test_error_message_is_not_found(av):
    stub(responses, fixture("error_message"))
    with pytest.raises(SymbolNotFoundError):
        av.get_daily_history("BAD", 10)


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (429, RateLimitError),
        (500, TransientProviderError),
        (503, TransientProviderError),
        (401, ProviderError),
    ],
)
@responses.activate
def test_http_status_mapping(av, status, error):
    stub(responses, status=status, body={})
    with pytest.raises(error):
        av.get_quote("AAPL")


@pytest.mark.parametrize(
    ("exc", "error"),
    [
        (requests.Timeout(), TransientProviderError),
        (requests.ConnectionError(), TransientProviderError),
        (requests.TooManyRedirects(), ProviderError),
    ],
)
@responses.activate
def test_transport_errors_are_mapped_without_leaking_the_key(av, exc, error):
    responses.add(responses.GET, BASE_URL, body=exc)
    with pytest.raises(error) as info:
        av.get_quote("AAPL")
    assert KEY not in str(info.value)
    assert info.value.__cause__ is None  # the original carries the URL with the key


@responses.activate
def test_invalid_json_and_wrong_shape(av):
    responses.add(responses.GET, BASE_URL, body="<html>oops</html>")
    with pytest.raises(ProviderError, match="valid JSON"):
        av.get_quote("AAPL")
    responses.replace(responses.GET, BASE_URL, json=["unexpected"])
    with pytest.raises(ProviderError, match="shape"):
        av.get_quote("AAPL")


@responses.activate
def test_daily_history(av):
    stub(
        responses, fixture("time_series_daily"), function="TIME_SERIES_DAILY", outputsize="compact"
    )
    hist = av.get_daily_history("AAPL", 2)
    assert [b.date for b in hist.bars] == [date(2026, 9, 26), date(2026, 9, 29)]
    assert hist.closes == [223.95, 225.77] and hist.adjusted is False
    assert hist.bars[-1].volume == 41234567


def test_long_history_left_to_other_providers(av):
    with pytest.raises(ProviderError, match="100"):
        av.get_daily_history("AAPL", 252)


@responses.activate
def test_history_missing_or_malformed(av):
    stub(responses, {"Meta Data": {}})
    with pytest.raises(SymbolNotFoundError):
        av.get_daily_history("AAPL", 5)
    responses.replace(
        responses.GET, BASE_URL, json={"Time Series (Daily)": {"2026-09-29": {"1. open": "x"}}}
    )
    with pytest.raises(ProviderError, match="malformed"):
        av.get_daily_history("AAPL", 5)


@responses.activate
def test_overview(av):
    stub(responses, fixture("overview"), function="OVERVIEW")
    o = av.get_company_overview("AAPL")
    assert o.name == "Apple Inc" and o.sector == "Technology"
    assert o.industry == "Electronic Computers"
    assert (o.market_cap, o.pe_ratio, o.dividend_yield, o.beta) == (3.4e12, 33.2, 0.0044, 1.24)
    assert (o.week_52_high, o.week_52_low) == (237.23, 164.08)


@responses.activate
def test_overview_empty_for_etfs_is_not_found(av):
    stub(responses, {})
    with pytest.raises(SymbolNotFoundError):
        av.get_company_overview("VTI")


@responses.activate
def test_news_sentiment(av):
    stub(responses, fixture("news_sentiment"), function="NEWS_SENTIMENT", tickers="AAPL")
    news = av.get_news(ticker="AAPL", limit=5)
    assert [n.title for n in news] == [
        "Apple unveils new devices ahead of holiday season",
        "Tech stocks mixed",
    ]
    first, second = news
    assert first.published_at == datetime(2026, 9, 29, 14, 30, tzinfo=UTC)
    assert (first.sentiment_score, first.sentiment_label) == (0.21, "Somewhat-Bullish")
    assert first.tickers == ["AAPL"]
    assert second.published_at is None and second.sentiment_score is None


@responses.activate
def test_news_empty_feed(av):
    stub(responses, {"items": "0"})
    assert av.get_news(ticker="AAPL") == []


def test_news_without_ticker_is_unsupported(av):
    assert av.get_news(query="inflation") == []


def test_local_limiter_blocks_before_network(clock):
    limiter = SlidingWindowRateLimiter(0, 60)
    av = AlphaVantageProvider(SecretStr(KEY), limiter=limiter, clock=clock)
    with pytest.raises(RateLimitError, match="per-minute"):
        av.get_quote("AAPL")  # no responses mock: a network call would fail differently


@responses.activate
def test_daily_budget_consumed_per_request(cache, clock):
    budget = DailyBudget(cache, "alpha_vantage", 1, clock=clock)
    av = AlphaVantageProvider(SecretStr(KEY), budget=budget, clock=clock)
    stub(responses, fixture("global_quote"))
    av.get_quote("AAPL")
    assert budget.remaining() == 0
    with pytest.raises(RateLimitError, match="daily"):
        av.get_quote("AAPL")
    assert len(responses.calls) == 1


def test_value_parsers():
    from src.data.alpha_vantage import _num, _parse_av_time

    assert _num("12.5%") == 12.5 and _num("None") is None and _num("abc") is None
    assert _parse_av_time(None) is None
