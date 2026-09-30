"""The provider chain: cache -> providers (with backoff) -> stale cache -> demo data."""

import pytest
from pydantic import SecretStr

from src.core.config import load_settings
from src.data import service as service_module
from src.data.errors import (
    DataUnavailableError,
    InvalidTickerError,
    ProviderError,
    RateLimitError,
    SymbolNotFoundError,
    TransientProviderError,
)
from src.data.mock_provider import MockMarketDataProvider
from src.data.models import NewsArticle
from src.data.service import MarketDataService, build_market_data_service
from tests.fakes.market import BatchScriptedProvider, FakeYF, ScriptedProvider, price_frame


def make_service(
    cache, md_config, clock, sleeps, price=(), overview=None, news=(), mock=True, budget=None
):
    return MarketDataService(
        cache=cache,
        config=md_config,
        price_providers=list(price),
        overview_providers=None if overview is None else list(overview),
        news_providers=list(news),
        mock=MockMarketDataProvider(clock=clock) if mock else None,
        budget=budget,
        clock=clock,
        sleep=sleeps.append,
    )


@pytest.fixture
def av(clock):
    return ScriptedProvider("alpha_vantage", clock)


@pytest.fixture
def yfp(clock):
    return BatchScriptedProvider("yfinance", clock)


@pytest.fixture
def svc(cache, md_config, clock, sleeps, av, yfp):
    """Production order: quotes/history yfinance first, overviews Alpha Vantage first."""
    return make_service(
        cache, md_config, clock, sleeps, price=[yfp, av], overview=[av, yfp], news=[yfp, av]
    )


def test_quotes_and_history_try_yfinance_first(svc, av, yfp):
    assert svc.get_quote("AAPL").freshness.source == "yfinance"
    assert svc.get_daily_history("AAPL", 10).freshness.source == "yfinance"
    assert av.calls_to("get_quote") == 0 and av.calls_to("get_daily_history") == 0


def test_overviews_try_alpha_vantage_first(svc, av, yfp):
    assert svc.get_company_overview("AAPL").freshness.source == "alpha_vantage"
    assert yfp.calls_to("get_company_overview") == 0


def test_quote_and_history_fall_back_to_alpha_vantage(svc, av, yfp):
    yfp.scripts["get_quote"] = [ProviderError("yahoo down")]
    yfp.scripts["get_daily_history"] = [RateLimitError("429")]
    assert svc.get_quote("AAPL").freshness.source == "alpha_vantage"
    assert svc.get_daily_history("AAPL", 10).freshness.source == "alpha_vantage"


def test_overview_chain_defaults_to_price_chain(cache, md_config, clock, sleeps, av):
    solo = make_service(cache, md_config, clock, sleeps, price=[av])
    assert solo.get_company_overview("AAPL").freshness.source == "alpha_vantage"


def test_primary_success_is_cached_for_ttl(svc, av, yfp, clock):
    q = svc.get_quote("aapl")
    assert q.ticker == "AAPL" and q.freshness.status == "live"
    assert q.freshness.source == "yfinance"

    clock.advance(minutes=29)
    cached = svc.get_quote("AAPL")
    assert cached.freshness.status == "cached" and cached.freshness.origin == "yfinance"
    assert cached.freshness.fetched_at == q.freshness.fetched_at
    assert yfp.calls_to("get_quote") == 1 and av.calls_to("get_quote") == 0

    clock.advance(minutes=2)  # past the 30-minute TTL
    assert svc.get_quote("AAPL").freshness.status == "live"
    assert yfp.calls_to("get_quote") == 2


def test_transient_errors_retry_with_backoff(svc, yfp, sleeps):
    yfp.scripts["get_quote"] = [TransientProviderError("503"), TransientProviderError("503")]
    assert svc.get_quote("AAPL").freshness.source == "yfinance"
    assert yfp.calls_to("get_quote") == 3 and len(sleeps) == 2


def test_rate_limit_falls_through_immediately(svc, av, yfp, sleeps):
    yfp.scripts["get_quote"] = [RateLimitError("429")]
    assert svc.get_quote("AAPL").freshness.source == "alpha_vantage"
    assert yfp.calls_to("get_quote") == 1 and sleeps == []


def test_exhausted_retries_fall_through(svc, av, yfp):
    yfp.scripts["get_quote"] = [TransientProviderError("x")] * 3
    assert svc.get_quote("AAPL").freshness.source == "alpha_vantage"


def test_unexpected_provider_bug_is_contained(svc, yfp, caplog):
    yfp.scripts["get_quote"] = [KeyError("boom")]
    assert svc.get_quote("AAPL").freshness.source == "alpha_vantage"
    assert "Unexpected provider error" in caplog.text


def test_stale_cache_when_all_providers_fail(svc, av, yfp, clock):
    svc.get_quote("AAPL")
    clock.advance(hours=2)
    av.scripts["get_quote"] = [ProviderError("down")]
    yfp.scripts["get_quote"] = [ProviderError("down")]
    q = svc.get_quote("AAPL")
    assert q.freshness.status == "stale" and q.freshness.is_stale
    assert q.freshness.label(clock()) == "Stale · 2 h ago"


def test_mock_when_no_cache_and_all_fail(svc, av, yfp, caplog):
    av.scripts["get_quote"] = [ProviderError("down")]
    yfp.scripts["get_quote"] = [RateLimitError("429")]
    q = svc.get_quote("TSLA")
    assert q.freshness.is_mock and q.price == 240.0
    assert "demo data" in caplog.text


def test_unavailable_when_no_mock_data_for_ticker(svc, av, yfp):
    av.scripts["get_quote"] = [ProviderError("down")]
    yfp.scripts["get_quote"] = [ProviderError("down")]
    with pytest.raises(DataUnavailableError, match="ZZZZ"):
        svc.get_quote("ZZZZ")


def test_not_found_everywhere_is_reported_not_mocked(svc, av, yfp):
    av.scripts["get_quote"] = [SymbolNotFoundError("x")]
    yfp.scripts["get_quote"] = [SymbolNotFoundError("x")]
    with pytest.raises(SymbolNotFoundError):
        svc.get_quote("AAPL")  # AAPL has demo data, but a real "not found" wins


def test_not_found_on_primary_only_uses_fallback(svc, av):
    av.scripts["get_company_overview"] = [SymbolNotFoundError("ETF")]
    assert svc.get_company_overview("VTI").freshness.source == "yfinance"


def test_no_providers_and_no_mock(cache, md_config, clock, sleeps):
    bare = make_service(cache, md_config, clock, sleeps, mock=False)
    with pytest.raises(DataUnavailableError, match="no providers configured"):
        bare.get_quote("AAPL")


def test_invalid_ticker_rejected_before_any_call(svc, av, yfp):
    with pytest.raises(InvalidTickerError):
        svc.get_quote("DROP TABLE")
    assert av.calls == [] and yfp.calls == []


def test_corrupt_cache_entry_is_dropped(svc, cache, yfp):
    cache.set("quote:AAPL", {"unexpected": True}, source="yfinance")
    assert svc.get_quote("AAPL").freshness.status == "live"
    assert yfp.calls_to("get_quote") == 1


def test_mock_results_stay_flagged_after_caching(svc, av, yfp, clock):
    av.scripts["get_quote"] = [ProviderError("down")]
    yfp.scripts["get_quote"] = [ProviderError("down")]
    svc.get_quote("KO")
    # demo data is not written to cache, so the next call tries providers again
    assert svc.get_quote("KO").freshness.source == "yfinance"


def test_history_and_overview_use_longer_ttl(svc, av, yfp, clock):
    hist = svc.get_daily_history("VOO", 30)
    assert len(hist.bars) == 30
    clock.advance(hours=11)
    assert svc.get_daily_history("VOO", 30).freshness.status == "cached"
    svc.get_company_overview("VOO")
    clock.advance(hours=11)
    assert svc.get_company_overview("VOO").freshness.status == "cached"
    assert yfp.calls_to("get_daily_history") == 1
    assert av.calls_to("get_company_overview") == 1


@pytest.mark.parametrize("days", [0, 5001])
def test_history_day_bounds(svc, days):
    with pytest.raises(ValueError, match="days"):
        svc.get_daily_history("VOO", days)


def test_batch_quotes_mix_cache_batch_and_errors(svc, av, yfp, clock):
    svc.get_quote("AAPL")  # cached
    yfp.scripts["get_quotes"] = [lambda ts: {t: yfp.quote(t, 50) for t in ts if t != "BRK.B"}]
    result = svc.get_quotes(["AAPL", "msft", "GOOGL", "BRK.B", "VTI", "MSFT", "bad ticker"])
    assert list(result.quotes) == ["AAPL", "MSFT", "GOOGL", "BRK.B", "VTI"]
    assert result.quotes["AAPL"].freshness.status == "cached"
    assert (
        result.quotes["MSFT"].price == 50 and result.quotes["MSFT"].freshness.source == "yfinance"
    )
    # BRK.B was missing from the batch, so it was fetched on its own
    assert ("get_quote", ("BRK.B",)) in yfp.calls
    assert "bad ticker" in result.errors
    assert ("get_quotes", (["MSFT", "GOOGL", "BRK.B", "VTI"],)) in yfp.calls
    # batch results are cached for later single lookups
    assert svc.get_quote("GOOGL").freshness.status == "cached"


def test_batch_below_threshold_goes_individually(svc, yfp):
    svc.get_quotes(["AAPL", "MSFT"])
    assert yfp.calls_to("get_quotes") == 0


def test_batch_failure_falls_back_to_individual(svc, av, yfp):
    yfp.scripts["get_quotes"] = [ProviderError("download failed")]
    yfp.scripts["get_quote"] = [ProviderError("x")]  # AAPL fails on yfinance
    av.scripts["get_quote"] = [ProviderError("x")]  # ...and on Alpha Vantage -> demo data
    result = svc.get_quotes(["AAPL", "MSFT", "ZZZZ"])
    assert result.quotes["AAPL"].freshness.is_mock
    assert set(result.quotes) >= {"AAPL"}


def test_batch_reports_per_ticker_errors(svc, av, yfp):
    av.scripts["get_quote"] = [SymbolNotFoundError("x")]
    yfp.scripts["get_quote"] = [SymbolNotFoundError("x")]
    result = svc.get_quotes(["NOPE"])
    assert result.quotes == {} and "NOPE" in result.errors


def test_batch_without_batch_capable_provider(cache, md_config, clock, sleeps, av):
    svc = make_service(cache, md_config, clock, sleeps, price=[av])
    assert len(svc.get_quotes(["A", "B", "C"]).quotes) == 3


def test_news_chain_order_and_dedupe(svc, yfp, av):
    dup = NewsArticle(title="Same", url="https://x.test/1")
    yfp.scripts["get_news"] = [[dup, dup, NewsArticle(title="Other")]]
    feed = svc.get_news(ticker="aapl", limit=5)
    assert feed.query == "AAPL" and [a.title for a in feed.articles] == ["Same", "Other"]
    assert feed.freshness.source == "yfinance" and av.calls_to("get_news") == 0
    assert yfp.calls[0] == ("get_news", ("AAPL", None, 5))


def test_news_empty_first_provider_moves_on(svc, yfp, av):
    yfp.scripts["get_news"] = [[]]
    feed = svc.get_news(query="  inflation ")
    assert feed.freshness.source == "alpha_vantage" and feed.query == "inflation"


def test_news_nobody_has_any_returns_empty_feed(svc, yfp, av):
    yfp.scripts["get_news"] = [[]]
    av.scripts["get_news"] = [[]]
    feed = svc.get_news(ticker="AAPL")
    assert feed.articles == [] and not feed.freshness.is_mock


def test_news_all_failing_serves_demo(svc, yfp, av):
    yfp.scripts["get_news"] = [ProviderError("x")]
    av.scripts["get_news"] = [RateLimitError("x")]
    feed = svc.get_news(ticker="AAPL", limit=50)
    assert feed.freshness.is_mock and len(feed.articles) == 4


def test_news_uses_latest_published_date_as_of(svc, yfp):
    from datetime import UTC, datetime

    older = NewsArticle(title="a", published_at=datetime(2026, 9, 28, tzinfo=UTC))
    newer = NewsArticle(title="b", published_at=datetime(2026, 9, 29, tzinfo=UTC))
    yfp.scripts["get_news"] = [[older, newer]]
    assert svc.get_news(ticker="AAPL").freshness.as_of == newer.published_at


def test_news_requires_ticker_or_query(svc):
    with pytest.raises(ValueError, match="ticker or a query"):
        svc.get_news(query="   ")


def test_provider_status(cache, md_config, clock, sleeps, av, yfp):
    from src.data.rate_limit import DailyBudget

    budget = DailyBudget(cache, "alpha_vantage", 25, clock=clock)
    budget.try_consume()
    svc = make_service(cache, md_config, clock, sleeps, price=[yfp, av], news=[yfp], budget=budget)
    status = {s.name: s for s in svc.provider_status()}
    assert status["alpha_vantage"].detail == "24 requests left today"
    assert status["yfinance"].detail == "available (no key needed)"
    assert not status["tavily"].enabled and status["tavily"].detail == "no API key"
    empty = make_service(cache, md_config, clock, sleeps)
    assert {s.name: s.detail for s in empty.provider_status()}["yfinance"] == "disabled"


# ---- wiring ---------------------------------------------------------------------------


def settings_with(tmp_path, monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "llm: {providers: {openai: {main: {model: m}, fast: {model: m}},"
        " anthropic: {main: {model: a}, fast: {model: a}}}}\n"
        f"market_data: {{cache_path: '{(tmp_path / 'cache.sqlite').as_posix()}'}}\n",
        encoding="utf-8",
    )
    return load_settings(cfg)


def test_build_without_keys_uses_yfinance_only(tmp_path, monkeypatch, clock):
    svc = build_market_data_service(
        settings_with(tmp_path, monkeypatch), clock=clock, yf_module=FakeYF()
    )
    assert [p.name for p in svc._price] == ["yfinance"]
    assert [p.name for p in svc._overview] == ["yfinance"]
    assert [p.name for p in svc._news] == ["yfinance"]
    assert (tmp_path / "cache.sqlite").exists()


def test_build_with_all_keys_orders_chains(tmp_path, monkeypatch, clock):
    s = settings_with(
        tmp_path,
        monkeypatch,
        ALPHA_VANTAGE_API_KEY="FAKEAV1234567890",
        TAVILY_API_KEY="tvly-fake-00000000",
    )
    svc = build_market_data_service(s, clock=clock, yf_module=FakeYF())
    assert [p.name for p in svc._price] == ["yfinance", "alpha_vantage"]
    assert [p.name for p in svc._overview] == ["alpha_vantage", "yfinance"]
    assert [p.name for p in svc._news] == ["yfinance", "tavily", "alpha_vantage"]
    assert svc.provider_status()[0].detail == "25 requests left today"


def test_end_to_end_with_real_clients_and_fake_yfinance(tmp_path, monkeypatch, clock):
    """No AV key: yfinance serves the quote; then yfinance breaks and the cache serves it."""
    yf = FakeYF()
    yf.history["VTI"] = price_frame([270.0, 275.0])
    svc = build_market_data_service(settings_with(tmp_path, monkeypatch), clock=clock, yf_module=yf)
    svc._sleep = lambda s: None
    assert svc.get_quote("VTI").price == 275.0
    clock.advance(hours=1)
    yf.history["VTI"] = ConnectionError("offline")
    stale = svc.get_quote("VTI")
    assert stale.freshness.is_stale and stale.price == 275.0


def test_get_market_data_service_is_cached(monkeypatch):
    service_module.get_market_data_service.cache_clear()
    monkeypatch.setattr(service_module, "build_market_data_service", lambda: object())
    assert service_module.get_market_data_service() is service_module.get_market_data_service()
    service_module.get_market_data_service.cache_clear()


def test_secret_str_keys_are_passed_through(tmp_path, monkeypatch, clock):
    s = settings_with(tmp_path, monkeypatch, ALPHA_VANTAGE_API_KEY="FAKEAV1234567890")
    av = build_market_data_service(s, clock=clock, yf_module=FakeYF())._overview[0]
    assert isinstance(av._api_key, SecretStr)


def test_unreadable_stale_entry_falls_through_to_demo(svc, cache, av, yfp):
    cache.set("quote:KO", {"broken": True}, source="yfinance")
    av.scripts["get_quote"] = [ProviderError("down")]
    yfp.scripts["get_quote"] = [ProviderError("down")]
    assert svc.get_quote("KO").freshness.is_mock


# ---- 13-week T-bill yield --------------------------------------------------------------


def test_treasury_yield_uses_yfinance_only_with_daily_ttl(svc, av, yfp, clock):
    yfp.scripts["get_quote"] = [lambda t: yfp.quote(t, 4.21), lambda t: yfp.quote(t, 4.3)]
    first = svc.get_treasury_bill_yield()
    assert (first.ticker, first.price) == ("^IRX", 4.21)
    clock.advance(hours=11)
    assert svc.get_treasury_bill_yield().freshness.status == "cached"
    clock.advance(hours=2)  # past the 12-hour daily-data TTL
    assert svc.get_treasury_bill_yield().price == 4.3
    assert av.calls == []  # Alpha Vantage doesn't carry ^IRX; never spend budget on it


def test_treasury_yield_has_no_demo_value(svc, yfp):
    yfp.scripts["get_quote"] = [ProviderError("down")]
    with pytest.raises(DataUnavailableError):
        svc.get_treasury_bill_yield()


def test_treasury_yield_without_yfinance(cache, md_config, clock, sleeps, av):
    solo = make_service(cache, md_config, clock, sleeps, price=[av])
    with pytest.raises(DataUnavailableError, match="no providers configured"):
        solo.get_treasury_bill_yield()
