"""Market data service: the single entry point agents, the UI, and MCP tools use.

Each lookup walks the same chain:

1. a fresh cache entry (within TTL)
2. each live provider in order, with backoff on transient errors
   (quotes and history: yfinance, then Alpha Vantage; company overviews: Alpha Vantage,
   then yfinance; news: yfinance, Tavily, Alpha Vantage)
3. the stale cache entry, flagged ``is_stale``
4. demo data, flagged ``is_mock``

Every result carries a ``Freshness`` record, so callers can always tell users how
current the data is.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import cache
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from src.core.config import MarketDataConfig, Settings, get_settings
from src.core.market_hours import quote_max_age
from src.core.models import Freshness, normalize_ticker
from src.data.alpha_vantage import AlphaVantageProvider
from src.data.cache import TTLCache
from src.data.errors import (
    DataUnavailableError,
    InvalidTickerError,
    MarketDataError,
    ProviderError,
    SymbolNotFoundError,
    TransientProviderError,
)
from src.data.mock_provider import MockMarketDataProvider
from src.data.models import BatchQuotes, CompanyOverview, NewsArticle, NewsFeed, PriceHistory, Quote
from src.data.news import TavilyNewsProvider, is_article, is_english
from src.data.rate_limit import DailyBudget, SlidingWindowRateLimiter
from src.data.yfinance_client import YFinanceProvider
from src.utils.clock import Clock, utcnow
from src.utils.retry import call_with_backoff

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)
MAX_HISTORY_DAYS = 5000
TBILL_13_WEEK = "^IRX"  # yfinance quotes its yield in percent
MAX_NEWS = 20


class PriceProvider(Protocol):
    """What the service needs from a quote, history, and company overview provider."""

    name: str

    def get_quote(self, ticker: str) -> Quote:
        """Latest quote for ``ticker``."""

    def get_daily_history(self, ticker: str, days: int) -> PriceHistory:
        """The last ``days`` daily bars for ``ticker``."""

    def get_company_overview(self, ticker: str) -> CompanyOverview:
        """Company profile and fundamentals for ``ticker``."""


class NewsProvider(Protocol):
    """What the service needs from a news provider."""

    name: str

    def get_news(
        self, ticker: str | None = None, query: str | None = None, limit: int = 5
    ) -> list[NewsArticle]:
        """Up to ``limit`` articles about ``ticker`` or ``query``."""


@dataclass(frozen=True)
class ProviderStatus:
    """Whether one provider is configured, with a short detail line for the UI sidebar."""

    name: str
    enabled: bool
    detail: str


class MarketDataService:
    """Cache-first lookups that fall back through providers, stale cache, then demo data."""

    def __init__(
        self,
        *,
        cache: TTLCache,
        config: MarketDataConfig,
        price_providers: Sequence[PriceProvider],
        overview_providers: Sequence[PriceProvider] | None = None,
        news_providers: Sequence[NewsProvider] = (),
        mock: MockMarketDataProvider | None = None,
        budget: DailyBudget | None = None,
        clock: Clock = utcnow,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._cache = cache
        self._config = config
        self._price = list(price_providers)  # quotes and daily history
        self._overview = list(price_providers if overview_providers is None else overview_providers)
        self._news = list(news_providers)
        self._mock = mock
        self._budget = budget
        self._clock = clock
        self._sleep = sleep
        self._quote_ttl = timedelta(minutes=config.quote_ttl_minutes)
        self._quote_live_ttl = timedelta(seconds=config.quote_live_ttl_seconds)
        self._news_ttl = timedelta(minutes=config.news_ttl_minutes)
        self._history_ttl = timedelta(hours=config.history_ttl_hours)

    # ---- public API -------------------------------------------------------------------

    def get_quote(self, ticker: str) -> Quote:
        """Latest quote, cached for a market-aware TTL.

        Raises ``InvalidTickerError`` for a malformed ticker, ``SymbolNotFoundError`` when every
        provider reports the ticker unknown, and ``DataUnavailableError`` when no live, cached, or
        demo data is available.
        """
        symbol = _normalize(ticker)
        return self._fetch(
            Quote,
            f"quote:{symbol}",
            self._quote_max_age(),
            [(p.name, _bind(p.get_quote, symbol)) for p in self._price],
            mock=lambda m: m.get_quote(symbol),
        )

    def _quote_max_age(self) -> timedelta:
        """Market-aware: a minute while trading, never older than the last close."""
        return quote_max_age(self._clock(), self._quote_live_ttl, self._quote_ttl)

    def get_treasury_bill_yield(self) -> Quote:
        """13-week T-bill yield (``price`` is percent), cached for the daily-data TTL.

        Only yfinance carries ``^IRX``, so other providers aren't asked (that would spend
        Alpha Vantage budget for nothing). There is no demo value: callers fall back to
        the configured rate instead.
        """
        return self._fetch(
            Quote,
            f"rate:{TBILL_13_WEEK}",
            self._history_ttl,
            [
                (p.name, _bind(p.get_quote, TBILL_13_WEEK))
                for p in self._price
                if p.name == "yfinance"
            ],
            mock=None,
        )

    def get_quotes(self, tickers: Iterable[str]) -> BatchQuotes:
        """Quotes for many tickers. Failures are reported per ticker, never raised."""
        errors: dict[str, str] = {}
        symbols: list[str] = []
        for raw in dict.fromkeys(tickers):
            try:
                symbols.append(_normalize(raw))
            except InvalidTickerError as exc:
                errors[raw] = str(exc)

        quotes: dict[str, Quote] = {}
        misses = []
        for symbol in dict.fromkeys(symbols):
            entry = self._cache.get_fresh(f"quote:{symbol}", self._quote_max_age())
            cached = self._from_cache(Quote, entry, stale=False) if entry else None
            if cached:
                quotes[symbol] = cached
            else:
                misses.append(symbol)

        if len(misses) >= self._config.batch_threshold:
            quotes.update(self._batch_quotes(misses))

        for symbol in misses:
            if symbol in quotes:
                continue
            try:
                quotes[symbol] = self.get_quote(symbol)
            except MarketDataError as exc:
                errors[symbol] = str(exc)

        ordered = {s: quotes[s] for s in dict.fromkeys(symbols) if s in quotes}
        return BatchQuotes(quotes=ordered, errors=errors)

    def get_daily_history(self, ticker: str, days: int = 252) -> PriceHistory:
        """The last ``days`` daily bars, cached for the daily-data TTL.

        Raises ``ValueError`` unless ``days`` is 1-5000; otherwise falls back and fails like
        ``get_quote``.
        """
        symbol = _normalize(ticker)
        if not 1 <= days <= MAX_HISTORY_DAYS:
            raise ValueError(f"days must be between 1 and {MAX_HISTORY_DAYS}")
        return self._fetch(
            PriceHistory,
            f"history:{symbol}:{days}",
            self._history_ttl,
            [(p.name, _bind(p.get_daily_history, symbol, days)) for p in self._price],
            mock=lambda m: m.get_daily_history(symbol, days),
        )

    def get_company_overview(self, ticker: str) -> CompanyOverview:
        """Company profile and fundamentals, cached for the daily-data TTL.

        Tries the overview providers (Alpha Vantage first when configured); falls back and fails
        like ``get_quote``.
        """
        symbol = _normalize(ticker)
        return self._fetch(
            CompanyOverview,
            f"overview:{symbol}",
            self._history_ttl,
            [(p.name, _bind(p.get_company_overview, symbol)) for p in self._overview],
            mock=lambda m: m.get_company_overview(symbol),
        )

    def get_news(
        self, ticker: str | None = None, query: str | None = None, limit: int = 5
    ) -> NewsFeed:
        """Recent news for a ticker or a topic. An empty feed means no provider found any."""
        symbol = _normalize(ticker) if ticker else None
        topic = (query or "").strip() or None
        if not symbol and not topic:
            raise ValueError("Provide a ticker or a query")
        limit = max(1, min(limit, MAX_NEWS))
        label = symbol or topic or ""
        key = f"news:{symbol or ''}:{(topic or '').lower()}:{limit}"

        def from_provider(provider: NewsProvider) -> Callable[[], NewsFeed]:
            def call() -> NewsFeed:
                found = provider.get_news(ticker=symbol, query=topic, limit=limit)
                articles = [
                    a
                    for a in found
                    if is_english(f"{a.title} {a.summary or ''}") and is_article(a.title, a.url)
                ]
                if not articles:  # none, none in English, or only quote pages: next provider
                    raise SymbolNotFoundError(f"{provider.name}: no news for {label}")
                return self._feed(label, articles, provider.name)

            return call

        try:
            return self._fetch(
                NewsFeed,
                key,
                self._news_ttl,
                [(p.name, from_provider(p)) for p in self._news],
                mock=lambda m: NewsFeed(
                    query=label, articles=m.get_news(limit=limit), freshness=m.freshness()
                ),
            )
        except SymbolNotFoundError:
            now = self._clock()
            return NewsFeed(
                query=label,
                articles=[],
                freshness=Freshness(source="yfinance", as_of=now, fetched_at=now),
            )

    def provider_status(self) -> list[ProviderStatus]:
        """What's configured, for the UI sidebar and troubleshooting."""
        names = {p.name for p in (*self._price, *self._overview)} | {p.name for p in self._news}
        statuses = []
        for name in ("alpha_vantage", "yfinance", "tavily"):
            enabled = name in names
            detail = "configured" if enabled else "no API key"
            if name == "yfinance":
                detail = "available (no key needed)" if enabled else "disabled"
            if name == "alpha_vantage" and enabled and self._budget:
                detail = f"{self._budget.remaining()} requests left today"
            statuses.append(ProviderStatus(name=name, enabled=enabled, detail=detail))
        return statuses

    # ---- chain ------------------------------------------------------------------------

    def _fetch(
        self,
        model: type[M],
        key: str,
        ttl: timedelta,
        attempts: list[tuple[str, Callable[[], M]]],
        mock: Callable[[MockMarketDataProvider], M] | None,
    ) -> M:
        entry = self._cache.get(key)
        if entry and entry.age(self._clock()) <= ttl:
            cached = self._from_cache(model, entry, stale=False)
            if cached:
                return cached

        not_found = 0
        failures: list[str] = []
        for name, call in attempts:
            try:
                result = call_with_backoff(
                    call,
                    self._config.backoff,
                    retry_on=(TransientProviderError,),
                    sleep=self._sleep,
                )
            except SymbolNotFoundError:
                not_found += 1
                continue
            except ProviderError as exc:
                failures.append(str(exc))
                logger.warning("Provider failed for %s: %s", key, exc, extra={"provider": name})
                continue
            except Exception as exc:  # a provider bug must not take the app down
                failures.append(f"{name}: {type(exc).__name__}")
                logger.exception("Unexpected provider error for %s", key)
                continue
            self._cache.set(key, result.model_dump(mode="json"), source=name)
            return result

        if entry:
            stale = self._from_cache(model, entry, stale=True)
            if stale:
                logger.info("Serving stale cache for %s", key)
                return stale

        if attempts and not_found == len(attempts):
            raise SymbolNotFoundError(f"No data found for {key.split(':')[1]}")

        if mock and self._mock:
            try:
                result = mock(self._mock)
            except SymbolNotFoundError:
                pass
            else:
                logger.warning("Serving demo data for %s", key)
                return result
        raise DataUnavailableError(
            f"Market data unavailable for {key.split(':')[1]}: "
            + ("; ".join(failures) or "no providers configured")
        )

    def _from_cache(self, model: type[M], entry: Any, *, stale: bool) -> M | None:
        try:
            result = model.model_validate(entry.payload)
        except ValidationError:
            logger.warning("Dropping cache entry %s that no longer matches its model", entry.key)
            self._cache.delete(entry.key)
            return None
        original: Freshness = result.freshness  # type: ignore[attr-defined]
        result.freshness = Freshness(  # type: ignore[attr-defined]
            source="cache",
            origin=original.origin or original.source,
            as_of=original.as_of,
            fetched_at=entry.fetched_at,
            is_stale=stale,
            is_mock=original.is_mock,
        )
        return result

    def _batch_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        batcher = next((p for p in self._price if hasattr(p, "get_quotes")), None)
        if batcher is None:
            return {}
        try:
            quotes: dict[str, Quote] = call_with_backoff(
                lambda: batcher.get_quotes(symbols),
                self._config.backoff,
                retry_on=(TransientProviderError,),
                sleep=self._sleep,
            )
        except Exception as exc:
            logger.warning("Batch quote fetch failed (%s); fetching individually", exc)
            return {}
        for symbol, quote in quotes.items():
            self._cache.set(f"quote:{symbol}", quote.model_dump(mode="json"), source=batcher.name)
        return quotes

    def _feed(self, label: str, articles: list[NewsArticle], source: str) -> NewsFeed:
        now = self._clock()
        dates = [a.published_at for a in articles if a.published_at]
        return NewsFeed(
            query=label,
            articles=_dedupe(articles),
            freshness=Freshness(source=source, as_of=max(dates, default=now), fetched_at=now),
        )


def _normalize(ticker: str) -> str:
    try:
        return normalize_ticker(ticker)
    except ValueError as exc:
        raise InvalidTickerError(str(exc)) from None


def _bind[R](fn: Callable[..., R], *args: Any) -> Callable[[], R]:
    return lambda: fn(*args)


def _dedupe(articles: list[NewsArticle]) -> list[NewsArticle]:
    seen: set[str] = set()
    unique = []
    for article in articles:
        key = (article.url or article.title).lower()
        if key not in seen:
            seen.add(key)
            unique.append(article)
    return unique


def build_market_data_service(
    settings: Settings | None = None,
    *,
    clock: Clock = utcnow,
    yf_module: Any = None,
) -> MarketDataService:
    """Wire providers from settings. Providers without an API key are left out of the chain.

    Quotes and history try yfinance first: Alpha Vantage's free quotes are end-of-day, so
    it would show yesterday's close during market hours (found in the Phase 2 live check).
    Alpha Vantage stays first for company overviews, where its fundamentals are richer.
    """
    settings = settings or get_settings()
    cfg = settings.market_data
    cache = TTLCache(settings.resolve_path(cfg.cache_path), clock=clock)
    yfinance = YFinanceProvider(yf_module=yf_module, clock=clock)

    price: list[PriceProvider] = [yfinance]
    overview: list[PriceProvider] = []
    news: list[NewsProvider] = [yfinance]
    budget = None
    if settings.alpha_vantage_api_key:
        budget = DailyBudget(cache, "alpha_vantage", cfg.alpha_vantage.daily_budget, clock=clock)
        alpha_vantage = AlphaVantageProvider(
            settings.alpha_vantage_api_key,
            timeout_s=cfg.request_timeout_s,
            limiter=SlidingWindowRateLimiter(cfg.alpha_vantage.requests_per_minute, 60),
            budget=budget,
            clock=clock,
        )
        price.append(alpha_vantage)
        overview.append(alpha_vantage)
    overview.append(yfinance)
    if settings.tavily_api_key:
        news.append(TavilyNewsProvider(settings.tavily_api_key, timeout_s=cfg.request_timeout_s))
    if settings.alpha_vantage_api_key:
        news.append(alpha_vantage)

    return MarketDataService(
        cache=cache,
        config=cfg,
        price_providers=price,
        overview_providers=overview,
        news_providers=news,
        mock=MockMarketDataProvider(clock=clock),
        budget=budget,
        clock=clock,
    )


@cache
def get_market_data_service() -> MarketDataService:
    """Process-wide service. Call ``get_market_data_service.cache_clear()`` to rebuild."""
    return build_market_data_service()
