"""Market data exceptions.

The provider chain decides what to do next from the exception type:

- ``TransientProviderError``: retried with backoff, then the next provider is tried.
- ``RateLimitError`` / other ``ProviderError``: not retried; the next provider is tried.
- ``SymbolNotFoundError``: the provider has no such symbol; the next provider is tried,
  and if every provider agrees the service reports "not found" instead of serving mock data.
"""


class MarketDataError(Exception):
    """Base class for all market data errors."""


class InvalidTickerError(MarketDataError, ValueError):
    """The ticker failed validation before any network call."""


class ProviderError(MarketDataError):
    """A provider failed in a way that another provider might not."""


class TransientProviderError(ProviderError):
    """Timeouts, connection failures, and 5xx responses. Safe to retry."""


class RateLimitError(ProviderError):
    """The provider (or our own limiter/budget) refused the request because of rate limits."""


class SymbolNotFoundError(MarketDataError):
    """The provider has no data for this symbol."""


class DataUnavailableError(MarketDataError):
    """Every provider failed and neither cached nor demo data exists."""
