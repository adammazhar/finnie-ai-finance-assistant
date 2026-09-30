import pytest

from src.core.config import BackoffConfig, MarketDataConfig
from src.data.cache import TTLCache
from tests.fakes.market import FakeClock


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def cache(clock) -> TTLCache:
    c = TTLCache(clock=clock)
    yield c
    c.close()


@pytest.fixture
def md_config() -> MarketDataConfig:
    return MarketDataConfig(backoff=BackoffConfig(max_attempts=3, initial_s=0.01, max_s=0.05))


@pytest.fixture
def sleeps() -> list[float]:
    return []
