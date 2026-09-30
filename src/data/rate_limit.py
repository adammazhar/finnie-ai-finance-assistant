"""Client-side rate limiting, so we stop before a provider starts refusing us.

Both limiters are non-blocking: when a limit is reached the caller moves on to the next
provider instead of making the user wait.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

from src.data.cache import TTLCache
from src.utils.clock import Clock, utcnow


class SlidingWindowRateLimiter:
    """Allow at most ``max_calls`` in any rolling ``period_s`` window."""

    def __init__(
        self, max_calls: int, period_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._max_calls = max_calls
        self._period = period_s
        self._clock = clock
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def _evict(self, now: float) -> None:
        while self._calls and now - self._calls[0] >= self._period:
            self._calls.popleft()

    def try_acquire(self) -> bool:
        with self._lock:
            now = self._clock()
            self._evict(now)
            if len(self._calls) >= self._max_calls:
                return False
            self._calls.append(now)
            return True

    def seconds_until_available(self) -> float:
        with self._lock:
            now = self._clock()
            self._evict(now)
            if len(self._calls) < self._max_calls:
                return 0.0
            return self._period - (now - self._calls[0])


class DailyBudget:
    """A per-UTC-day request budget persisted in the cache database."""

    def __init__(self, cache: TTLCache, name: str, limit: int, clock: Clock = utcnow) -> None:
        self._cache = cache
        self._name = name
        self._limit = limit
        self._clock = clock

    def _period(self) -> str:
        return self._clock().date().isoformat()

    def used(self) -> int:
        return self._cache.get_counter(self._name, self._period())

    def remaining(self) -> int:
        return max(0, self._limit - self.used())

    def try_consume(self) -> bool:
        if self._limit <= 0 or self.remaining() <= 0:
            return False
        return self._cache.increment_counter(self._name, self._period()) <= self._limit
