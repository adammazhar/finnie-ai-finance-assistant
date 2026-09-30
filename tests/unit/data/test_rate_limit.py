from src.data.rate_limit import DailyBudget, SlidingWindowRateLimiter


class Tick:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_sliding_window():
    tick = Tick()
    limiter = SlidingWindowRateLimiter(2, 60, clock=tick)
    assert limiter.try_acquire() and limiter.seconds_until_available() == 0
    tick.t += 10
    assert limiter.try_acquire()
    assert not limiter.try_acquire()
    assert limiter.seconds_until_available() == 50
    tick.t += 50  # first call leaves the window
    assert limiter.try_acquire()
    assert not limiter.try_acquire()


def test_daily_budget_resets_each_utc_day(cache, clock):
    budget = DailyBudget(cache, "alpha_vantage", 2, clock=clock)
    assert budget.remaining() == 2
    assert budget.try_consume() and budget.try_consume()
    assert not budget.try_consume()
    assert (budget.used(), budget.remaining()) == (2, 0)
    clock.advance(days=1)
    assert budget.remaining() == 2 and budget.try_consume()


def test_zero_budget_never_allows(cache, clock):
    assert not DailyBudget(cache, "x", 0, clock=clock).try_consume()


def test_budget_shared_through_cache(cache, clock):
    a = DailyBudget(cache, "av", 3, clock=clock)
    b = DailyBudget(cache, "av", 3, clock=clock)  # e.g. the MCP server process
    a.try_consume()
    b.try_consume()
    assert a.remaining() == 1


def test_budget_guard_against_race_overshoot(cache, clock):
    budget = DailyBudget(cache, "av", 1, clock=clock)
    cache.increment_counter("av", clock().date().isoformat())  # another process used it
    assert not budget.try_consume()
