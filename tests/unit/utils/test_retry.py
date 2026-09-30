import pytest

from src.core.config import BackoffConfig
from src.utils.retry import call_with_backoff

CFG = BackoffConfig(max_attempts=3, initial_s=1.0, multiplier=2.0, max_s=10.0)


class FlakyError(Exception):
    pass


def make(outcomes):
    calls = []

    def fn():
        calls.append(1)
        item = outcomes[len(calls) - 1]
        if isinstance(item, BaseException):
            raise item
        return item

    return fn, calls


def test_retries_then_succeeds_with_jittered_exponential_waits(caplog):
    fn, calls = make([FlakyError(), FlakyError(), "ok"])
    sleeps = []
    with caplog.at_level("INFO"):
        assert call_with_backoff(fn, CFG, retry_on=(FlakyError,), sleep=sleeps.append) == "ok"
    assert len(calls) == 3 and len(sleeps) == 2
    # full jitter: each wait is within [0, min(max_s, initial * multiplier**n)]
    assert 0 <= sleeps[0] <= 1.0 and 0 <= sleeps[1] <= 2.0
    assert "Retrying after FlakyError" in caplog.text


def test_gives_up_and_reraises_last_error():
    fn, calls = make([FlakyError("1"), FlakyError("2"), FlakyError("3")])
    with pytest.raises(FlakyError, match="3"):
        call_with_backoff(fn, CFG, retry_on=(FlakyError,), sleep=lambda s: None)
    assert len(calls) == 3


def test_non_retryable_errors_propagate_immediately():
    fn, calls = make([ValueError("bad")])
    with pytest.raises(ValueError):
        call_with_backoff(fn, CFG, retry_on=(FlakyError,), sleep=lambda s: None)
    assert len(calls) == 1


def test_wait_is_capped():
    cfg = BackoffConfig(max_attempts=6, initial_s=5, multiplier=10, max_s=3)
    fn, _ = make([FlakyError()] * 5 + ["ok"])
    sleeps = []
    call_with_backoff(fn, cfg, retry_on=(FlakyError,), sleep=sleeps.append)
    assert max(sleeps) <= 3
