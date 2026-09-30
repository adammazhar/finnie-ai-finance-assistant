"""Exponential backoff with full jitter, built on tenacity."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.core.config import BackoffConfig

logger = logging.getLogger(__name__)


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    wait = state.next_action.sleep if state.next_action else 0
    logger.info(
        "Retrying after %s (attempt %d, waiting %.2fs)",
        type(exc).__name__,
        state.attempt_number,
        wait,
    )


def call_with_backoff[T](
    fn: Callable[[], T],
    config: BackoffConfig,
    *,
    retry_on: tuple[type[BaseException], ...],
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn``, retrying only on ``retry_on`` exceptions. The last error is re-raised."""
    retrying = Retrying(
        stop=stop_after_attempt(config.max_attempts),
        wait=wait_random_exponential(
            multiplier=config.initial_s, exp_base=config.multiplier, max=config.max_s
        ),
        retry=retry_if_exception_type(retry_on),
        reraise=True,
        sleep=sleep,
        before_sleep=_log_retry,
    )
    return retrying(fn)
