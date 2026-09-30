"""HTTP helper that maps transport and status failures onto market data errors.

Error messages never include the request URL, because query strings can carry API keys.
"""

from __future__ import annotations

from typing import Any

import requests

from src.data.errors import ProviderError, RateLimitError, TransientProviderError


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    provider: str,
    timeout_s: float,
    **kwargs: Any,
) -> Any:
    try:
        response = session.request(method, url, timeout=timeout_s, **kwargs)
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise TransientProviderError(f"{provider}: {type(exc).__name__}") from None
    except requests.RequestException as exc:
        raise ProviderError(f"{provider}: {type(exc).__name__}") from None

    status = response.status_code
    if status == 429:
        raise RateLimitError(f"{provider}: HTTP 429 rate limited")
    if status >= 500:
        raise TransientProviderError(f"{provider}: HTTP {status}")
    if status >= 400:
        raise ProviderError(f"{provider}: HTTP {status}")
    try:
        return response.json()
    except ValueError:
        raise ProviderError(f"{provider}: response was not valid JSON") from None
