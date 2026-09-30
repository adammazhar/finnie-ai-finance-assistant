from datetime import UTC

from src.utils.clock import utcnow


def test_utcnow_is_timezone_aware():
    assert utcnow().tzinfo is UTC
