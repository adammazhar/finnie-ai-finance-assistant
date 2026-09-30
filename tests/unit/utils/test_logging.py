import io
import json
import logging

import pytest

from src.utils.logging import REDACTED, configure_logging, redact


@pytest.fixture
def capture():
    stream = io.StringIO()
    root = logging.getLogger()
    old_level, old_handlers = root.level, list(root.handlers)
    yield stream
    for h in list(root.handlers):
        if h not in old_handlers:
            root.removeHandler(h)
    root.setLevel(old_level)


@pytest.mark.parametrize(
    "secret",
    [
        "sk-proj-abcdefghijklmnopqrstuvwx",
        "sk-abcdefghijklmnopqrstuvwxyz0123",
        "sk-ant-api03-abcdefghijkl",
        "tvly-abcdefghijkl",
    ],
)
def test_redact_known_key_shapes(secret):
    out = redact(f"calling with {secret} now")
    assert secret not in out and REDACTED in out


def test_redact_generic_api_key_assignment():
    out = redact('url?apikey=ABCDEF1234567890&x=1 and "api_key": "QWERTY12345678"')
    assert "ABCDEF1234567890" not in out and "QWERTY12345678" not in out
    assert "apikey=" + REDACTED in out


def test_redact_leaves_normal_text():
    assert redact("AAPL rose 2% (risk-free rate 4%)") == "AAPL rose 2% (risk-free rate 4%)"


def test_json_logging_includes_extras_and_redacts(capture):
    configure_logging("DEBUG", "json", stream=capture)
    log = logging.getLogger("src.test")
    log.info(
        "agent done key=%s", "sk-ant-api03-secretsecret", extra={"agent": "tax", "latency_ms": 12}
    )
    record = json.loads(capture.getvalue().strip().splitlines()[-1])
    assert record["level"] == "INFO" and record["logger"] == "src.test"
    assert record["agent"] == "tax" and record["latency_ms"] == 12
    assert "secretsecret" not in record["message"]
    assert record["ts"].endswith("+00:00")


def test_json_logging_exceptions_are_redacted(capture):
    configure_logging("INFO", "json", stream=capture)
    try:
        raise ValueError("bad key sk-ant-api03-leakedleaked")
    except ValueError:
        logging.getLogger("src.test").exception("failed")
    record = json.loads(capture.getvalue().strip().splitlines()[-1])
    assert "ValueError" in record["exc"] and "leakedleaked" not in record["exc"]


def test_text_format_and_level_filtering(capture):
    configure_logging("warning", "text", stream=capture)
    log = logging.getLogger("src.test")
    log.info("hidden")
    try:
        raise KeyError("tvly-leakedleaked")
    except KeyError:
        log.exception("shown")
    out = capture.getvalue()
    assert "hidden" not in out
    assert "WARNING" not in out and "ERROR" in out and "shown" in out
    assert "leakedleaked" not in out


def test_configure_is_idempotent_and_quiets_http_libs(capture):
    configure_logging(stream=capture)
    configure_logging(stream=capture)
    ours = [h for h in logging.getLogger().handlers if getattr(h, "_finnie_handler", False)]
    assert len(ours) == 1
    assert logging.getLogger("httpx").level == logging.WARNING


def test_json_redacts_traceback_cached_by_another_handler(capture):
    """Another handler (e.g. pytest's) may format the traceback first and cache exc_text."""
    earlier = logging.StreamHandler(io.StringIO())
    earlier.setFormatter(logging.Formatter())
    root = logging.getLogger()
    root.addHandler(earlier)
    configure_logging("INFO", "json", stream=capture)
    try:
        try:
            raise ValueError("sk-ant-api03-fakecached")
        except ValueError:
            logging.getLogger("src.test").exception("failed")
    finally:
        root.removeHandler(earlier)
    assert "fakecached" not in capture.getvalue()
