"""Structured logging with secret redaction.

``configure_logging`` installs one handler on the root logger. JSON output includes any
``extra=`` fields (for example ``turn_id``, ``agent``, ``latency_ms``) so logs can be
queried in CloudWatch or ``docker compose logs``. API-key-shaped strings are masked
before anything is written.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any, TextIO

_HANDLER_MARKER = "_finnie_handler"
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "openai", "anthropic")

SECRET_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?<![A-Za-z0-9])tvly-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)(api[_-]?key[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9_\-]{8,}"),
)
REDACTED = "[REDACTED]"

_STANDARD_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys() | {"message", "asctime"}
)


def redact(text: str) -> str:
    """Mask anything that looks like an API key."""
    for pattern in SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda m: m.group(1) + REDACTED, text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


class SecretRedactingFilter(logging.Filter):
    """Masks API keys in each record's message before any handler formats it."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Swap in the formatted, redacted message and clear ``args``; never drops a record."""
        record.msg = redact(record.getMessage())
        record.args = None
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, including any ``extra=`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize the record; tracebacks are redacted here, messages by the filter."""
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = redact(record.exc_text or self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable single-line format for local development, with secrets redacted."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        """Format the record and redact the whole output, tracebacks included."""
        # Redact the full output: tracebacks may come from a cached ``record.exc_text``
        # formatted by another handler before ours ran.
        return redact(super().format(record))


def configure_logging(
    level: str = "INFO", fmt: str = "json", stream: TextIO | None = None
) -> logging.Handler:
    """Configure the root logger. Safe to call repeatedly; the previous handler is replaced."""
    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, _HANDLER_MARKER, False):
            root.removeHandler(existing)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    handler.addFilter(SecretRedactingFilter())
    setattr(handler, _HANDLER_MARKER, True)
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    # yfinance logs its own "possibly delisted" errors; our client already reports them.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    return handler
