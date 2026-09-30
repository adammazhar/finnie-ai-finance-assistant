"""SQLite-backed cache with TTL reads and stale fallback.

Entries are never deleted when they expire: ``get_fresh`` ignores them, but ``get`` still
returns them so the service can serve stale data when every provider is down. The same
database holds small counters (e.g. the Alpha Vantage daily request budget) so they are
shared by the web app and the MCP server.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.utils.clock import Clock, utcnow

logger = logging.getLogger(__name__)

MEMORY = ":memory:"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    key TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS counters (
    name TEXT NOT NULL,
    period TEXT NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (name, period)
);
"""


@dataclass(frozen=True)
class CacheEntry:
    key: str
    payload: dict[str, Any]
    source: str
    fetched_at: datetime

    def age(self, now: datetime) -> timedelta:
        return now - self.fetched_at


class TTLCache:
    def __init__(self, path: Path | str = MEMORY, clock: Clock = utcnow) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._conn = self._open(str(path))

    def _open(self, path: str) -> sqlite3.Connection:
        try:
            if path != MEMORY:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
            if path != MEMORY:
                conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            return conn
        except (sqlite3.DatabaseError, OSError) as exc:
            logger.warning("Market cache at %s unusable (%s); using in-memory cache", path, exc)
            conn = sqlite3.connect(MEMORY, check_same_thread=False)
            conn.executescript(_SCHEMA)
            return conn

    def get(self, key: str) -> CacheEntry | None:
        """Return the entry regardless of age, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT payload, source, fetched_at FROM entries WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row[0])
            fetched_at = datetime.fromisoformat(row[2])
        except ValueError:
            logger.warning("Dropping corrupt cache entry %s", key)
            self.delete(key)
            return None
        return CacheEntry(key=key, payload=payload, source=row[1], fetched_at=fetched_at)

    def get_fresh(self, key: str, ttl: timedelta) -> CacheEntry | None:
        entry = self.get(key)
        if entry is None or entry.age(self._clock()) > ttl:
            return None
        return entry

    def set(self, key: str, payload: dict[str, Any], source: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO entries (key, payload, source, fetched_at) "
                "VALUES (?, ?, ?, ?)",
                (key, json.dumps(payload, default=str), source, self._clock().isoformat()),
            )

    def delete(self, key: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM entries WHERE key = ?", (key,))

    def purge_older_than(self, age: timedelta) -> int:
        cutoff = (self._clock() - age).isoformat()
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM entries WHERE fetched_at < ?", (cutoff,))
        return cursor.rowcount

    def increment_counter(self, name: str, period: str) -> int:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO counters (name, period, count) VALUES (?, ?, 1) "
                "ON CONFLICT(name, period) DO UPDATE SET count = count + 1",
                (name, period),
            )
            row = self._conn.execute(
                "SELECT count FROM counters WHERE name = ? AND period = ?", (name, period)
            ).fetchone()
        return int(row[0])

    def get_counter(self, name: str, period: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT count FROM counters WHERE name = ? AND period = ?", (name, period)
            ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
