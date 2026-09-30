import sqlite3
from datetime import timedelta

from src.data.cache import TTLCache

TTL = timedelta(minutes=30)


def test_set_get_and_ttl(cache, clock):
    cache.set("quote:AAPL", {"price": 1.5}, source="yfinance")
    entry = cache.get_fresh("quote:AAPL", TTL)
    assert entry.payload == {"price": 1.5} and entry.source == "yfinance"
    assert entry.fetched_at == clock()

    clock.advance(minutes=30)
    assert cache.get_fresh("quote:AAPL", TTL) is not None  # boundary is inclusive
    clock.advance(seconds=1)
    assert cache.get_fresh("quote:AAPL", TTL) is None
    stale = cache.get("quote:AAPL")  # expired entries stay readable for stale fallback
    assert stale.age(clock()) == timedelta(minutes=30, seconds=1)


def test_missing_key(cache):
    assert cache.get("nope") is None
    assert cache.get_fresh("nope", TTL) is None


def test_overwrite_refreshes_timestamp(cache, clock):
    cache.set("k", {"v": 1}, source="a")
    clock.advance(hours=1)
    cache.set("k", {"v": 2}, source="b")
    entry = cache.get_fresh("k", TTL)
    assert entry.payload == {"v": 2} and entry.source == "b"


def test_delete_and_purge(cache, clock):
    cache.set("old", {}, source="a")
    clock.advance(days=3)
    cache.set("new", {}, source="a")
    assert cache.purge_older_than(timedelta(days=1)) == 1
    assert cache.get("old") is None and cache.get("new") is not None
    cache.delete("new")
    assert cache.get("new") is None


def test_corrupt_payload_is_dropped(cache):
    cache.set("k", {"v": 1}, source="a")
    cache._conn.execute("UPDATE entries SET payload = '{not json' WHERE key = 'k'")
    assert cache.get("k") is None
    assert cache._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 0


def test_counters(cache):
    assert cache.get_counter("av", "2026-09-30") == 0
    assert cache.increment_counter("av", "2026-09-30") == 1
    assert cache.increment_counter("av", "2026-09-30") == 2
    assert cache.increment_counter("av", "2026-10-01") == 1
    assert cache.get_counter("av", "2026-09-30") == 2


def test_file_backed_cache_persists_across_instances(tmp_path, clock):
    path = tmp_path / "nested" / "market.sqlite"
    first = TTLCache(path, clock=clock)
    first.set("k", {"v": 1}, source="a")
    first.close()
    second = TTLCache(path, clock=clock)
    assert second.get("k").payload == {"v": 1}
    assert second._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    second.close()


def test_unusable_database_falls_back_to_memory(tmp_path, clock, caplog):
    bad = tmp_path / "market.sqlite"
    bad.write_bytes(b"this is not a sqlite database" * 100)
    cache = TTLCache(bad, clock=clock)
    cache.set("k", {"v": 1}, source="a")
    assert cache.get("k").payload == {"v": 1}
    assert "in-memory" in caplog.text
    cache.close()


def test_concurrent_writes_are_safe(cache):
    import threading

    def work(n: int) -> None:
        for i in range(50):
            cache.set(f"k{n}-{i}", {"i": i}, source="t")
            cache.increment_counter("c", "p")

    threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cache.get_counter("c", "p") == 200
    assert isinstance(cache._conn, sqlite3.Connection)
