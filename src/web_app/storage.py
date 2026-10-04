"""What Finnie keeps for each browser: profile, portfolio, goal inputs, and conversations.

There is no login. Each browser gets a random ID in a cookie, and everything here is
keyed by it, in a SQLite file (``app.data_path``, git-ignored). Every query is scoped to
one browser ID. The workflow's own memory (summaries, per-goal choices) lives in the same
file, in LangGraph's checkpoint tables.

Message content is stored here and never written to logs.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.core.models import Holding, UserProfile

TitleSource = Literal["auto", "user"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS browsers (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    current_thread TEXT
);
CREATE TABLE IF NOT EXISTS profiles (
    browser_id TEXT PRIMARY KEY,
    profile TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS portfolios (
    browser_id TEXT PRIMARY KEY,
    holdings TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS goals (
    browser_id TEXT PRIMARY KEY,
    inputs TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    thread_id TEXT PRIMARY KEY,
    browser_id TEXT NOT NULL,
    title TEXT,
    title_source TEXT,
    chat TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS conversations_by_browser ON conversations (browser_id, updated_at);
"""


class SavedConversation(BaseModel):
    """One stored conversation: its thread ID, title, messages, and last update."""

    thread_id: str
    title: str | None = None
    title_source: TitleSource | None = None
    chat: list[dict[str, Any]] = Field(default_factory=list)
    updated_at: datetime


class BrowserData(BaseModel):
    """Everything saved for one browser, as loaded when a session starts."""

    profile: UserProfile | None = None
    portfolio: list[Holding] = Field(default_factory=list)
    goal: dict[str, Any] | None = None  # the Goals tab's last inputs
    conversations: list[SavedConversation] = Field(default_factory=list)  # newest first
    current_thread: str | None = None


def utcnow() -> datetime:
    """The current time in UTC (the store's default clock)."""
    return datetime.now(UTC)


class AppStore:
    """Per-browser profile, portfolio, and conversations in SQLite.

    Each call opens its own connection, and every query is scoped to one browser ID.
    """

    def __init__(self, path: Path, clock: Callable[[], datetime] = utcnow) -> None:
        self.path = path
        self._clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")  # readers and the checkpointer don't block
            db.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A short-lived connection per call: safe across Streamlit's session threads."""
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:  # commit on success, roll back on error
                yield db
        finally:
            db.close()

    def _now(self) -> str:
        return self._clock().isoformat()

    def _touch(self, db: sqlite3.Connection, browser_id: str) -> None:
        db.execute(
            "INSERT INTO browsers (id, created_at, last_seen) VALUES (?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET last_seen = excluded.last_seen",
            (browser_id, self._now(), self._now()),
        )

    # ---- read --------------------------------------------------------------------------

    def load(self, browser_id: str) -> BrowserData:
        """Everything saved for a browser, newest conversation first; records the visit."""
        with self._connect() as db:
            self._touch(db, browser_id)
            profile = db.execute(
                "SELECT profile FROM profiles WHERE browser_id = ?", (browser_id,)
            ).fetchone()
            holdings = db.execute(
                "SELECT holdings FROM portfolios WHERE browser_id = ?", (browser_id,)
            ).fetchone()
            current = db.execute(
                "SELECT current_thread FROM browsers WHERE id = ?", (browser_id,)
            ).fetchone()
            goal = db.execute(
                "SELECT inputs FROM goals WHERE browser_id = ?", (browser_id,)
            ).fetchone()
            rows = db.execute(
                "SELECT thread_id, title, title_source, chat, updated_at FROM conversations "
                "WHERE browser_id = ? ORDER BY updated_at DESC, rowid DESC",
                (browser_id,),
            ).fetchall()
        return BrowserData(
            profile=UserProfile.model_validate_json(profile[0]) if profile else None,
            portfolio=[Holding.model_validate(h) for h in json.loads(holdings[0])]
            if holdings
            else [],
            conversations=[
                SavedConversation(
                    thread_id=thread,
                    title=title,
                    title_source=source,
                    chat=json.loads(chat),
                    updated_at=datetime.fromisoformat(updated),
                )
                for thread, title, source, chat, updated in rows
            ],
            goal=json.loads(goal[0]) if goal else None,
            current_thread=current[0],
        )

    def thread_ids(self, browser_id: str) -> list[str]:
        """The thread IDs of a browser's saved conversations."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT thread_id FROM conversations WHERE browser_id = ?", (browser_id,)
            ).fetchall()
        return [row[0] for row in rows]

    # ---- write -------------------------------------------------------------------------

    def save_profile(self, browser_id: str, profile: UserProfile) -> None:
        """Insert or replace a browser's profile."""
        with self._connect() as db:
            self._touch(db, browser_id)
            db.execute(
                "INSERT INTO profiles (browser_id, profile, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(browser_id) DO UPDATE SET profile = excluded.profile, "
                "updated_at = excluded.updated_at",
                (browser_id, profile.model_dump_json(), self._now()),
            )

    def save_portfolio(self, browser_id: str, holdings: list[Holding]) -> None:
        """Insert or replace a browser's holdings (stored as JSON)."""
        payload = json.dumps([h.model_dump(mode="json") for h in holdings])
        with self._connect() as db:
            self._touch(db, browser_id)
            db.execute(
                "INSERT INTO portfolios (browser_id, holdings, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(browser_id) DO UPDATE SET holdings = excluded.holdings, "
                "updated_at = excluded.updated_at",
                (browser_id, payload, self._now()),
            )

    def save_goal(self, browser_id: str, inputs: dict[str, Any]) -> None:
        """Insert or replace a browser's Goals tab inputs (stored as JSON)."""
        with self._connect() as db:
            self._touch(db, browser_id)
            db.execute(
                "INSERT INTO goals (browser_id, inputs, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(browser_id) DO UPDATE SET inputs = excluded.inputs, "
                "updated_at = excluded.updated_at",
                (browser_id, json.dumps(inputs), self._now()),
            )

    def save_chat(self, browser_id: str, thread_id: str, chat: list[dict[str, Any]]) -> None:
        """Save a conversation's messages and make it the current one."""
        now = self._now()
        with self._connect() as db:
            self._touch(db, browser_id)
            db.execute(
                "INSERT INTO conversations (thread_id, browser_id, chat, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(thread_id) DO UPDATE SET "
                "chat = excluded.chat, updated_at = excluded.updated_at "
                "WHERE conversations.browser_id = excluded.browser_id",
                (thread_id, browser_id, json.dumps(chat), now, now),
            )
            db.execute(
                "UPDATE browsers SET current_thread = ? WHERE id = ?", (thread_id, browser_id)
            )

    def set_title(
        self, browser_id: str, thread_id: str, title: str, source: TitleSource
    ) -> str | None:
        """Store a title; an automatic one never replaces a name the user chose.

        Returns the conversation's title afterwards (None if it doesn't exist).
        """
        with self._connect() as db:
            if source == "user":
                db.execute(
                    "UPDATE conversations SET title = ?, title_source = 'user' "
                    "WHERE thread_id = ? AND browser_id = ?",
                    (title, thread_id, browser_id),
                )
            else:
                db.execute(
                    "UPDATE conversations SET title = ?, title_source = 'auto' "
                    "WHERE thread_id = ? AND browser_id = ? "
                    "AND COALESCE(title_source, '') != 'user'",
                    (title, thread_id, browser_id),
                )
            row = db.execute(
                "SELECT title FROM conversations WHERE thread_id = ? AND browser_id = ?",
                (thread_id, browser_id),
            ).fetchone()
        return row[0] if row else None

    def set_current(self, browser_id: str, thread_id: str | None) -> None:
        """Remember which conversation is open (None for a new, unsaved one)."""
        with self._connect() as db:
            self._touch(db, browser_id)
            db.execute(
                "UPDATE browsers SET current_thread = ? WHERE id = ?", (thread_id, browser_id)
            )

    def delete_conversation(self, browser_id: str, thread_id: str) -> bool:
        """Delete one of a browser's conversations; returns whether it existed.

        If it was the current conversation, the browser is left with none.
        """
        with self._connect() as db:
            deleted = db.execute(
                "DELETE FROM conversations WHERE thread_id = ? AND browser_id = ?",
                (thread_id, browser_id),
            ).rowcount
            db.execute(
                "UPDATE browsers SET current_thread = NULL WHERE id = ? AND current_thread = ?",
                (browser_id, thread_id),
            )
        return bool(deleted)

    def delete_browser(self, browser_id: str) -> list[str]:
        """Remove everything for this browser. Returns the conversations' thread IDs, so
        the workflow's memory for them can be removed too."""
        threads = self.thread_ids(browser_id)
        with self._connect() as db:
            for table in ("conversations", "portfolios", "profiles", "goals"):  # fixed names
                db.execute(f"DELETE FROM {table} WHERE browser_id = ?", (browser_id,))
            db.execute("DELETE FROM browsers WHERE id = ?", (browser_id,))
        return threads
