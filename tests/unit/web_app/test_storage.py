"""The per-browser store: what's saved, how titles behave, and what deleting removes."""

from datetime import UTC, datetime, timedelta

import pytest

from src.core.models import Holding, UserProfile
from src.web_app.storage import AppStore

A, B = "a" * 32, "b" * 32


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


@pytest.fixture
def store(tmp_path):
    return AppStore(tmp_path / "nested" / "finnie.sqlite", clock=Clock())


def chat(*questions):
    return [{"role": "user", "content": q, "output": None} for q in questions]


def test_a_new_browser_has_nothing_saved(store):
    saved = store.load(A)
    assert (saved.profile, saved.portfolio, saved.conversations, saved.current_thread) == (
        None,
        [],
        [],
        None,
    )


def test_profile_and_portfolio_round_trip(store):
    store.save_profile(A, UserProfile(knowledge_level="advanced", age=40))
    store.save_portfolio(A, [Holding(ticker="VTI", shares=10, cost_basis=2500)])
    store.save_profile(A, UserProfile(knowledge_level="intermediate"))  # replaced, not added
    saved = store.load(A)
    assert saved.profile == UserProfile(knowledge_level="intermediate")
    assert saved.portfolio == [Holding(ticker="VTI", shares=10, cost_basis=2500)]
    assert store.load(B).profile is None  # other browsers see nothing


def test_conversations_newest_first_with_current_thread(store):
    store.save_chat(A, "t1", chat("first"))
    store.save_chat(A, "t2", chat("second"))
    store.save_chat(A, "t1", chat("first", "again"))  # t1 is now the most recent
    saved = store.load(A)
    assert [c.thread_id for c in saved.conversations] == ["t1", "t2"]
    assert saved.conversations[0].chat == chat("first", "again")
    assert saved.current_thread == "t1"
    store.set_current(A, None)
    assert store.load(A).current_thread is None
    assert sorted(store.thread_ids(A)) == ["t1", "t2"]


def test_a_browser_cant_touch_another_browsers_conversation(store):
    store.save_chat(A, "t1", chat("mine"))
    store.save_chat(B, "t1", chat("not mine"))  # same thread ID from another browser
    assert store.load(A).conversations[0].chat == chat("mine")
    assert store.load(B).conversations == []
    assert store.set_title(B, "t1", "Hijacked", "user") is None
    assert store.delete_conversation(B, "t1") is False
    assert store.load(A).conversations[0].title is None


def test_a_user_chosen_name_beats_automatic_titles(store):
    store.save_chat(A, "t1", chat("q"))
    assert store.set_title(A, "t1", "Auto Title", "auto") == "Auto Title"
    assert store.set_title(A, "t1", "My Plan", "user") == "My Plan"
    assert store.set_title(A, "t1", "Newer Auto Title", "auto") == "My Plan"
    saved = store.load(A).conversations[0]
    assert (saved.title, saved.title_source) == ("My Plan", "user")


def test_delete_conversation(store):
    store.save_chat(A, "t1", chat("q1"))
    store.save_chat(A, "t2", chat("q2"))
    assert store.delete_conversation(A, "t2") is True
    saved = store.load(A)
    assert [c.thread_id for c in saved.conversations] == ["t1"]
    assert saved.current_thread is None  # t2 was current


def test_delete_browser_removes_everything_and_reports_threads(store):
    store.save_profile(A, UserProfile())
    store.save_portfolio(A, [Holding(ticker="VTI", shares=1)])
    store.save_chat(A, "t1", chat("q1"))
    store.save_chat(B, "t9", chat("other browser"))
    assert store.delete_browser(A) == ["t1"]
    assert store.load(A).model_dump() == {
        "profile": None,
        "portfolio": [],
        "conversations": [],
        "current_thread": None,
    }
    assert [c.thread_id for c in store.load(B).conversations] == ["t9"]


def test_data_survives_a_new_store_on_the_same_file(store, tmp_path):
    store.save_chat(A, "t1", chat("q1"))
    reopened = AppStore(store.path)
    assert [c.thread_id for c in reopened.load(A).conversations] == ["t1"]
