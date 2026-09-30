from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from src.core.models import (
    AGENT_NAMES,
    AgentResult,
    Freshness,
    Holding,
    Source,
    UserProfile,
    normalize_ticker,
)

NOW = datetime(2026, 9, 30, 15, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("aapl", "AAPL"), (" $vti ", "VTI"), ("brk.b", "BRK.B"), ("^gspc", "^GSPC"), ("BF-B", "BF-B")],
)
def test_normalize_ticker(raw, expected):
    assert normalize_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["", "1ABC", "TOOLONGTICKER", "AA PL", "DROP;TABLE", "^"])
def test_invalid_tickers(raw):
    with pytest.raises(ValueError, match="Invalid ticker"):
        normalize_ticker(raw)


def test_holding_validation():
    assert Holding(ticker="voo", shares=1.5).ticker == "VOO"
    for bad in (
        {"ticker": "VOO", "shares": 0},
        {"ticker": "VOO", "shares": -1},
        {"ticker": "VOO", "shares": 1, "cost_basis": -5},
        {"ticker": "??", "shares": 1},
        {"ticker": "VOO", "shares": 1, "extra": 1},
    ):
        with pytest.raises(ValidationError):
            Holding(**bad)


def test_user_profile_defaults_and_bounds():
    p = UserProfile()
    assert (p.knowledge_level, p.risk_tolerance) == ("beginner", "moderate")
    with pytest.raises(ValidationError):
        UserProfile(age=5)
    with pytest.raises(ValidationError):
        UserProfile(risk_tolerance="yolo")


def make_freshness(source="alpha_vantage", minutes_old=3.0, **kw):
    fetched = NOW - timedelta(minutes=minutes_old)
    return Freshness(source=source, as_of=fetched, fetched_at=fetched, **kw)


@pytest.mark.parametrize(
    ("kwargs", "status", "label"),
    [
        ({"minutes_old": 0.2}, "live", "Live · just now"),
        ({"minutes_old": 3}, "live", "Live · 3 min ago"),
        ({"source": "cache", "minutes_old": 47}, "cached", "Cached · 47 min ago"),
        ({"source": "cache", "minutes_old": 150, "is_stale": True}, "stale", "Stale · 2 h ago"),
        ({"source": "yfinance", "minutes_old": 3000, "is_stale": True}, "stale", "Stale · 2 d ago"),
        ({"source": "mock"}, "mock", "Demo data: live feed unavailable"),
        ({"source": "cache", "is_mock": True}, "mock", "Demo data: live feed unavailable"),
    ],
)
def test_freshness_status_and_label(kwargs, status, label):
    f = make_freshness(**kwargs)
    assert f.status == status
    assert f.label(NOW) == label


def test_freshness_mock_source_forces_flag_and_age_never_negative():
    f = make_freshness(source="mock")
    assert f.is_mock is True
    future = make_freshness(minutes_old=-10)
    assert future.age_minutes(NOW) == 0
    assert make_freshness().age_minutes() > 0  # defaults to the real clock


def test_agent_result_contract():
    ok = AgentResult(agent="market", answer="hi", sources=[Source(title="t", kind="market_data")])
    assert ok.ok and ok.handoff == [] and ok.data == {}
    failed = AgentResult(agent="tax", error="boom")
    assert not failed.ok
    with pytest.raises(ValidationError):
        AgentResult(agent="astrology")
    assert set(AGENT_NAMES) == {"finance_qa", "portfolio", "market", "goal_planning", "news", "tax"}
