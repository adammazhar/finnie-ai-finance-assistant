from datetime import UTC, datetime

import pytest
import responses
from pydantic import SecretStr

from src.data.errors import ProviderError, RateLimitError, TransientProviderError
from src.data.news import TAVILY_URL, TavilyNewsProvider, parse_published

KEY = "tvly-fake-key-000000"


@pytest.fixture
def tavily():
    return TavilyNewsProvider(SecretStr(KEY))


RESULTS = {
    "query": "AAPL stock news",
    "results": [
        {
            "title": "Apple shares climb",
            "url": "https://www.reuters.com/markets/apple",
            "content": "x" * 800,
            "score": 0.9,
            "published_date": "Mon, 29 Sep 2026 14:00:00 GMT",
        },
        {"title": "", "url": "https://skip.me"},
        {"title": "No url or date", "content": ""},
    ],
}


@responses.activate
def test_search_by_ticker(tavily):
    responses.add(
        responses.POST,
        TAVILY_URL,
        json=RESULTS,
        match=[
            responses.matchers.json_params_matcher(
                {"query": "AAPL stock news", "topic": "news", "max_results": 3, "days": 7}
            ),
            responses.matchers.header_matcher({"Authorization": f"Bearer {KEY}"}),
        ],
    )
    news = tavily.get_news(ticker="AAPL", limit=3)
    assert [n.title for n in news] == ["Apple shares climb", "No url or date"]
    first, second = news
    assert first.source == "reuters.com" and len(first.summary) == 500
    assert first.published_at == datetime(2026, 9, 29, 14, tzinfo=UTC)
    assert first.tickers == ["AAPL"]
    assert second.source is None and second.summary is None and second.url is None


@responses.activate
def test_search_by_topic(tavily):
    responses.add(
        responses.POST,
        TAVILY_URL,
        json={"results": []},
        match=[responses.matchers.json_params_matcher({"query": "inflation"}, strict_match=False)],
    )
    assert tavily.get_news(query="inflation") == []


def test_nothing_to_search(tavily):
    assert tavily.get_news() == []


@pytest.mark.parametrize(
    ("status", "error"),
    [(429, RateLimitError), (502, TransientProviderError), (401, ProviderError)],
)
@responses.activate
def test_errors(tavily, status, error):
    responses.add(responses.POST, TAVILY_URL, status=status, json={})
    with pytest.raises(error):
        tavily.get_news(query="x")


@responses.activate
def test_unexpected_shape(tavily):
    responses.add(responses.POST, TAVILY_URL, json={"answer": "no results key"})
    with pytest.raises(ProviderError, match="shape"):
        tavily.get_news(query="x")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Mon, 29 Sep 2026 14:00:00 GMT", datetime(2026, 9, 29, 14, tzinfo=UTC)),
        ("2026-09-29T14:00:00Z", datetime(2026, 9, 29, 14, tzinfo=UTC)),
        ("2026-09-29", datetime(2026, 9, 29, tzinfo=UTC)),
        ("yesterday", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_published(value, expected):
    assert parse_published(value) == expected
