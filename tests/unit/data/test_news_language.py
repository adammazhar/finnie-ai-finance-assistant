import pytest

from src.data.news import is_article, is_english


@pytest.mark.parametrize(
    "text",
    [
        "Tesla shares rise after deliveries beat estimates",
        "Why Nvidia stock is moving today",
        "AAPL Q3 earnings",  # short headlines pass on character set alone
        "Fed holds rates steady; markets rally",
    ],
)
def test_english(text):
    assert is_english(text)


@pytest.mark.parametrize(
    "text",
    [
        "Tesla-Aktie: Analysten sehen weiteres Potenzial nach Zahlen",
        "Las acciones de Tesla suben tras superar las entregas previstas",
        "テスラ株が急騰、納車台数が予想を上回る",
        "",
        "12345 67890",
    ],
)
def test_not_english(text):
    assert not is_english(text)


@pytest.mark.parametrize(
    ("title", "url", "expected"),
    [
        (
            "SPY Sep 2026 665.000 call (SPY260929C00665000) Stock Price, News, Quote & History",
            "https://ca.finance.yahoo.com/quote/SPY260929C00665000/",
            False,
        ),
        ("Nvidia (NVDA) Stock Price, News, Quote & History", None, False),
        ("NVDA options chain", "https://example.com/quote/NVDA/options", False),
        ("Stocks rise as the Fed signals patience", "https://www.reuters.com/markets/x", True),
        ("Is Tesla still a buy?", None, True),
    ],
)
def test_quote_pages_are_not_news(title, url, expected):
    assert is_article(title, url) is expected
