import pytest

from src.data.news import is_english


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
