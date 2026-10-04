"""Chart builders, formatting helpers, and the risk quiz: no Streamlit needed."""

from datetime import timedelta

import plotly.graph_objects as go
import pytest

from src.core.indicators import MoverSummary
from src.core.monte_carlo import GoalInputs, simulate
from src.web_app import charts
from src.web_app.formatting import (
    agent_badges,
    big_money,
    domain,
    first_sentence,
    freshness_caption,
    match_label,
    md,
    money,
    percent,
    provider_name,
    snippet,
    stream_words,
)
from src.web_app.quiz import QUESTIONS, tolerance_from_scores
from tests.fakes.market_service import FakeMarketService, fresh


def titled(fig: go.Figure) -> str:
    assert isinstance(fig, go.Figure)
    return fig.layout.title.text


@pytest.fixture(scope="module")
def projection():
    inputs = GoalInputs(
        current_balance=10_000,
        monthly_contribution=500,
        years=10,
        target_amount=100_000,
        expected_return=0.06,
        volatility=0.12,
        simulations=500,
        seed=1,
    )
    return simulate(inputs)


# ---- charts -------------------------------------------------------------------------------


def test_allocation_charts():
    mix = {"equity": 0.5, "bond": 0.3, "unknown_mix": 0.1, "new_class": 0.1}
    donut = charts.allocation_donut(mix)
    assert titled(donut) == "Asset allocation"
    assert list(donut.data[0].labels) == ["Stocks", "Bonds", "Mix unknown", "New class"]
    bar = charts.allocation_bar({"Technology": 0.5, "Energy": 0.1, "Health": 0.4})
    assert list(bar.data[0].y) == ["Energy", "Health", "Technology"]  # smallest at the bottom
    assert bar.layout.xaxis.tickformat == ".0%"
    assert bar.layout.xaxis.title.text == "Share of portfolio"


def test_correlation_heatmap():
    fig = charts.correlation_heatmap({"A": {"A": 1.0, "B": 0.3}, "B": {"A": 0.3, "B": 1.0}})
    assert [list(row) for row in fig.data[0].z] == [[1.0, 0.3], [0.3, 1.0]]
    assert (fig.data[0].zmin, fig.data[0].zmax) == (-1, 1)


def test_fan_chart(projection):
    fig = charts.fan_chart(projection)
    names = [trace.name for trace in fig.data]
    assert "Median" in names and "P10 to P90 range" in names
    assert fig.layout.yaxis.tickprefix == "$"
    assert fig.layout.xaxis.title.text == "Years from now"
    assert fig.layout.shapes[0].y0 == 100_000  # the target line
    assert titled(fig) == "Projected balance: range of outcomes"


def test_probability_gauge():
    fig = charts.probability_gauge(0.6432)
    assert fig.data[0].value == 64  # rounded like the heading's "64%"
    assert fig.data[0].number.suffix == "%" and fig.data[0].number.prefix == ""
    low, high = charts.probability_gauge(0.004), charts.probability_gauge(0.996)
    assert (low.data[0].number.prefix, low.data[0].value) == ("<", 1)  # "under 1%"
    assert (high.data[0].number.prefix, high.data[0].value) == (">", 99)  # "over 99%"


def test_price_and_rsi_charts():
    history = FakeMarketService().get_daily_history("SPY", 400)
    price = charts.price_chart(history)
    assert [t.name for t in price.data] == ["Close", "50-day average", "200-day average"]
    assert price.layout.yaxis.title.text == "Price (USD)"
    short = FakeMarketService().get_daily_history("SPY", 30)
    assert [t.name for t in charts.price_chart(short).data] == ["Close"]  # too short for averages
    rsi = charts.rsi_chart(history)
    assert rsi.layout.yaxis.range == (0, 100)
    assert "SPY relative strength" in titled(rsi)


def test_performance_chart():
    from datetime import date

    from src.core.portfolio import Backtest

    test = Backtest(
        dates=[date(2026, 1, 2), date(2026, 1, 5)],
        portfolio=[100.0, 104.0],
        benchmark=[100.0, 102.0],
        benchmark_ticker="SPY",
        coverage=1.0,
    )
    fig = charts.performance_chart(test)
    assert [t.name for t in fig.data] == ["Today's holdings", "SPY (S&P 500)"]
    assert fig.layout.yaxis.title.text == "Value (start of period = 100)"
    assert "Back-test" in fig.layout.title.text
    alone = charts.performance_chart(test.model_copy(update={"benchmark": None}))
    assert len(alone.data) == 1


def test_movers_bar_colors():
    movers = [
        MoverSummary(ticker=t, name=t, price=1.0, change_percent=c, freshness=fresh())
        for t, c in (("UP", 1.2), ("DN", -0.5), ("FLAT", None))
    ]
    fig = charts.movers_bar(movers, "Sectors")
    assert list(fig.data[0].marker.color) == [charts.UP, charts.DOWN, charts.NEUTRAL]
    assert fig.layout.xaxis.ticksuffix == "%"


def test_movers_bar_leaves_room_for_every_label():
    """Regression: a tiny negative move ("-0.01%") had its label cut off at the axis."""
    movers = [
        MoverSummary(ticker=t, name=t, price=1.0, change_percent=c, freshness=fresh())
        for t, c in (("XLY", 1.13), ("XLV", -0.01))
    ]
    fig = charts.movers_bar(movers, "Sectors")
    low, high = fig.layout.xaxis.range
    assert low < -0.01 - 0.2 and high > 1.13 + 0.2
    assert fig.data[0].cliponaxis is False and fig.data[0].textposition == "outside"
    flat = charts.movers_bar([movers[0]], "x")
    assert flat.layout.xaxis.range[0] < 0  # all moves up: still padded below zero


def test_metric_rows_wrap_instead_of_truncating():
    from src.web_app.theme import PALETTES, base_css

    css = base_css(PALETTES["light"])
    assert "flex-wrap: wrap" in css and "min-width: 11rem" in css


# ---- formatting ---------------------------------------------------------------------------


def test_money_and_percent():
    assert money(1234.567) == "$1,235"
    assert money(1234.567, cents=True) == "$1,234.57"
    assert money(None) == percent(None) == "n/a"
    assert percent(0.0523) == "5.2%"
    assert percent(0.0523, 0, signed=True) == "+5%"


def test_freshness_caption_separates_market_data_and_news():
    assert freshness_caption([]) is None
    assert freshness_caption([fresh(), fresh(mock=True)]) == (
        "Market data: demo data (live feed unavailable)"
    )
    older = fresh().model_copy(update={"fetched_at": fresh().fetched_at - timedelta(hours=2)})
    assert freshness_caption([fresh(), older]) == f"Market data: {older.label()}"
    news = fresh().model_copy(update={"kind": "news"})
    caption = freshness_caption([fresh(), news])
    assert caption.startswith("Market data: ") and " · News fetched " in caption
    mock_news = fresh(mock=True).model_copy(update={"kind": "news"})
    assert freshness_caption([mock_news]) == "News: demo data (live feed unavailable)"


def test_dollar_signs_are_escaped_for_markdown():
    assert md("Costs $5 to $10.") == r"Costs \$5 to \$10."
    assert md(r"Already \$5") == r"Already \$5"  # not escaped twice
    assert md(None) == ""


def test_money_helpers():
    assert big_money(2.94e12) == "$2.9T"
    assert big_money(415.2e9) == "$415.2B"
    assert big_money(87e6) == "$87.0M"
    assert big_money(5_000) == "$5,000"
    assert big_money(None) == "n/a"


def test_text_helpers():
    assert domain("https://www.investor.gov/glossary") == "investor.gov"
    assert domain(None) is None
    assert first_sentence("Apple makes phones. In 2020 it earned a lot.") == "Apple makes phones."
    assert first_sentence("   ") is None
    assert first_sentence("No full stop here") == "No full stop here"
    assert first_sentence("word " * 100, limit=20).endswith("…")
    text = "One idea here. Another idea follows. " * 20
    short = snippet(text, limit=60)
    assert short == "One idea here. Another idea follows. One idea here."
    assert snippet("Short text.") == "Short text."
    assert snippet("x" * 50 + " " + "y" * 50, limit=60).endswith("…")


def test_match_labels_and_providers():
    assert [match_label(s) for s in (0.72, 0.55, 0.41)] == ["Strong match", "Good match", "Related"]
    assert provider_name(fresh()) == "Yahoo Finance"
    cached = fresh().model_copy(update={"source": "cache", "origin": "alpha_vantage"})
    assert provider_name(cached) == "Alpha Vantage"
    assert provider_name(fresh().model_copy(update={"source": "some_new"})) == "Some New"


def test_badges():
    assert agent_badges(["portfolio", "goal_planning"]) == (
        ":blue-badge[Portfolio] :blue-badge[Goal planning]"
    )


def test_stream_words_reassembles_the_text():
    text = "An ETF is a fund that trades on an exchange."
    pauses: list[float] = []
    chunks = list(stream_words(text, delay=0.01, sleep=pauses.append))
    assert "".join(chunks) == text
    assert len(chunks) == len(pauses) == 4
    assert "".join(stream_words(text, delay=0)) == text


# ---- quiz ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scores, expected",
    [
        ([1] * 5, "conservative"),
        ([2, 2, 2, 1, 1], "conservative"),
        ([2] * 5, "moderate"),
        ([3, 3, 2, 2, 2], "aggressive"),
    ],
)
def test_quiz_scoring(scores, expected):
    assert tolerance_from_scores(scores) == expected


def test_quiz_needs_every_answer():
    assert len(QUESTIONS) == 5
    with pytest.raises(ValueError, match="Expected 5 answers"):
        tolerance_from_scores([3, 3])


def test_palettes_meet_wcag_aa_for_text():
    from src.web_app.theme import PALETTES

    def luminance(color):
        channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    def ratio(a, b):
        high, low = sorted([luminance(a), luminance(b)], reverse=True)
        return (high + 0.05) / (low + 0.05)

    for name, c in PALETTES.items():
        for text in ("text", "muted", "heading"):
            for surface in ("page", "subtle", "sidebar", "hover", "bubble"):
                assert ratio(c[text], c[surface]) >= 4.5, (name, text, surface)
        assert ratio(c["selected_text"], c["selected"]) >= 4.5, name
        assert ratio(c["muted"], c["selected"]) >= 4.5, name
        for button in ("primary", "primary_hover"):
            assert ratio(c["on_primary"], c[button]) >= 4.5, (name, button)


def test_conversation_row_styles_leave_action_buttons_alone():
    """Regression: a rule on every button in the conversation list painted the inline
    Save/Delete buttons white on white. Row styles must target the row buttons only."""
    import re

    from src.web_app.theme import PALETTES, base_css

    css = base_css(PALETTES["light"])
    selectors = re.findall(r"\.st-key-conversations[^{,]*button[^{,]*", css)
    assert selectors
    for selector in selectors:
        assert '[class*="st-key-conversation_"] button' in selector, selector


def test_theme_type_defaults_to_light_outside_a_session():
    from src.web_app.theme import theme_type

    assert theme_type() == "light"


def test_snippet_keeps_list_items_apart():
    text = "Key takeaways:\n- Born 1951-1959: age 73.\n- Born in 1960 or later: age 75.\n1. Plan."
    assert snippet(text) == (
        "Key takeaways: \u2022 Born 1951-1959: age 73. \u2022 Born in 1960 or later: age 75. "
        "\u2022 Plan."
    )


def test_index_levels_are_not_dollars():
    history = FakeMarketService().get_daily_history("^GSPC", 400)
    fig = charts.price_chart(history)
    assert fig.layout.yaxis.title.text == "Index level" and fig.layout.yaxis.tickprefix == ""
    assert "$" not in fig.data[0].hovertemplate
    assert titled(fig) == "^GSPC level with moving averages"


def test_market_amounts():
    from src.web_app.tabs.markets import _amount

    assert _amount("^GSPC", 6745.123) == "6,745.12" and _amount("^GSPC", 6745.1, 0) == "6,745"
    assert _amount("VTI", 300.5) == "$300.50" and _amount("VTI", 300.5, 0) == "$300"
    assert _amount("VTI", None) == "n/a"
