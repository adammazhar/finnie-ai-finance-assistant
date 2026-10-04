"""Portfolio metrics against hand-computed values.

Base case (prices VTI=100, BND=50, AAPL=200):
    VTI 6 sh = $600 (60%), BND 6 sh = $300 (30%), AAPL 0.5 sh = $100 (10%); total $1,000
"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.core.config import AnalyticsConfig
from src.core.models import Freshness, Holding
from src.core.portfolio import (
    PortfolioError,
    analyze_portfolio,
    backtest,
    correlation_matrix,
    daily_returns,
    expense_ratio_label,
    fetch_and_analyze,
    herfindahl,
    holding_error,
    holdings_from_csv,
    merge_holdings,
    risk_metrics,
)
from src.core.rates import fallback_risk_free_rate as rf
from src.core.reference import (
    ExpenseRatio,
    SecurityInfo,
    asset_class_label,
    get_catalog,
    get_risk_profiles,
)
from src.data.errors import DataUnavailableError, SymbolNotFoundError
from src.data.models import BatchQuotes, CompanyOverview, PriceBar, PriceHistory, Quote
from tests.fakes.market import START

CATALOG = get_catalog()
PRICES = {"VTI": 100.0, "BND": 50.0, "AAPL": 200.0}
BASE = [
    Holding(ticker="VTI", shares=6, cost_basis=500),
    Holding(ticker="BND", shares=6),
    Holding(ticker="AAPL", shares=0.5, cost_basis=120),
]
CONFIG = AnalyticsConfig(min_history_days=3)


def securities(*tickers, **extra):
    table = {t: CATALOG.get(t) for t in tickers}
    table.update(extra)
    return table


@pytest.fixture
def base():
    return analyze_portfolio(
        BASE,
        PRICES,
        securities=securities("VTI", "BND", "AAPL"),
        profile=get_risk_profiles()["moderate"],
        config=CONFIG,
    )


def test_values_weights_and_totals(base):
    assert base.total_value == 1000
    assert [(h.ticker, h.value, round(h.weight, 4)) for h in base.holdings] == [
        ("VTI", 600, 0.6),
        ("BND", 300, 0.3),
        ("AAPL", 100, 0.1),
    ]
    assert base.asset_allocation == {"equity": 0.7, "bond": 0.3}
    assert base.sector_allocation == {"Broad Market": 0.6, "Fixed Income": 0.3, "Technology": 0.1}
    assert base.hhi == 0.46 and base.effective_holdings == 2.17


def test_unrealized_gains(base):
    vti, bnd, aapl = base.holdings
    assert (vti.unrealized_gain, vti.unrealized_gain_pct) == (100, 0.2)
    assert bnd.unrealized_gain is None and bnd.unrealized_gain_pct is None
    assert aapl.unrealized_gain == -20
    assert (base.total_cost_basis, base.unrealized_gain) == (620, 80)
    assert base.unrealized_gain_pct == pytest.approx(80 / 620)


def test_diversification_score(base):
    # concentration: AAPL is the only single-company holding -> 1 - sqrt(0.1^2) = 0.9
    # asset class: 1 - (0.7^2 + 0.3^2) = 0.42, scaled by 1/(2/3) -> 0.63
    # sector: broad VTI spread over 11 sectors plus 10% Technology -> capped at 1.0
    assert base.diversification_components == {
        "concentration": 0.9,
        "asset_class": 0.63,
        "sector": 1.0,
    }
    assert base.diversification_score == 84.9  # 100 * (0.4*0.9 + 0.3*0.63 + 0.3*1.0)


@pytest.mark.parametrize(
    ("holdings", "expected"),
    [
        ({"TSLA": 1}, 0.0),  # one stock: nothing diversified
        ({"VTI": 1}, 70.0),  # broad fund but only stocks
        ({"VTI": 0.6, "BND": 0.4}, 91.6),  # classic 60/40
        ({"BND": 1}, 70.0),  # no stocks -> no sector risk
    ],
)
def test_diversification_reference_cases(holdings, expected):
    prices = dict.fromkeys(holdings, 1.0)
    rows = [Holding(ticker=t, shares=w) for t, w in holdings.items()]
    result = analyze_portfolio(rows, prices, securities=securities(*holdings), config=CONFIG)
    assert result.diversification_score == expected


def test_expenses_and_fee_drag(base):
    # (0.6 * 0.03% + 0.3 * 0.03% + 0.1 * 0 for a stock) = 0.027%
    assert base.weighted_expense_ratio == pytest.approx(0.00027)
    assert base.expense_ratio_coverage == 1.0 and base.annual_fees == 0.27
    # 1000 * (1.06^n - (1.06 - 0.00027)^n)
    assert base.fee_drag == {10: 4.56, 20: 16.3, 30: 43.73}


def test_expense_ratio_unknown_for_unclassified_fund():
    mystery = CATALOG.classify("MYST", asset_type="ETF")
    result = analyze_portfolio(
        [Holding(ticker="MYST", shares=1), Holding(ticker="VTI", shares=1)],
        {"MYST": 100, "VTI": 100},
        securities=securities("VTI", MYST=mystery),
        config=CONFIG,
    )
    assert result.expense_ratio_coverage == 0.5
    assert result.weighted_expense_ratio == pytest.approx(0.0003)
    only_unknown = analyze_portfolio(
        [Holding(ticker="MYST", shares=1)], {"MYST": 1}, securities={"MYST": mystery}, config=CONFIG
    )
    assert only_unknown.weighted_expense_ratio is None and only_unknown.fee_drag == {}


def test_risk_score_without_history(base):
    # 0.6 * 6 (VTI) + 0.3 * 3 (BND) + 0.1 * 8 (AAPL) = 5.3
    assert (base.risk_score, base.risk_level) == (5.3, "Moderate")
    assert base.risk_metrics is None


@pytest.mark.parametrize(
    ("ticker", "level"), [("SGOV", "Low"), ("BND", "Low"), ("VTI", "High"), ("TSLA", "Very high")]
)
def test_risk_levels(ticker, level):
    result = analyze_portfolio(
        [Holding(ticker=ticker, shares=1)],
        {ticker: 1},
        securities=securities(ticker),
        config=CONFIG,
    )
    assert result.risk_level == level


def test_profile_gaps(base):
    gaps = {g.group: (g.current, g.target, round(g.difference, 4)) for g in base.profile_gaps}
    assert gaps == {
        "equity": (0.7, 0.6, 0.1),
        "bond": (0.3, 0.35, -0.05),
        "cash": (0.0, 0.05, -0.05),
    }


def test_profile_gaps_include_other_assets():
    result = analyze_portfolio(
        [Holding(ticker="GLD", shares=1)],
        {"GLD": 1},
        securities=securities("GLD"),
        profile=get_risk_profiles()["aggressive"],
        config=CONFIG,
    )
    assert {g.group: g.current for g in result.profile_gaps}["other"] == 1.0


def test_herfindahl():
    assert herfindahl([1.0]) == 1.0
    assert herfindahl([0.25] * 4) == 0.25


# ---- risk metrics ---------------------------------------------------------------------


def series(values, start="2026-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_risk_metrics_drawdown_and_constant_growth():
    prices = series([100, 120, 60, 90])
    metrics = risk_metrics(
        prices.pct_change().dropna(),
        benchmark=None,
        risk_free=rf(0.0),
        periods_per_year=252,
        min_observations=3,
    )
    assert metrics.max_drawdown == pytest.approx(-0.5)  # 120 -> 60
    assert metrics.observations == 3 and metrics.beta is None
    assert metrics.start == date(2026, 1, 2) and metrics.end == date(2026, 1, 6)

    steady = series([0.001] * 100)
    m = risk_metrics(
        steady, benchmark=None, risk_free=rf(0.02), periods_per_year=252, min_observations=10
    )
    assert m.annual_return == pytest.approx(1.001**252 - 1)
    assert m.annual_volatility == pytest.approx(0, abs=1e-12) and m.sharpe_ratio is None
    assert m.max_drawdown == 0


def test_beta_and_sharpe():
    bench = series([0.01, -0.01] * 50)
    port = bench * 2
    m = risk_metrics(
        port, benchmark=bench, risk_free=rf(0.0), periods_per_year=252, min_observations=10
    )
    assert m.beta == pytest.approx(2.0)
    assert m.sharpe_ratio == pytest.approx(m.annual_return / m.annual_volatility)
    flat_bench = series([0.0] * 100)
    assert (
        risk_metrics(
            port, benchmark=flat_bench, risk_free=rf(0), periods_per_year=252, min_observations=10
        ).beta
        is None
    )


def test_risk_metrics_needs_enough_data():
    assert (
        risk_metrics(
            series([0.01] * 5),
            benchmark=None,
            risk_free=rf(0),
            periods_per_year=252,
            min_observations=10,
        )
        is None
    )


def test_beta_skipped_when_overlap_too_short():
    port = series([0.01, -0.01] * 50)
    bench = series([0.01, -0.02, 0.03], start="2026-05-01")
    assert (
        risk_metrics(
            port, benchmark=bench, risk_free=rf(0), periods_per_year=252, min_observations=10
        ).beta
        is None
    )


def test_correlation_matrix():
    a = series([0.01, -0.02, 0.03, -0.01])
    returns = pd.DataFrame({"A": a, "B": a * 3, "C": -a, "CASH": 0.0})
    corr = correlation_matrix(returns)
    assert set(corr) == {"A", "B", "C"}  # constant cash is left out
    assert corr["A"]["B"] == 1.0 and corr["A"]["C"] == -1.0
    assert correlation_matrix(returns[["A"]]) is None
    assert correlation_matrix(pd.DataFrame()) is None


def test_daily_returns_aligns_dates():
    assert daily_returns({}).empty
    frame = daily_returns({"A": series([1, 2, 4]), "B": series([10, 11], start="2026-01-02")})
    assert list(frame.columns) == ["A", "B"] and len(frame) == 1
    assert frame.iloc[0]["A"] == 1.0  # 2 -> 4 on the one shared return date


def test_analysis_with_history_blends_volatility_into_risk_score():
    closes = {
        "VTI": series([100, 101, 99, 102, 103, 101]),
        "BND": series([50, 50.1, 50.05, 50.2, 50.1, 50.15]),
    }
    result = analyze_portfolio(
        [Holding(ticker="VTI", shares=1), Holding(ticker="BND", shares=2)],
        {"VTI": 100, "BND": 50},
        securities=securities("VTI", "BND"),
        closes=closes,
        benchmark_closes=closes["VTI"],
        config=CONFIG,
    )
    m = result.risk_metrics
    assert m.coverage == 1.0 and m.observations == 5 and m.beta is not None
    vol_score = min(10, max(1, 1 + m.annual_volatility * 25))
    assert result.risk_score == round(0.5 * (0.5 * 6 + 0.5 * 3) + 0.5 * vol_score, 1)
    assert set(result.correlation) == {"VTI", "BND"}


def test_history_for_unrelated_tickers_is_ignored():
    result = analyze_portfolio(
        BASE,
        PRICES,
        securities=securities("VTI", "BND", "AAPL"),
        closes={"MSFT": series([1, 2, 3, 4])},
        config=CONFIG,
    )
    assert result.risk_metrics is None


# ---- observations ---------------------------------------------------------------------


def test_concentrated_tech_observations():
    # TSLA 8/11 = 73%, NVDA 1/11 = 9%, IBIT 2/11 = 18%
    holdings = [
        Holding(ticker="TSLA", shares=8),
        Holding(ticker="NVDA", shares=1),
        Holding(ticker="IBIT", shares=2),
    ]
    result = analyze_portfolio(
        holdings,
        {"TSLA": 100, "NVDA": 100, "IBIT": 100},
        securities=securities("TSLA", "NVDA", "IBIT"),
        profile=get_risk_profiles()["conservative"],
        config=CONFIG,
    )
    text = " ".join(result.observations)
    assert result.concentrated_holdings == ["TSLA"]
    assert "TSLA is 73% of this portfolio" in text and "concentration risk" in text
    assert "About 73% of the portfolio is in Consumer Discretionary" in text
    assert result.diversification_score == 29.7
    assert "diversification score is 30/100, which is low" in text
    assert "Crypto assets are 18%" in text
    assert "Stocks are 82% of the portfolio, more than the 30%" in text
    assert "Not enough price history" in text
    for phrase in ("you should", "sell", "buy "):
        assert phrase not in text.lower()


def test_broad_portfolio_gets_positive_note_and_less_stock_note():
    result = analyze_portfolio(
        [Holding(ticker="VTI", shares=3), Holding(ticker="BND", shares=7)],
        {"VTI": 1, "BND": 1},
        securities=securities("VTI", "BND"),
        profile=get_risk_profiles()["aggressive"],
        config=CONFIG,
    )
    text = " ".join(result.observations)
    assert "Broad funds and a mix of asset types" in text
    assert "less than the 90%" in text


def test_high_fee_unknown_and_missing_price_notes():
    pricey = SecurityInfo(
        ticker="PRCY",
        name="Pricey Active Fund",
        type="mutual_fund",
        asset_class="equity",
        sector="Broad Market",
        diversified=True,
        fees=ExpenseRatio(ratio=0.012, status="VERIFY", source_url="https://example.com/prcy"),
        risk=6,
    )
    newco = CATALOG.classify("NEWCO", asset_type="Common Stock")
    holdings = [
        Holding(ticker="PRCY", shares=9),
        Holding(ticker="NEWCO", shares=1),
        Holding(ticker="GONE", shares=1),
    ]
    result = analyze_portfolio(
        holdings,
        {"PRCY": 10, "NEWCO": 10},
        securities={"PRCY": pricey, "NEWCO": newco},
        config=CONFIG,
    )
    text = " ".join(result.observations)
    assert "portfolio expense ratio (funds only) is about 1.20%" in text
    assert "expense ratio for PRCY couldn't be confirmed" in text
    assert "NEWCO isn't in Finnie's reference list" in text
    assert "No price was available for GONE, so it is left out" in text
    assert result.missing_prices == ["GONE"]


def test_multiple_missing_prices_wording():
    result = analyze_portfolio(
        [*BASE, Holding(ticker="X", shares=1), Holding(ticker="Y", shares=1)],
        PRICES,
        securities=securities("VTI", "BND", "AAPL"),
        config=CONFIG,
    )
    assert any("X, Y, so they are left out" in n for n in result.observations)


def test_errors_for_empty_or_unpriced_portfolios():
    with pytest.raises(PortfolioError, match="no holdings"):
        analyze_portfolio([], {}, securities={})
    with pytest.raises(PortfolioError, match="No prices"):
        analyze_portfolio(BASE, {}, securities=securities("VTI", "BND", "AAPL"))


# ---- input parsing --------------------------------------------------------------------


def test_merge_holdings():
    merged = merge_holdings(
        [
            Holding(ticker="VTI", shares=1, cost_basis=10),
            Holding(ticker="vti", shares=2, cost_basis=20),
            Holding(ticker="BND", shares=1, cost_basis=5),
            Holding(ticker="BND", shares=1),
        ]
    )
    assert [(h.ticker, h.shares, h.cost_basis) for h in merged] == [
        ("VTI", 3, 30),
        ("BND", 2, None),
    ]


def test_holdings_from_csv():
    text = (
        'Ticker, Shares ,Cost_Basis\nvti,10,"1,200.50"\nBND,5,\n,,\n'
        "VTI,2,100\nBAD!,1,\nAAPL,-3,\nMSFT,abc,\n"
    )
    holdings, errors = holdings_from_csv(text)
    assert [(h.ticker, h.shares, h.cost_basis) for h in holdings] == [
        ("VTI", 12, 1300.5),
        ("BND", 5, None),
    ]
    assert len(errors) == 3
    assert errors[0].startswith("Row 6 (BAD!)") and "Row 7 (AAPL)" in errors[1]
    assert "Row 8 (MSFT)" in errors[2]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("symbol,qty\nVTI,1\n", "Missing required column(s): shares, ticker"),
        ("ticker,shares\n", "No holdings found"),
    ],
)
def test_holdings_from_csv_errors(text, message):
    holdings, errors = holdings_from_csv(text)
    assert holdings == [] and message in errors[0]


def test_sample_portfolios_parse_cleanly():
    folder = Path(__file__).parents[3] / "data" / "sample_portfolios"
    files = sorted(folder.glob("*.csv"))
    assert len(files) == 3
    for path in files:
        holdings, errors = holdings_from_csv(path.read_text(encoding="utf-8"))
        assert errors == [] and holdings, path.name
        assert all(h.ticker in CATALOG for h in holdings), path.name


# ---- fetch_and_analyze ----------------------------------------------------------------


def fresh(source="yfinance", mock=False):
    return Freshness(source=source, as_of=START, fetched_at=START, is_mock=mock)


def history(ticker, closes):
    days = pd.bdate_range("2026-01-01", periods=len(closes))
    bars = [
        PriceBar(date=d.date(), open=c, high=c, low=c, close=c)
        for d, c in zip(days, closes, strict=True)
    ]
    return PriceHistory(ticker=ticker, bars=bars, adjusted=True, freshness=fresh())


class FakeMarket:
    def __init__(self, prices, histories=None, overviews=None, mock=(), tbill=None):
        self.prices = prices
        self.histories = histories or {}
        self.overviews = overviews or {}
        self.mock = set(mock)
        self.quote_requests = []
        self.tbill = tbill

    def get_quotes(self, tickers):
        tickers = list(tickers)
        self.quote_requests.append(tickers)
        quotes = {
            t: Quote(ticker=t, price=self.prices[t], freshness=fresh(mock=t in self.mock))
            for t in tickers
            if t in self.prices
        }
        return BatchQuotes(
            quotes=quotes, errors={t: "nope" for t in tickers if t not in self.prices}
        )

    def get_daily_history(self, ticker, days=252):
        if ticker not in self.histories:
            raise DataUnavailableError(ticker)
        return self.histories[ticker]

    def get_treasury_bill_yield(self):
        if self.tbill is None:
            raise DataUnavailableError("^IRX")
        return Quote(ticker="^IRX", price=self.tbill, freshness=fresh())

    def get_company_overview(self, ticker):
        if ticker not in self.overviews:
            raise SymbolNotFoundError(ticker)
        return self.overviews[ticker]


def test_fetch_and_analyze_end_to_end():
    closes = [100 + (i % 5) for i in range(80)]
    market = FakeMarket(
        prices={"VTI": 100, "NEWCO": 50},
        histories={
            "VTI": history("VTI", closes),
            "SPY": history("SPY", closes),
            "NEWCO": history("NEWCO", [50 + (i % 3) for i in range(80)]),
        },
        overviews={
            "NEWCO": CompanyOverview(
                ticker="NEWCO",
                name="New Co",
                asset_type="Equity",
                sector="Energy",
                freshness=fresh(),
            )
        },
        tbill=4.21,
    )
    holdings = [
        Holding(ticker="VTI", shares=5),
        Holding(ticker="NEWCO", shares=2),
        Holding(ticker="CASH", shares=250),
    ]
    result = fetch_and_analyze(
        holdings,
        market,
        profile=get_risk_profiles()["moderate"],
        config=AnalyticsConfig(min_history_days=20),
    )
    assert market.quote_requests == [["VTI", "NEWCO"]]  # cash is never quoted
    assert result.total_value == 850
    newco = next(h for h in result.holdings if h.ticker == "NEWCO")
    assert (newco.name, newco.sector, newco.classification_known) == ("New Co", "Energy", False)
    assert result.asset_allocation["cash"] == pytest.approx(250 / 850, abs=1e-4)
    metrics = result.risk_metrics
    assert metrics is not None and metrics.beta is not None
    # the Sharpe ratio uses the live T-bill yield and says where it came from
    assert metrics.risk_free.rate == pytest.approx(0.0421) and not metrics.risk_free.is_fallback
    assert metrics.risk_free.as_of == START.date()
    assert metrics.sharpe_ratio == pytest.approx(
        (metrics.annual_return - 0.0421) / metrics.annual_volatility
    )
    assert "CASH" not in (result.correlation or {})
    assert len(result.freshness) == 2


def test_sharpe_falls_back_to_configured_rate():
    closes = [100 + (i % 5) for i in range(80)]
    market = FakeMarket(prices={"VTI": 100}, histories={"VTI": history("VTI", closes)})
    config = AnalyticsConfig(min_history_days=20, risk_free_rate=0.05)
    metrics = fetch_and_analyze(
        [Holding(ticker="VTI", shares=1)], market, config=config
    ).risk_metrics
    assert metrics.risk_free.is_fallback and metrics.risk_free.rate == 0.05
    assert "live T-bill yield unavailable" in metrics.risk_free.label()
    assert metrics.sharpe_ratio == pytest.approx(
        (metrics.annual_return - 0.05) / metrics.annual_volatility
    )


def test_analyze_without_supplied_rate_uses_config_fallback():
    closes = {"VTI": series([100, 101, 99, 102, 103, 101])}
    result = analyze_portfolio(
        [Holding(ticker="VTI", shares=1)],
        {"VTI": 100},
        securities=securities("VTI"),
        closes=closes,
        config=CONFIG,
    )
    assert result.risk_metrics.risk_free.is_fallback
    assert result.risk_metrics.risk_free.rate == 0.042


def test_fetch_and_analyze_degrades_gracefully():
    market = FakeMarket(prices={"VTI": 100, "MYST": 5}, mock={"VTI"})
    result = fetch_and_analyze(
        [
            Holding(ticker="VTI", shares=1),
            Holding(ticker="MYST", shares=2),
            Holding(ticker="GONE", shares=1),
        ],
        market,
        config=CONFIG,
    )
    text = " ".join(result.observations)
    assert result.missing_prices == ["GONE"] and result.risk_metrics is None
    assert "MYST isn't in Finnie's reference list" in text
    assert "demo data" in text


def test_fetch_and_analyze_without_history():
    market = FakeMarket(prices={"VTI": 100}, histories={"VTI": history("VTI", [1, 2, 3])})
    result = fetch_and_analyze(
        [Holding(ticker="VTI", shares=1)], market, config=CONFIG, include_history=False
    )
    assert result.risk_metrics is None


def test_fetch_and_analyze_uses_global_defaults_and_rejects_empty():
    market = FakeMarket(prices={"VTI": 100})
    assert fetch_and_analyze([Holding(ticker="VTI", shares=1)], market).total_value == 100
    with pytest.raises(PortfolioError):
        fetch_and_analyze([], market)


# ---- portfolio_total ------------------------------------------------------------------


def test_portfolio_total_from_quotes_and_cash():
    from src.core.portfolio import portfolio_total

    market = FakeMarket({"VTI": 300.0})
    holdings = [Holding(ticker="VTI", shares=2), Holding(ticker="CASH", shares=50)]
    assert portfolio_total(holdings, market) == 650.0
    assert portfolio_total([Holding(ticker="ZZZZ", shares=1)], market) is None


def test_portfolio_total_when_quotes_fail():
    from src.core.portfolio import portfolio_total
    from src.data.errors import DataUnavailableError

    class Down:
        def get_quotes(self, tickers):
            raise DataUnavailableError("down")

    assert portfolio_total([Holding(ticker="VTI", shares=1)], Down()) is None  # type: ignore[arg-type]
    assert portfolio_total([Holding(ticker="CASH", shares=5)], Down()) == 5.0  # type: ignore[arg-type]


# ---- back-test and fee label ----------------------------------------------------------


def test_backtest_holds_todays_shares_and_starts_at_100():
    closes = {"VTI": series([100, 110, 120]), "BND": series([50, 50, 55])}
    bench = series([200, 220, 180])
    result = backtest(
        closes, {"VTI": 1, "BND": 2}, bench, benchmark_ticker="SPY", days=252, coverage=1.0
    )
    # values 200, 210, 230 -> 100, 105, 115
    assert result.portfolio == [100.0, 105.0, 115.0]
    assert result.benchmark == [100.0, 110.0, 90.0]
    assert result.dates[0].isoformat() == "2026-01-01"
    window = backtest(closes, {"VTI": 1}, None, benchmark_ticker="SPY", days=1, coverage=0.5)
    assert window.portfolio == [100.0, 109.09]  # only the last day's move: 110 -> 120
    assert window.benchmark is None
    assert (
        backtest({"VTI": series([1])}, {"VTI": 1}, None, benchmark_ticker="SPY", days=5, coverage=1)
        is None
    )


def test_backtest_benchmark_gaps_are_filled_and_bad_benchmarks_dropped():
    closes = {"VTI": series([100, 110, 120])}
    gappy = pd.Series([200.0, 210.0], index=series([1, 2]).index[[0, 2 - 1]])
    filled = backtest(closes, {"VTI": 1}, gappy, benchmark_ticker="SPY", days=10, coverage=1)
    assert filled.benchmark == [100.0, 105.0, 105.0]  # carried forward
    zero = series([0, 1, 2])
    assert (
        backtest(closes, {"VTI": 1}, zero, benchmark_ticker="SPY", days=10, coverage=1).benchmark
        is None
    )


def test_analysis_includes_the_backtest():
    closes = {"VTI": series([100, 101, 102, 103]), "BND": series([50, 50, 51, 51])}
    result = analyze_portfolio(
        [Holding(ticker="VTI", shares=1), Holding(ticker="BND", shares=2)],
        {"VTI": 103, "BND": 51},
        securities=securities("VTI", "BND"),
        closes=closes,
        benchmark_closes=closes["VTI"],
        config=CONFIG,
    )
    assert result.backtest.benchmark_ticker == "SPY"
    assert result.backtest.portfolio[0] == 100.0 and len(result.backtest.dates) == 4


def test_expense_ratio_label_groups_funds_and_mentions_stocks():
    result = analyze_portfolio(
        BASE, PRICES, securities=securities("VTI", "BND", "AAPL"), config=CONFIG
    )
    label = expense_ratio_label(result)
    assert label.startswith("Portfolio expense ratio: 0.03% (funds only; ")
    assert "VTI and BND charge 0.03% each" in label
    assert label.endswith("individual stocks have no expense ratio).")
    stocks = analyze_portfolio(
        [Holding(ticker="AAPL", shares=1)], PRICES, securities=securities("AAPL"), config=CONFIG
    )
    assert expense_ratio_label(stocks).startswith("Portfolio expense ratio: none")
    funds = analyze_portfolio(
        [Holding(ticker="VTI", shares=1), Holding(ticker="VXUS", shares=1)],
        {"VTI": 100, "VXUS": 60},
        securities=securities("VTI", "VXUS"),
        config=CONFIG,
    )
    text = expense_ratio_label(funds)
    assert "VTI charges 0.03%" in text and "individual stocks" not in text
    # the headline is the value-weighted average, not the last fund's fee
    # (regression: the label once showed VXUS's 0.05% for a 0.04% portfolio)
    weighted = (100 * 0.0003 + 60 * 0.0005) / 160
    assert text.startswith(f"Portfolio expense ratio: {weighted:.2%} (funds only; ")
    assert text.endswith("VTI charges 0.03%, VXUS charges 0.05%).")


# ---- persona testing fixes ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        ("AAPL,-3,", "shares must be more than 0"),
        ("AAPL,0,", "shares must be more than 0"),
        ("TSLA,1000000000000,", "shares can't be more than 1,000,000,000 (check for extra zeros)"),
        ("AAPL,2,-50", "total cost can't be negative"),
        ("NVDA,abc,", "shares and cost must be numbers"),
    ],
)
def test_holdings_from_csv_explains_rejected_rows(row, reason):
    holdings, errors = holdings_from_csv(f"ticker,shares,cost_basis\n{row}\n")
    assert holdings == [] and errors[0].endswith(reason)


def test_holding_error_for_other_problems():
    try:
        Holding(ticker="AAPL!", shares=1)
    except ValueError as exc:
        reason = holding_error(exc)
    assert "Invalid ticker" in reason


def test_expense_ratio_label_discloses_funds_with_unknown_fees():
    unknown = {"NEWF": CATALOG.classify("NEWF", name="New Fund", asset_type="ETF")}
    prices = PRICES | {"NEWF": 20.0}
    mixed = analyze_portfolio(
        [Holding(ticker="VTI", shares=1), Holding(ticker="NEWF", shares=5)],
        prices,
        securities=securities("VTI", **unknown),
        config=CONFIG,
    )
    label = expense_ratio_label(mixed)
    assert label.startswith("Portfolio expense ratio: 0.03% (funds with a known fee only; ")
    assert "fee not known for NEWF, so the true figure may be higher" in label
    only_unknown = analyze_portfolio(
        [Holding(ticker="NEWF", shares=5)], prices, securities=unknown, config=CONFIG
    )
    assert expense_ratio_label(only_unknown) == (
        "Portfolio expense ratio: unknown (Finnie doesn't have the fee for NEWF; check the "
        "fund's prospectus)."
    )
    two = {"NEWG": CATALOG.classify("NEWG", asset_type="ETF")} | unknown
    both = analyze_portfolio(
        [Holding(ticker="NEWF", shares=5), Holding(ticker="NEWG", shares=5)],
        prices | {"NEWG": 10.0},
        securities=two,
        config=CONFIG,
    )
    assert "the fee for NEWF and NEWG" in expense_ratio_label(both)


# ---- multi-asset funds and funds whose mix is unknown (persona testing) ---------------------


def test_target_date_funds_count_their_real_stock_bond_mix():
    vffvx = CATALOG.get("VFFVX")
    assert vffvx.look_through == {"equity": 0.912, "bond": 0.088}
    assert vffvx.expense_ratio == 0.0008
    result = analyze_portfolio(
        [Holding(ticker="VFFVX", shares=10)],
        {"VFFVX": 50.0},
        securities=securities("VFFVX"),
        config=CONFIG,
    )
    assert result.asset_allocation == {"equity": 0.912, "bond": 0.088}


def test_unknown_funds_are_mix_unknown_not_stocks():
    unknown = {"NEWF": CATALOG.classify("NEWF", name="New Target 2050", asset_type="Mutual Fund")}
    result = analyze_portfolio(
        [Holding(ticker="VTI", shares=1), Holding(ticker="NEWF", shares=10)],
        PRICES | {"NEWF": 30.0},
        securities=securities("VTI", **unknown),
        profile=get_risk_profiles()["moderate"],
        config=CONFIG,
    )
    assert result.asset_allocation == {"equity": 0.25, "unknown_mix": 0.75}
    note = next(n for n in result.observations if "stock/bond mix" in n)
    assert note.startswith("Finnie doesn't know the stock/bond mix of NEWF (75% of the portfolio)")
    assert not any("isn't in Finnie's reference list" in n for n in result.observations)
    groups = {g.group: g.current for g in result.profile_gaps}
    assert groups["unknown_mix"] == 0.75 and groups["equity"] == 0.25


def test_stock_reference_note_says_horizon_matters_too():
    result = analyze_portfolio(
        [Holding(ticker="AAPL", shares=10)],
        PRICES,
        securities=securities("AAPL"),
        profile=get_risk_profiles()["conservative"],
        config=CONFIG,
    )
    note = next(n for n in result.observations if "reference allocation" in n)
    assert note.endswith(
        "That reference is based on risk tolerance only; time horizon matters too."
    )


def test_asset_class_labels():
    assert asset_class_label("equity") == "Stocks"
    assert asset_class_label("unknown_mix") == "Mix unknown"
    assert asset_class_label("other") == "Other"
    assert asset_class_label("brand_new") == "Brand new"
