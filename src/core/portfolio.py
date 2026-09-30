"""Portfolio analytics.

``analyze_portfolio`` is a pure function over holdings, prices, classifications, and
(optionally) price histories, so every metric can be tested against hand-computed values.
``fetch_and_analyze`` gathers that input from the market data service and calls it.

Results are framed for education: ``observations`` describe what the numbers mean and
never tell the user to buy or sell anything.
"""

from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import Protocol

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from src.core.config import AnalyticsConfig, get_settings
from src.core.models import Freshness, Holding
from src.core.reference import RiskProfile, SecurityCatalog, SecurityInfo, get_catalog
from src.data.errors import MarketDataError
from src.data.models import BatchQuotes, CompanyOverview, PriceHistory

SECTOR_COUNT = 11  # GICS sectors
BROAD_SECTORS = frozenset({"Broad Market", "International"})
NON_EQUITY_SECTORS = frozenset({"Fixed Income", "Cash", "Commodities", "Crypto", "Real Estate"})
PROFILE_GROUPS = ("equity", "bond", "cash", "other")
FEE_DRAG_YEARS = (10, 20, 30)
CASH_TYPES = frozenset({"cash", "money_market"})

# Diversification score weights (see diversification_score)
W_CONCENTRATION, W_ASSET_CLASS, W_SECTOR = 0.4, 0.3, 0.3


class PortfolioError(ValueError):
    """The portfolio can't be analyzed (empty, or no prices at all)."""


class HoldingValuation(BaseModel):
    ticker: str
    name: str
    shares: float
    price: float
    value: float
    weight: float
    type: str
    asset_class: str
    sector: str
    diversified: bool
    expense_ratio: float | None
    classification_known: bool
    cost_basis: float | None = None
    unrealized_gain: float | None = None
    unrealized_gain_pct: float | None = None


class RiskMetrics(BaseModel):
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float | None
    max_drawdown: float
    beta: float | None
    observations: int
    start: date
    end: date
    coverage: float = Field(description="Share of portfolio value with price history")


class AllocationGap(BaseModel):
    group: str
    current: float
    target: float

    @property
    def difference(self) -> float:
        return self.current - self.target


class PortfolioAnalysis(BaseModel):
    total_value: float
    holdings: list[HoldingValuation]
    asset_allocation: dict[str, float]
    sector_allocation: dict[str, float]
    hhi: float
    effective_holdings: float
    diversification_score: float
    diversification_components: dict[str, float]
    weighted_expense_ratio: float | None
    expense_ratio_coverage: float
    annual_fees: float | None
    fee_drag: dict[int, float]
    risk_score: float
    risk_level: str
    concentrated_holdings: list[str]
    total_cost_basis: float | None
    unrealized_gain: float | None
    unrealized_gain_pct: float | None
    risk_metrics: RiskMetrics | None
    correlation: dict[str, dict[str, float]] | None
    profile: str | None
    profile_gaps: list[AllocationGap]
    observations: list[str]
    missing_prices: list[str]
    freshness: list[Freshness] = Field(default_factory=list)


# ---- input helpers --------------------------------------------------------------------


def merge_holdings(holdings: Iterable[Holding]) -> list[Holding]:
    """Combine repeated tickers. Cost basis is summed, or dropped if any lot lacks one."""
    shares: dict[str, float] = defaultdict(float)
    basis: dict[str, float | None] = {}
    for h in holdings:
        shares[h.ticker] += h.shares
        if h.ticker not in basis:
            basis[h.ticker] = h.cost_basis
        else:
            prior = basis[h.ticker]
            basis[h.ticker] = (
                None if prior is None or h.cost_basis is None else prior + h.cost_basis
            )
    return [Holding(ticker=t, shares=shares[t], cost_basis=basis[t]) for t in shares]


def holdings_from_csv(text: str) -> tuple[list[Holding], list[str]]:
    """Parse ``ticker,shares[,cost_basis]`` CSV text. Returns (holdings, row errors)."""
    reader = csv.DictReader(io.StringIO(text.strip()))
    if not reader.fieldnames:
        return [], ["The file is empty."]
    columns = {name.strip().lower(): name for name in reader.fieldnames}
    missing = {"ticker", "shares"} - set(columns)
    if missing:
        return [], [f"Missing required column(s): {', '.join(sorted(missing))}."]

    holdings: list[Holding] = []
    errors: list[str] = []
    for row_number, row in enumerate(reader, start=2):
        ticker = (row.get(columns["ticker"]) or "").strip()
        if not ticker:
            continue
        basis_raw = (
            (row.get(columns["cost_basis"]) or "").strip() if "cost_basis" in columns else ""
        )
        try:
            holdings.append(
                Holding(
                    ticker=ticker,
                    shares=float((row.get(columns["shares"]) or "").replace(",", "")),
                    cost_basis=float(basis_raw.replace(",", "")) if basis_raw else None,
                )
            )
        except (ValueError, ValidationError) as exc:
            reason = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
            errors.append(f"Row {row_number} ({ticker}): {reason}")
    if not holdings and not errors:
        errors.append("No holdings found.")
    return merge_holdings(holdings), errors


# ---- core metrics ---------------------------------------------------------------------


def value_holdings(
    holdings: Sequence[Holding],
    prices: Mapping[str, float],
    securities: Mapping[str, SecurityInfo],
) -> tuple[list[HoldingValuation], list[str]]:
    """Value each holding. Tickers without a price are returned separately."""
    priced = [h for h in merge_holdings(holdings) if prices.get(h.ticker)]
    missing = [h.ticker for h in merge_holdings(holdings) if not prices.get(h.ticker)]
    total = sum(h.shares * prices[h.ticker] for h in priced)
    rows = []
    for h in priced:
        info = securities[h.ticker]
        value = h.shares * prices[h.ticker]
        gain = value - h.cost_basis if h.cost_basis is not None else None
        rows.append(
            HoldingValuation(
                ticker=h.ticker,
                name=info.name,
                shares=h.shares,
                price=prices[h.ticker],
                value=value,
                weight=value / total,
                type=info.type,
                asset_class=info.asset_class,
                sector=info.sector,
                diversified=info.diversified,
                expense_ratio=_expense_ratio(info),
                classification_known=info.known,
                cost_basis=h.cost_basis,
                unrealized_gain=gain,
                unrealized_gain_pct=gain / h.cost_basis
                if gain is not None and h.cost_basis
                else None,
            )
        )
    rows.sort(key=lambda r: r.value, reverse=True)
    return rows, missing


def _expense_ratio(info: SecurityInfo) -> float | None:
    """Stocks and cash have no fund fee; unknown funds have an unknown one."""
    if info.type in ("stock", "cash"):
        return 0.0
    return info.expense_ratio


def asset_allocation(
    rows: Sequence[HoldingValuation], securities: Mapping[str, SecurityInfo]
) -> dict[str, float]:
    mix: dict[str, float] = defaultdict(float)
    for row in rows:
        for asset_class, share in securities[row.ticker].look_through.items():
            mix[asset_class] += row.weight * share
    return _sorted(mix)


def sector_allocation(rows: Sequence[HoldingValuation]) -> dict[str, float]:
    mix: dict[str, float] = defaultdict(float)
    for row in rows:
        mix[row.sector] += row.weight
    return _sorted(mix)


def herfindahl(weights: Iterable[float]) -> float:
    """Sum of squared weights: 1.0 for a single holding, 1/n for n equal holdings."""
    return sum(w * w for w in weights)


def diversification_score(
    rows: Sequence[HoldingValuation],
    allocation: Mapping[str, float],
    securities: Mapping[str, SecurityInfo],
) -> tuple[float, dict[str, float]]:
    """A 0-100 educational diversification score made of three parts.

    - Company concentration (40%): ``1 - sqrt(sum of squared weights of single-company
      holdings)``. Broad funds hold many companies, so they add no single-company risk.
    - Asset-class mix (30%): ``1 - HHI`` across asset classes, scaled so an even three-way
      stock/bond/cash split scores 1.
    - Sector spread (30%): ``1 - HHI`` across sectors of the stock portion, scaled so an
      even spread over the 11 sectors scores 1. Broad funds count as evenly spread.
    """
    single = [r.weight for r in rows if securities[r.ticker].is_single_issuer]
    concentration = 1 - math.sqrt(herfindahl(single))
    asset_mix = min(1.0, (1 - herfindahl(allocation.values())) / (1 - 1 / 3))

    sectors: dict[str, float] = defaultdict(float)
    for row in rows:
        equity = row.weight * securities[row.ticker].look_through.get("equity", 0.0)
        if equity <= 0:
            continue
        if row.sector in BROAD_SECTORS:
            for i in range(SECTOR_COUNT):
                sectors[f"_broad_{i}"] += equity / SECTOR_COUNT
        else:
            sectors[row.sector] += equity
    equity_total = sum(sectors.values())
    if equity_total > 0:
        shares = [w / equity_total for w in sectors.values()]
        sector = min(1.0, (1 - herfindahl(shares)) / (1 - 1 / SECTOR_COUNT))
    else:
        sector = 1.0  # no stocks, so no sector concentration

    components = {
        "concentration": round(concentration, 4),
        "asset_class": round(asset_mix, 4),
        "sector": round(sector, 4),
    }
    score = 100 * (W_CONCENTRATION * concentration + W_ASSET_CLASS * asset_mix + W_SECTOR * sector)
    return round(score, 1), components


def expense_summary(
    rows: Sequence[HoldingValuation], total_value: float, growth: float
) -> tuple[float | None, float, float | None, dict[int, float]]:
    """Weighted expense ratio, coverage, annual fee dollars, and fee drag over time.

    Fee drag is the gap after N years between growing at ``growth`` and growing at
    ``growth - expense ratio``, starting from today's value with no new contributions.
    """
    known = [(r, r.expense_ratio) for r in rows if r.expense_ratio is not None]
    coverage = sum(r.weight for r, _ in known)
    if coverage == 0:
        return None, 0.0, None, {}
    ratio = sum(r.weight * er for r, er in known) / coverage
    annual = sum(r.value * er for r, er in known)
    drag = {
        years: round(total_value * ((1 + growth) ** years - (1 + growth - ratio) ** years), 2)
        for years in FEE_DRAG_YEARS
    }
    return ratio, coverage, annual, drag


def risk_score(
    rows: Sequence[HoldingValuation],
    securities: Mapping[str, SecurityInfo],
    annual_volatility: float | None,
) -> tuple[float, str]:
    """1-10 score: holding risk ratings, blended 50/50 with realized volatility when known."""
    base = sum(r.weight * securities[r.ticker].risk for r in rows)
    if annual_volatility is not None:
        vol_score = min(10.0, max(1.0, 1 + annual_volatility * 25))
        base = 0.5 * base + 0.5 * vol_score
    score = round(base, 1)
    if score < 3.5:
        level = "Low"
    elif score < 5.5:
        level = "Moderate"
    elif score < 7.5:
        level = "High"
    else:
        level = "Very high"
    return score, level


def daily_returns(closes: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Daily returns for each ticker on the dates all of them traded."""
    if not closes:
        return pd.DataFrame()
    frame = pd.concat(closes, axis=1).sort_index().dropna()
    return frame.pct_change().dropna()


def risk_metrics(
    returns: pd.Series,
    *,
    benchmark: pd.Series | None,
    risk_free_rate: float,
    periods_per_year: int,
    min_observations: int,
    coverage: float = 1.0,
) -> RiskMetrics | None:
    """Annualized return and volatility, Sharpe ratio, max drawdown, and beta."""
    returns = returns.dropna()
    count = len(returns)
    if count < min_observations:
        return None
    growth = float((1 + returns).prod())
    annual_return = growth ** (periods_per_year / count) - 1
    annual_vol = float(returns.std(ddof=1)) * math.sqrt(periods_per_year)
    # A constant series can leave ~1e-18 of floating-point noise instead of exactly 0.
    if annual_vol < 1e-12:
        annual_vol = 0.0
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else None

    wealth = pd.concat([pd.Series([1.0]), (1 + returns).cumprod().reset_index(drop=True)])
    max_drawdown = float((wealth / wealth.cummax() - 1).min())

    beta = None
    if benchmark is not None:
        paired = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
        if len(paired) >= min_observations:
            variance = float(paired.iloc[:, 1].var(ddof=1))
            if variance > 0:
                beta = float(paired.iloc[:, 0].cov(paired.iloc[:, 1])) / variance

    return RiskMetrics(
        annual_return=annual_return,
        annual_volatility=annual_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        beta=beta,
        observations=count,
        start=returns.index[0].date(),
        end=returns.index[-1].date(),
        coverage=coverage,
    )


def correlation_matrix(returns: pd.DataFrame) -> dict[str, dict[str, float]] | None:
    """Pairwise correlation of daily returns; constant series (e.g. cash) are left out."""
    varying = returns.loc[:, returns.std(ddof=1) > 0] if not returns.empty else returns
    if varying.shape[1] < 2:
        return None
    corr = varying.corr().round(2)
    return {row: {col: float(corr.loc[row, col]) for col in corr.columns} for row in corr.index}


def profile_gaps(allocation: Mapping[str, float], profile: RiskProfile) -> list[AllocationGap]:
    current = {group: 0.0 for group in PROFILE_GROUPS}
    for asset_class, weight in allocation.items():
        current[asset_class if asset_class in current else "other"] += weight
    gaps = []
    for group in PROFILE_GROUPS:
        target = profile.allocation.get(group, 0.0)  # type: ignore[call-overload]
        if group == "other" and current[group] == 0 and target == 0:
            continue
        gaps.append(AllocationGap(group=group, current=round(current[group], 4), target=target))
    return gaps


# ---- analysis -------------------------------------------------------------------------


def analyze_portfolio(
    holdings: Sequence[Holding],
    prices: Mapping[str, float],
    *,
    securities: Mapping[str, SecurityInfo],
    closes: Mapping[str, pd.Series] | None = None,
    benchmark_closes: pd.Series | None = None,
    profile: RiskProfile | None = None,
    config: AnalyticsConfig | None = None,
) -> PortfolioAnalysis:
    """Analyze a portfolio from already-fetched inputs. See module docstring."""
    config = config or AnalyticsConfig()
    if not holdings:
        raise PortfolioError("The portfolio has no holdings.")
    rows, missing = value_holdings(holdings, prices, securities)
    if not rows:
        raise PortfolioError("No prices were available for any holding.")
    total = sum(r.value for r in rows)
    weights = {r.ticker: r.weight for r in rows}

    allocation = asset_allocation(rows, securities)
    div_score, components = diversification_score(rows, allocation, securities)
    ratio, coverage, fees, drag = expense_summary(rows, total, config.fee_growth_rate)

    metrics = None
    correlation = None
    if closes:
        available = {t: s for t, s in closes.items() if t in weights}
        returns = daily_returns(available)
        if not returns.empty:
            covered = sum(weights[t] for t in returns.columns)
            portfolio = returns @ pd.Series({t: weights[t] / covered for t in returns.columns})
            bench = (
                benchmark_closes.sort_index().pct_change().dropna()
                if benchmark_closes is not None
                else None
            )
            metrics = risk_metrics(
                portfolio,
                benchmark=bench,
                risk_free_rate=config.risk_free_rate,
                periods_per_year=config.trading_days_per_year,
                min_observations=config.min_history_days,
                coverage=round(covered, 4),
            )
            correlation = correlation_matrix(returns)

    score, level = risk_score(rows, securities, metrics.annual_volatility if metrics else None)
    concentrated = [
        r.ticker
        for r in rows
        if securities[r.ticker].is_single_issuer and r.weight > config.concentration_threshold
    ]
    basis_rows = [r for r in rows if r.cost_basis is not None]
    total_basis = sum(r.cost_basis for r in basis_rows) if basis_rows else None  # type: ignore[misc]
    gain = sum(r.unrealized_gain for r in basis_rows) if basis_rows else None  # type: ignore[misc]
    gaps = profile_gaps(allocation, profile) if profile else []
    hhi = herfindahl(weights.values())

    analysis = PortfolioAnalysis(
        total_value=round(total, 2),
        holdings=rows,
        asset_allocation=allocation,
        sector_allocation=sector_allocation(rows),
        hhi=round(hhi, 4),
        effective_holdings=round(1 / hhi, 2),
        diversification_score=div_score,
        diversification_components=components,
        weighted_expense_ratio=ratio,
        expense_ratio_coverage=round(coverage, 4),
        annual_fees=round(fees, 2) if fees is not None else None,
        fee_drag=drag,
        risk_score=score,
        risk_level=level,
        concentrated_holdings=concentrated,
        total_cost_basis=total_basis,
        unrealized_gain=gain,
        unrealized_gain_pct=gain / total_basis if gain is not None and total_basis else None,
        risk_metrics=metrics,
        correlation=correlation,
        profile=profile.name if profile else None,
        profile_gaps=gaps,
        observations=[],
        missing_prices=missing,
    )
    analysis.observations = build_observations(analysis, profile, config)
    return analysis


def build_observations(
    analysis: PortfolioAnalysis, profile: RiskProfile | None, config: AnalyticsConfig
) -> list[str]:
    """Plain-English, educational notes about the analysis. Never buy/sell directives."""
    notes: list[str] = []
    by_ticker = {r.ticker: r for r in analysis.holdings}

    for ticker in analysis.concentrated_holdings:
        notes.append(
            f"{ticker} is {by_ticker[ticker].weight:.0%} of this portfolio. When one company is "
            "this large a share, its ups and downs drive most of the portfolio's results. "
            "This is called concentration risk."
        )
    for sector, weight in analysis.sector_allocation.items():
        if (
            sector not in BROAD_SECTORS | NON_EQUITY_SECTORS | {"Unknown"}
            and weight > config.sector_concentration_threshold
        ):
            notes.append(
                f"About {weight:.0%} of the portfolio is in {sector}. Companies in one sector "
                "often move together, so a sector-wide slump would affect much of it at once."
            )

    score = analysis.diversification_score
    if score < 40:
        notes.append(
            f"The diversification score is {score:.0f}/100, which is low. Spreading money across "
            "more companies, sectors, and asset types generally reduces the impact of any one "
            "investment doing badly."
        )
    elif score >= 75:
        notes.append(
            f"The diversification score is {score:.0f}/100. Broad funds and a mix of asset "
            "types spread risk across many investments."
        )

    ratio = analysis.weighted_expense_ratio
    if ratio is not None and ratio > config.high_expense_ratio:
        notes.append(
            f"The weighted expense ratio is about {ratio:.2%} a year. At an assumed "
            f"{config.fee_growth_rate:.0%} annual growth, fees at this level would reduce the "
            f"portfolio's value by roughly ${analysis.fee_drag.get(30, 0):,.0f} over 30 years. "
            "Many broad index funds charge under 0.10%."
        )

    crypto = analysis.asset_allocation.get("crypto", 0.0)
    if crypto > 0.10:
        notes.append(
            f"Crypto assets are {crypto:.0%} of the portfolio. They have historically been far "
            "more volatile than stocks, with drops of more than 50% in some years."
        )

    if profile:
        for gap in analysis.profile_gaps:
            if gap.group == "equity" and abs(gap.difference) >= 0.15:
                direction = "more" if gap.difference > 0 else "less"
                notes.append(
                    f"Stocks are {gap.current:.0%} of the portfolio, {direction} than the "
                    f"{gap.target:.0%} in Finnie's reference allocation for a "
                    f"{profile.label.lower()} risk tolerance. More stock usually means higher "
                    "long-run growth potential and bigger short-term swings."
                )

    unknown = [r.ticker for r in analysis.holdings if not r.classification_known]
    if unknown:
        notes.append(
            f"{', '.join(unknown)} isn't in Finnie's reference list, so its classification is "
            "estimated from market data and may be imprecise."
        )
    if analysis.missing_prices:
        pronoun = "it is" if len(analysis.missing_prices) == 1 else "they are"
        notes.append(
            f"No price was available for {', '.join(analysis.missing_prices)}, so "
            f"{pronoun} left out of the totals."
        )
    if analysis.risk_metrics is None:
        notes.append(
            "Not enough price history was available to calculate volatility, drawdown, or beta."
        )
    return notes


def _sorted(mix: Mapping[str, float]) -> dict[str, float]:
    return {k: round(v, 4) for k, v in sorted(mix.items(), key=lambda kv: kv[1], reverse=True)}


# ---- orchestration --------------------------------------------------------------------


class MarketData(Protocol):
    def get_quotes(self, tickers: Iterable[str]) -> BatchQuotes: ...
    def get_daily_history(self, ticker: str, days: int = ...) -> PriceHistory: ...
    def get_company_overview(self, ticker: str) -> CompanyOverview: ...


def fetch_and_analyze(
    holdings: Sequence[Holding],
    market: MarketData,
    *,
    profile: RiskProfile | None = None,
    catalog: SecurityCatalog | None = None,
    config: AnalyticsConfig | None = None,
    include_history: bool = True,
) -> PortfolioAnalysis:
    """Fetch prices, classifications, and history, then analyze. Data failures degrade
    gracefully: a missing price drops that holding, missing history skips risk metrics."""
    catalog = catalog or get_catalog()
    config = config or get_settings().analytics
    merged = merge_holdings(holdings)
    if not merged:
        raise PortfolioError("The portfolio has no holdings.")

    cash = {h.ticker for h in merged if (info := catalog.get(h.ticker)) and info.type in CASH_TYPES}
    quotes = market.get_quotes([h.ticker for h in merged if h.ticker not in cash])
    prices = {t: q.price for t, q in quotes.quotes.items()} | dict.fromkeys(cash, 1.0)
    freshness = [q.freshness for q in quotes.quotes.values()]

    securities: dict[str, SecurityInfo] = {}
    for h in merged:
        known = catalog.get(h.ticker)
        if known:
            securities[h.ticker] = known
            continue
        try:
            overview = market.get_company_overview(h.ticker)
            securities[h.ticker] = catalog.classify(
                h.ticker, name=overview.name, asset_type=overview.asset_type, sector=overview.sector
            )
        except MarketDataError:
            securities[h.ticker] = catalog.classify(h.ticker)

    closes: dict[str, pd.Series] = {}
    benchmark = None
    if include_history:
        for ticker in prices:
            if ticker in cash:
                continue
            try:
                closes[ticker] = market.get_daily_history(ticker, config.history_days).to_frame()[
                    "close"
                ]
            except MarketDataError:
                continue
        try:
            benchmark = market.get_daily_history(config.benchmark, config.history_days).to_frame()[
                "close"
            ]
        except MarketDataError:
            benchmark = None
        if closes and cash:
            index = next(iter(closes.values())).index
            for ticker in cash:
                closes[ticker] = pd.Series(1.0, index=index)

    analysis = analyze_portfolio(
        merged,
        prices,
        securities=securities,
        closes=closes or None,
        benchmark_closes=benchmark,
        profile=profile,
        config=config,
    )
    analysis.freshness = freshness
    if any(f.is_mock for f in freshness):
        analysis.observations.append(
            "Some prices are demo data because live market data was unavailable, so values are "
            "illustrative only."
        )
    return analysis
