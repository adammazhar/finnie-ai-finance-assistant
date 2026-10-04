"""Goal projections: a deterministic projection plus a Monte Carlo simulation.

Model
-----
Monthly steps. Each month the contribution is added, then the balance grows by that
month's return: ``B[m] = (B[m-1] + c) * (1 + r[m])``. Monthly returns are drawn from a
Student-t distribution (fatter tails than a normal) scaled to the profile's volatility,
around the monthly equivalent of the expected annual return.

Because the recursion is linear in ``c``, each simulated path's balance is
``B = B0 * G + c * A``, where ``G`` is the path's total growth and ``A`` its contribution
growth factor. The simulation computes ``G`` and ``A`` once, which gives exact success
probabilities for any contribution and an exact solution for the contribution needed to
reach a target probability (no search loop).

Returns and volatility are illustrative long-run assumptions, not forecasts.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from src.core.config import MonteCarloConfig
from src.core.reference import RiskProfile

PERCENTILES = (10, 25, 50, 75, 90)
MAX_YEARS = 60


class GoalInputs(BaseModel):
    """Inputs for a savings-goal projection.

    Money amounts are dollars; ``target_amount`` is in today's dollars when
    ``target_in_todays_dollars`` is true, otherwise nominal. Return, volatility, and
    inflation are annual fractions (0.06 means 6%). ``seed`` makes runs reproducible.
    """

    current_balance: float = Field(ge=0)
    monthly_contribution: float = Field(ge=0)
    years: int = Field(ge=1, le=MAX_YEARS)
    target_amount: float = Field(gt=0)
    expected_return: float = Field(ge=-0.5, le=0.5, description="Annual, nominal")
    volatility: float = Field(ge=0, le=1, description="Annual standard deviation")
    inflation: float = Field(default=0.025, ge=-0.05, le=0.5)
    target_in_todays_dollars: bool = Field(
        default=True, description="If true, the target and results are inflation-adjusted"
    )
    simulations: int = Field(default=10_000, ge=100, le=200_000)
    t_degrees_of_freedom: float | None = Field(default=5, gt=2)
    seed: int | None = None

    @property
    def months(self) -> int:
        """The horizon in months."""
        return self.years * 12


class YearPercentiles(BaseModel):
    """Simulated balance percentiles (10th to 90th) at the end of one year."""

    year: int
    values: dict[int, float]  # percentile -> balance


class SimulationResult(BaseModel):
    """Outcome of :func:`simulate`: success odds, balance percentiles, and the deterministic path.

    Balances and the shortfall are in the dollars named by ``dollars`` (today's or
    nominal). ``total_contributions`` is the plain sum of monthly contributions, not
    inflation-adjusted.
    """

    inputs: GoalInputs
    success_probability: float = Field(ge=0, le=1)
    final_percentiles: dict[int, float]
    yearly: list[YearPercentiles]
    median_shortfall: float | None = Field(description="Median gap on paths that miss the target")
    deterministic_final: float
    total_contributions: float
    dollars: Literal["today's", "nominal"]


LOW_ODDS = 0.25  # below this, explain what would change the outcome


def chance_text(probability: float) -> str:
    """A probability in words: "under 1%", "64%", "over 99%" (never a misleading 0% or 100%)."""
    if probability < 0.01:
        return "under 1%"
    if probability > 0.99:
        return "over 99%"
    return f"{round(probability * 100)}%"


def years_text(years: int) -> str:
    """ "1 year", "10 years"."""
    return f"{years} year" if years == 1 else f"{years} years"


def low_odds_note() -> str:
    """Text explaining which levers improve a low chance of reaching the goal."""
    return (
        "The odds are low under these assumptions. Three things change the outcome most: "
        "more time to save, a higher monthly contribution, or a smaller target. Try adjusting "
        "them to see how the chance moves."
    )


def monthly_rate(annual_return: float) -> float:
    """Monthly rate that compounds to ``annual_return`` over 12 months."""
    return math.expm1(math.log1p(annual_return) / 12)  # stays exact for near-zero rates


def _annuity_factor(rate: float, months: int) -> float:
    """What contributions of 1 at the start of each month grow to: (1+r)((1+r)^n - 1)/r.

    ``expm1``/``log1p`` keep it accurate when ``rate`` is close to zero, where computing
    ``(1+r)^n - 1`` directly cancels out most of the digits.
    """
    if rate == 0:
        return float(months)
    return (1 + rate) * math.expm1(months * math.log1p(rate)) / rate


def deterministic_future_value(
    balance: float, monthly_contribution: float, annual_return: float, years: int
) -> float:
    """Future value with contributions at the start of each month (annuity due)."""
    rate = monthly_rate(annual_return)
    months = years * 12
    return balance * (1 + rate) ** months + monthly_contribution * _annuity_factor(rate, months)


def required_contribution_deterministic(
    balance: float, target: float, annual_return: float, years: int
) -> float:
    """Monthly contribution for the deterministic projection to reach ``target`` exactly."""
    rate = monthly_rate(annual_return)
    months = years * 12
    growth = (1 + rate) ** months
    remaining = target - balance * growth
    if remaining <= 0:
        return 0.0
    return remaining / _annuity_factor(rate, months)


def _shocks(inputs: GoalInputs, rng: np.random.Generator) -> np.ndarray:
    """One month of standardized (mean 0, variance 1) shocks, one per simulated path."""
    df = inputs.t_degrees_of_freedom
    if df is None:
        return rng.standard_normal(inputs.simulations)
    return rng.standard_t(df, size=inputs.simulations) * math.sqrt((df - 2) / df)


def _growth_factors(inputs: GoalInputs) -> tuple[np.ndarray, np.ndarray]:
    """Per-path growth factors G and contribution factors A at each year end.

    Shapes are (years + 1, simulations); row 0 is the start (G=1, A=0). Shocks are drawn
    one month at a time so memory stays proportional to the number of paths.
    """
    rng = np.random.default_rng(inputs.seed)
    mean = monthly_rate(inputs.expected_return)
    sigma = inputs.volatility / math.sqrt(12)

    sims = inputs.simulations
    g = np.ones(sims)
    a = np.zeros(sims)
    g_years = [g.copy()]
    a_years = [a.copy()]
    for month in range(inputs.months):
        growth = 1 + np.maximum(mean + sigma * _shocks(inputs, rng), -0.99)
        a = (a + 1) * growth
        g = g * growth
        if (month + 1) % 12 == 0:
            g_years.append(g.copy())
            a_years.append(a.copy())
    return np.array(g_years), np.array(a_years)


def _deflators(inputs: GoalInputs) -> np.ndarray:
    years = np.arange(inputs.years + 1)
    if not inputs.target_in_todays_dollars:
        return np.ones_like(years, dtype=float)
    return (1 + inputs.inflation) ** years


def simulate(inputs: GoalInputs) -> SimulationResult:
    """Run the Monte Carlo projection described in the module docstring.

    Success means the final balance is at least the target. When the target is in
    today's dollars, every balance is deflated by ``inputs.inflation`` first.
    """
    g, a = _growth_factors(inputs)
    deflators = _deflators(inputs)
    balances = (inputs.current_balance * g + inputs.monthly_contribution * a) / deflators[:, None]

    final = balances[-1]
    success = final >= inputs.target_amount
    shortfalls = inputs.target_amount - final[~success]

    deterministic = (
        deterministic_future_value(
            inputs.current_balance,
            inputs.monthly_contribution,
            inputs.expected_return,
            inputs.years,
        )
        / deflators[-1]
    )

    yearly = [
        YearPercentiles(
            year=year,
            values={
                p: round(float(v), 2)
                for p, v in zip(PERCENTILES, np.percentile(row, PERCENTILES), strict=True)
            },
        )
        for year, row in enumerate(balances)
    ]
    return SimulationResult(
        inputs=inputs,
        success_probability=round(float(success.mean()), 4),
        final_percentiles=yearly[-1].values,
        yearly=yearly,
        median_shortfall=round(float(np.median(shortfalls)), 2) if shortfalls.size else None,
        deterministic_final=round(deterministic, 2),
        total_contributions=round(inputs.monthly_contribution * inputs.months, 2),
        dollars="today's" if inputs.target_in_todays_dollars else "nominal",
    )


def required_monthly_contribution(inputs: GoalInputs, target_probability: float) -> float:
    """Smallest monthly contribution giving at least ``target_probability`` of success.

    Uses the same simulated paths as :func:`simulate` (same seed), so the answer is
    consistent with the reported probability.
    """
    if not 0 < target_probability < 1:
        raise ValueError("target_probability must be between 0 and 1")
    g, a = _growth_factors(inputs)
    target_nominal = inputs.target_amount * _deflators(inputs)[-1]
    needed = (target_nominal - inputs.current_balance * g[-1]) / a[-1]
    # A path needs contribution >= needed[i]; the p-quantile satisfies a fraction p of paths.
    value = float(np.quantile(needed, target_probability, method="higher"))
    return math.ceil(max(0.0, value) * 100) / 100  # round up to the cent


def inputs_for_profile(
    profile: RiskProfile,
    config: MonteCarloConfig,
    *,
    current_balance: float,
    monthly_contribution: float,
    years: int,
    target_amount: float,
    target_in_todays_dollars: bool = True,
    seed: int | None = None,
) -> GoalInputs:
    """Goal inputs using a risk profile's return/volatility assumptions and config defaults."""
    return GoalInputs(
        current_balance=current_balance,
        monthly_contribution=monthly_contribution,
        years=years,
        target_amount=target_amount,
        expected_return=profile.expected_return,
        volatility=profile.volatility,
        inflation=config.inflation,
        target_in_todays_dollars=target_in_todays_dollars,
        simulations=config.simulations,
        t_degrees_of_freedom=config.t_degrees_of_freedom,
        seed=seed,
    )
