import time

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from src.core.config import MonteCarloConfig
from src.core.monte_carlo import (
    PERCENTILES,
    GoalInputs,
    deterministic_future_value,
    inputs_for_profile,
    monthly_rate,
    required_contribution_deterministic,
    required_monthly_contribution,
    simulate,
)
from src.core.reference import get_risk_profiles


def goal(**overrides):
    base = dict(
        current_balance=10_000,
        monthly_contribution=500,
        years=20,
        target_amount=250_000,
        expected_return=0.06,
        volatility=0.10,
        inflation=0.025,
        target_in_todays_dollars=False,
        simulations=2_000,
        seed=42,
    )
    return GoalInputs(**(base | overrides))


def loop_future_value(balance, monthly, annual, years):
    """Independent month-by-month check of the annuity-due formula."""
    rate = (1 + annual) ** (1 / 12) - 1
    for _ in range(years * 12):
        balance = (balance + monthly) * (1 + rate)
    return balance


def test_monthly_rate_compounds_to_annual():
    assert (1 + monthly_rate(0.12)) ** 12 == pytest.approx(1.12)


@pytest.mark.parametrize(("annual", "years"), [(0.12, 1), (0.06, 30), (-0.02, 10), (0.0, 5)])
def test_deterministic_future_value(annual, years):
    expected = loop_future_value(1000, 100, annual, years)
    assert deterministic_future_value(1000, 100, annual, years) == pytest.approx(expected)
    assert deterministic_future_value(1000, 100, 0.12, 1) == pytest.approx(2396.6498, abs=1e-3)


def test_required_contribution_deterministic_round_trips():
    needed = required_contribution_deterministic(10_000, 500_000, 0.06, 30)
    assert deterministic_future_value(10_000, needed, 0.06, 30) == pytest.approx(500_000)
    assert required_contribution_deterministic(10_000, 50_000, 0.0, 10) == pytest.approx(
        40_000 / 120
    )
    assert required_contribution_deterministic(1_000_000, 5, 0.06, 1) == 0


def test_zero_volatility_matches_deterministic_projection():
    result = simulate(goal(volatility=0))
    final = deterministic_future_value(10_000, 500, 0.06, 20)
    assert result.deterministic_final == pytest.approx(final, abs=0.01)
    assert all(v == pytest.approx(final, abs=0.01) for v in result.final_percentiles.values())
    assert result.success_probability == (1.0 if final >= 250_000 else 0.0)


def test_result_shape_and_ordering():
    result = simulate(goal())
    assert [y.year for y in result.yearly] == list(range(21))
    assert result.yearly[0].values == dict.fromkeys(PERCENTILES, 10_000.0)
    for year in result.yearly:
        values = [year.values[p] for p in PERCENTILES]
        assert values == sorted(values)
    assert result.final_percentiles == result.yearly[-1].values
    assert result.total_contributions == 500 * 240
    assert 0 < result.success_probability < 1 and result.median_shortfall > 0
    assert result.dollars == "nominal"


def test_seed_reproducibility_and_distribution_choice():
    assert simulate(goal()).final_percentiles == simulate(goal()).final_percentiles
    assert simulate(goal(seed=1)).final_percentiles != simulate(goal(seed=2)).final_percentiles
    normal = simulate(goal(t_degrees_of_freedom=None))
    fat = simulate(goal(t_degrees_of_freedom=3))
    spread = lambda r: r.final_percentiles[90] - r.final_percentiles[10]  # noqa: E731
    assert spread(normal) > 0 and spread(fat) > 0


def test_inflation_adjustment_lowers_real_values():
    nominal = simulate(goal())
    real = simulate(goal(target_in_todays_dollars=True))
    assert real.dollars == "today's"
    assert real.final_percentiles[50] == pytest.approx(
        nominal.final_percentiles[50] / 1.025**20, rel=1e-6
    )
    assert real.success_probability <= nominal.success_probability


def test_certain_success_has_no_shortfall():
    result = simulate(goal(current_balance=5_000_000, volatility=0.05))
    assert result.success_probability == 1.0 and result.median_shortfall is None


def test_required_contribution_hits_target_probability():
    base = goal()
    needed = required_monthly_contribution(base, 0.8)
    check = simulate(base.model_copy(update={"monthly_contribution": needed}))
    assert check.success_probability >= 0.8
    slightly_less = simulate(base.model_copy(update={"monthly_contribution": needed - 5}))
    assert slightly_less.success_probability < 0.8
    assert required_monthly_contribution(goal(current_balance=5_000_000), 0.9) == 0


def test_required_contribution_validates_probability():
    for bad in (0, 1, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            required_monthly_contribution(goal(), bad)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    low=st.floats(min_value=0, max_value=2_000),
    extra=st.floats(min_value=1, max_value=2_000),
    vol=st.floats(min_value=0, max_value=0.3),
)
def test_more_contribution_never_lowers_success(low, extra, vol):
    base = goal(volatility=vol, simulations=300, years=10, target_amount=100_000)
    lower = simulate(base.model_copy(update={"monthly_contribution": low}))
    higher = simulate(base.model_copy(update={"monthly_contribution": low + extra}))
    assert higher.success_probability >= lower.success_probability
    assert higher.final_percentiles[50] >= lower.final_percentiles[50]


@settings(max_examples=25, deadline=None)
@given(
    balance=st.floats(min_value=0, max_value=1e6),
    monthly=st.floats(min_value=0, max_value=1e4),
    annual=st.floats(min_value=-0.1, max_value=0.15),
    years=st.integers(min_value=1, max_value=40),
)
def test_deterministic_formula_matches_loop(balance, monthly, annual, years):
    assert deterministic_future_value(balance, monthly, annual, years) == pytest.approx(
        loop_future_value(balance, monthly, annual, years), rel=1e-9, abs=1e-6
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"years": 0},
        {"years": 61},
        {"target_amount": 0},
        {"current_balance": -1},
        {"simulations": 10},
        {"t_degrees_of_freedom": 2},
        {"volatility": -0.1},
    ],
)
def test_input_validation(bad):
    with pytest.raises(ValidationError):
        goal(**bad)


def test_inputs_for_profile():
    profile = get_risk_profiles()["aggressive"]
    inputs = inputs_for_profile(
        profile,
        MonteCarloConfig(simulations=500),
        current_balance=1,
        monthly_contribution=2,
        years=3,
        target_amount=4,
        seed=7,
    )
    assert (inputs.expected_return, inputs.volatility) == (0.075, 0.15)
    assert inputs.simulations == 500 and inputs.seed == 7 and inputs.target_in_todays_dollars


def test_performance_default_size():
    start = time.perf_counter()
    simulate(goal(simulations=10_000, years=40))
    assert time.perf_counter() - start < 3  # design target is ~0.2s; generous for CI machines


def test_chance_text_never_says_zero_or_certain():
    from src.core.monte_carlo import chance_text, low_odds_note

    assert chance_text(0.0) == chance_text(0.004) == "under 1%"
    assert chance_text(0.6432) == "64%"
    assert chance_text(1.0) == "over 99%"
    assert "more time to save" in low_odds_note()


# ---- sanity properties of the chance of success ----------------------------------------

SANITY = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def sanity_goal(current, monthly, years, target, seed=7):
    return GoalInputs(
        current_balance=current,
        monthly_contribution=monthly,
        years=years,
        target_amount=target,
        expected_return=0.06,
        volatility=0.15,
        simulations=1_000,
        seed=seed,
    )


def test_poor_markets_far_above_the_target_reads_over_99_percent():
    """A $500K goal whose poor-markets (P10) outcome is millions must read "over 99%"."""
    from src.core.monte_carlo import chance_text

    result = simulate(sanity_goal(current=2_000_000, monthly=1_000, years=25, target=500_000))
    assert result.final_percentiles[10] > 3 * result.inputs.target_amount
    assert result.success_probability >= 0.99
    assert chance_text(result.success_probability) == "over 99%"


@SANITY
@given(
    current=st.floats(0, 2_000_000),
    monthly=st.floats(0, 10_000),
    years=st.integers(1, 40),
    target=st.floats(10_000, 3_000_000),
)
def test_p10_above_target_means_at_least_90_percent(current, monthly, years, target):
    result = simulate(sanity_goal(current, monthly, years, target))
    if result.final_percentiles[10] > target:
        assert result.success_probability >= 0.9
    if result.final_percentiles[50] > target:  # a median above the target: at least even odds
        assert result.success_probability >= 0.5


@SANITY
@given(
    current=st.floats(0, 500_000),
    years=st.integers(1, 40),
    target=st.floats(10_000, 3_000_000),
    contributions=st.lists(st.floats(0, 10_000), min_size=2, max_size=5),
)
def test_more_contributions_never_lower_the_odds(current, years, target, contributions):
    odds = [
        simulate(sanity_goal(current, monthly, years, target)).success_probability
        for monthly in sorted(contributions)
    ]
    assert odds == sorted(odds)
