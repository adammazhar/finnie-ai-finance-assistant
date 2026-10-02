"""Goals tab: a Monte Carlo projection for one savings goal, run directly (no chat).

Instead of the chat's question, a checkbox decides whether the saved portfolio counts,
with an editable amount that starts at the portfolio's current value.
"""

from __future__ import annotations

from typing import Any, get_args

import streamlit as st

from src.core.models import RiskTolerance
from src.core.monte_carlo import (
    LOW_ODDS,
    chance_text,
    inputs_for_profile,
    low_odds_note,
    required_monthly_contribution,
    simulate,
)
from src.core.portfolio import portfolio_total
from src.web_app import charts, services, state
from src.web_app.formatting import md, money, percent

GOAL_TYPES = ["Retirement", "House down payment", "College", "Emergency fund", "Other"]
RISKS: list[str] = list(get_args(RiskTolerance))
RESULT = "goal_result"
ODDS_NOTE = (
    "How to read the chance: a median (typical) outcome above the target means roughly even "
    "odds. Odds near 99% need even the poor-markets (P10) outcome, which only 1 in 10 "
    "simulated paths falls below, to clear the target."
)


def _portfolio_amount() -> float:
    """How much of the saved portfolio counts: 0 unless the box is ticked."""
    holdings = state.portfolio()
    value = portfolio_total(holdings, services.context().market) if holdings else None
    include = st.checkbox(
        "Include saved portfolio",
        key="goal_include_portfolio",
        disabled=not value,
        help="Count some or all of the portfolio from the Portfolio tab toward this goal.",
    )
    if not value:
        st.caption("No saved portfolio with current prices. Add one on the Portfolio tab.")
        return 0.0
    if not include:
        st.caption(
            md(f"Your saved portfolio is worth {money(value, cents=True)}; it isn't counted.")
        )
        return 0.0
    amount = st.number_input(
        "Amount of the portfolio to count",
        min_value=0.0,
        value=round(value, 2),
        step=1000.0,
        format="%.2f",
        key="goal_portfolio_amount",
        help=md(
            f"Your portfolio is worth {money(value, cents=True)}. Lower this to count part of it."
        ),
    )
    return float(amount)


def _run(inputs: dict[str, Any]) -> None:
    context = services.context()
    profile = context.risk_profiles[inputs["risk"]]
    mc = context.settings.analytics.monte_carlo
    goal = inputs_for_profile(
        profile,
        mc,
        current_balance=inputs["current_balance"],
        monthly_contribution=inputs["monthly"],
        years=inputs["years"],
        target_amount=inputs["target"],
        target_in_todays_dollars=inputs["todays_dollars"],
        seed=42,
    )
    result = simulate(goal)
    needed = required_monthly_contribution(goal, mc.target_success_probability)
    st.session_state[RESULT] = {
        "goal": inputs["goal"],
        "result": result,
        "needed": needed,
        "target_probability": mc.target_success_probability,
        "risk": inputs["risk"],
        "expected_return": profile.expected_return,
        "volatility": profile.volatility,
    }


def _inputs() -> dict[str, Any]:
    left, right = st.columns(2)
    with left:
        goal = st.selectbox("Goal", GOAL_TYPES, key="goal_type")
        target = st.number_input(
            "Target amount ($)", min_value=1000.0, value=500_000.0, step=10_000.0, key="goal_target"
        )
        years = st.slider(
            "Years until the goal", min_value=1, max_value=50, value=25, key="goal_years"
        )
        todays_dollars = st.toggle(
            "Target is in today's dollars (adjust for inflation)", value=True, key="goal_inflation"
        )
    with right:
        other = st.number_input(
            "Other savings already set aside for this goal ($)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            key="goal_other",
        )
        from_portfolio = _portfolio_amount()
        monthly = st.number_input(
            "Monthly contribution ($)", min_value=0.0, value=500.0, step=50.0, key="goal_monthly"
        )
        risk = st.selectbox(
            "Risk level for the projection",
            RISKS,
            index=RISKS.index(state.profile().risk_tolerance),
            key="goal_risk",
        )
    current = other + from_portfolio
    st.caption(md(f"Starting balance for the projection: {money(current, cents=True)}"))
    return {
        "goal": goal,
        "target": target,
        "years": years,
        "todays_dollars": todays_dollars,
        "current_balance": current,
        "monthly": monthly,
        "risk": risk,
    }


def _results(saved: dict[str, Any]) -> None:
    result = saved["result"]
    p = result.final_percentiles
    st.markdown(
        md(
            f"### {saved['goal']}: {money(result.inputs.target_amount)} in "
            f"{result.inputs.years} years"
        )
    )
    st.markdown(f"#### Chance of reaching this goal: {chance_text(result.success_probability)}")
    st.caption(ODDS_NOTE)
    if result.success_probability < LOW_ODDS:
        st.info(low_odds_note(), icon=":material/lightbulb:")
    left, right = st.columns([1, 2])
    with left:
        st.plotly_chart(
            charts.probability_gauge(result.success_probability), width="stretch", key="goal_gauge"
        )
    with right:
        cols = st.columns(3)
        cols[0].metric("Poor markets (P10)", money(p[10]))
        cols[1].metric("Median", money(p[50]))
        cols[2].metric("Good markets (P90)", money(p[90]))
        st.metric(
            f"Monthly contribution for {percent(saved['target_probability'], 0)} odds of success",
            money(saved["needed"]),
            help="What the projection says would be needed under these assumptions. "
            "It's an illustration, not a recommendation.",
        )
        st.caption(
            md(f"With steady returns and no market swings: {money(result.deterministic_final)}.")
        )
    st.plotly_chart(charts.fan_chart(result), width="stretch", key="goal_fan")
    with st.expander("Assumptions"):
        inputs = result.inputs
        expected, volatility = percent(saved["expected_return"]), percent(saved["volatility"], 0)
        lines = [
            f"- Risk level: **{saved['risk']}**, expected return {expected} a year, "
            f"volatility {volatility}",
            f"- Inflation: {percent(inputs.inflation)} a year; results in {result.dollars} dollars",
            f"- Starting balance {money(inputs.current_balance, cents=True)}, "
            f"{money(inputs.monthly_contribution)} a month for {inputs.years} years",
            f"- {inputs.simulations:,} simulated market paths with fat-tailed returns",
        ]
        st.markdown(md("\n".join(lines)))
    st.caption(
        "Projections are hypothetical, based on simplified assumptions, and not forecasts. "
        "Finnie models saving toward a target, not withdrawals in retirement."
    )


def render() -> None:
    st.subheader("Plan a savings goal")
    inputs = _inputs()
    if st.button("Run projection", type="primary", key="goal_run"):
        _run(inputs)
    saved = st.session_state.get(RESULT)
    if saved:
        _results(saved)
    else:
        st.info("Set your goal and press **Run projection** to see the range of outcomes.")
