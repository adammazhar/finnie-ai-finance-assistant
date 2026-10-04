"""Portfolio tab: enter holdings (table, CSV, or a sample), then see the analysis."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.core.models import Holding
from src.core.portfolio import (
    FUND_TYPES,
    PortfolioAnalysis,
    PortfolioError,
    expense_ratio_label,
    fetch_and_analyze,
    fund_expense_ratio,
    holding_error,
    holdings_from_csv,
)
from src.core.reference import asset_class_label
from src.web_app import charts, services, state
from src.web_app.formatting import freshness_caption, md, money, percent

SAMPLES_DIR = Path(__file__).resolve().parents[3] / "data" / "sample_portfolios"
SAMPLES = {
    "Three-fund beginner": "three_fund_beginner.csv",
    "Concentrated in tech": "concentrated_tech.csv",
    "Near retirement, income focus": "near_retirement_income.csv",
}
EXPLAIN_PROMPT = "Explain my portfolio: how diversified is it, and what are its main risks?"
EDITOR_VERSION = "pf_editor_version"
UPLOAD_USED = "pf_upload_used"  # file_id of the upload last loaded with "Use uploaded file"


@st.cache_data(ttl=600, show_spinner="Analyzing your portfolio…")
def analyze(holdings_json: str, risk_tolerance: str) -> PortfolioAnalysis:
    """Fetch prices and analyze the holdings against the user's risk profile.

    The holdings arrive as JSON so the result can be cached (10 minutes) by value.
    """
    context = services.context()
    holdings = [Holding.model_validate(h) for h in json.loads(holdings_json)]
    return fetch_and_analyze(
        holdings,
        context.market,
        profile=context.risk_profiles[risk_tolerance],  # type: ignore[index]
        catalog=context.catalog,
        config=context.settings.analytics,
    )


def _warn_all(messages: Iterable[str]) -> None:
    for message in messages:
        st.warning(md(message))


def _replace(holdings: list[Holding]) -> None:
    state.set_portfolio(holdings)
    st.session_state[EDITOR_VERSION] = st.session_state.get(EDITOR_VERSION, 0) + 1


def _load_from_csv(text: str, label: str) -> None:
    holdings, errors = holdings_from_csv(text)
    _warn_all(f"{label}: {error}" for error in errors)
    if holdings:
        _replace(holdings)
        st.success(md(f"Loaded {len(holdings)} holdings from {label}."))


def _inputs() -> None:
    left, right = st.columns(2)
    with left:
        sample = st.selectbox(
            "Start from a sample portfolio", list(SAMPLES), index=None, key="pf_sample"
        )
        if st.button("Load sample", disabled=sample is None, key="pf_load_sample") and sample:
            _load_from_csv((SAMPLES_DIR / SAMPLES[sample]).read_text(encoding="utf-8"), sample)
    with right:
        upload = st.file_uploader(
            "Or upload a CSV with columns ticker, shares, cost_basis (optional)",
            type="csv",
            key="pf_upload",
        )
        if upload is not None and st.button("Use uploaded file", key="pf_use_upload"):
            st.session_state[UPLOAD_USED] = upload.file_id
            _load_from_csv(upload.getvalue().decode("utf-8", errors="replace"), upload.name)

    rows = pd.DataFrame(
        [h.model_dump() for h in state.portfolio()] or [],
        columns=["ticker", "shares", "cost_basis"],
    )
    edited = st.data_editor(
        rows,
        num_rows="dynamic",
        key=f"pf_editor_{st.session_state.get(EDITOR_VERSION, 0)}",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", required=True),
            # no min_value: the editor would silently flip "-5" to 5; Save explains instead
            "shares": st.column_config.NumberColumn("Shares", format="%.4g"),
            "cost_basis": st.column_config.NumberColumn(
                "Total cost (optional)",
                format="dollar",
                help="What you paid for all the shares together (cost_basis in a CSV file).",
            ),
        },
        width="stretch",
    )
    if st.button("Save holdings", type="primary", key="pf_save"):
        _save(edited, upload)


def _save(edited: pd.DataFrame, upload: Any) -> None:
    """Save the table, unless a chosen file hasn't been loaded yet or there's nothing to save."""
    if upload is not None and st.session_state.get(UPLOAD_USED) != upload.file_id:
        st.warning(
            "You chose a file but haven't loaded it yet. Click **Use uploaded file** to "
            "load it, or remove the file to save the table as it is."
        )
        return
    holdings, problems = _rows_to_holdings(edited)
    _warn_all(problems)
    if not holdings and not problems and not state.portfolio():
        st.info("The table is empty. Add a row (ticker and shares) or load a sample first.")
        return
    _replace(holdings)  # an empty table clears a saved portfolio


def _rows_to_holdings(frame: pd.DataFrame) -> tuple[list[Holding], list[str]]:
    holdings, problems = [], []
    for number, row in enumerate(frame.to_dict("records"), 1):
        ticker = str(row.get("ticker") or "").strip()
        if not ticker:
            continue
        cost = row.get("cost_basis")
        try:
            holdings.append(
                Holding(
                    ticker=ticker,
                    shares=float(row.get("shares") or 0),
                    cost_basis=None if cost is None or pd.isna(cost) else float(cost),
                )
            )
        except ValueError as exc:
            problems.append(f"Row {number} ({ticker}) was skipped: {holding_error(exc)}")
    return holdings, problems


def _metrics(analysis: PortfolioAnalysis) -> None:
    cols = st.columns(4)
    cols[0].metric("Total value", money(analysis.total_value, cents=True))
    cols[1].metric(
        "Diversification",
        f"{analysis.diversification_score:.0f} / 100",
        help="How spread out the money is across holdings, asset types, and sectors. "
        "Higher means one holding or sector going wrong hurts less.",
    )
    cols[2].metric(
        "Risk level",
        f"{analysis.risk_level} ({analysis.risk_score:.1f}/10)",
        help="How much the portfolio's value could swing, from 1 (cash-like) to 10 "
        "(concentrated in volatile stocks), based on what the holdings are.",
    )
    cols[3].metric(
        "Portfolio expense ratio",
        percent(fund_expense_ratio(analysis.holdings), 2),
        help="Value-weighted yearly fee across the funds you hold.",
    )
    st.caption(md(expense_ratio_label(analysis)))
    if analysis.unrealized_gain is not None:
        st.caption(
            md(
                f"Unrealized gain: {money(analysis.unrealized_gain)} "
                f"({percent(analysis.unrealized_gain_pct, signed=True)}) on the holdings "
                "with a cost."
            )
        )


def _risk(analysis: PortfolioAnalysis) -> None:
    metrics = analysis.risk_metrics
    if metrics is None:
        st.info("Not enough price history to measure past-year risk.")
        return
    table = pd.DataFrame(
        [
            (
                "Return over the period",
                percent(metrics.annual_return, signed=True),
                "How much today's holdings gained or lost over the past year.",
            ),
            (
                "Volatility (annualized)",
                percent(metrics.annual_volatility),
                "How much the value bounced around. Higher means bumpier.",
            ),
            (
                "Largest drop from a high",
                percent(metrics.max_drawdown),
                "The worst fall from a peak to a low during the period.",
            ),
            (
                "Beta vs S&P 500",
                f"{metrics.beta:.2f}" if metrics.beta is not None else "n/a",
                "1 moves like the market; 0.5 moves about half as much; above 1 moves more.",
            ),
            (
                "Sharpe ratio",
                f"{metrics.sharpe_ratio:.2f}" if metrics.sharpe_ratio is not None else "n/a",
                "Return earned per unit of bumpiness, above a safe Treasury bill. "
                "Higher is better; below 0 means the bill did better.",
            ),
        ],
        columns=["Measure", "Value", "What it means"],
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        f"{metrics.start:%b %d, %Y} to {metrics.end:%b %d, %Y}. Sharpe uses a risk-free rate of "
        f"{md(metrics.risk_free.label())}. Past results don't predict future returns."
    )


def _fee(ratio: float | None, is_fund: bool) -> str:
    if not is_fund:
        return "—"
    return f"{ratio:.2%}" if ratio is not None else "not known"


def _holdings_table(analysis: PortfolioAnalysis) -> None:
    table = pd.DataFrame(
        [
            {
                "Ticker": h.ticker,
                "Name": h.name,
                "Shares": h.shares,
                "Price": h.price,
                "Value": h.value,
                "Weight": f"{h.weight:.1%}",
                "Sector": h.sector,
                "Expense ratio": _fee(h.expense_ratio, h.type in FUND_TYPES),
            }
            for h in analysis.holdings
        ]
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Price": st.column_config.NumberColumn(format="dollar"),
            "Value": st.column_config.NumberColumn(format="dollar"),
            "Expense ratio": st.column_config.TextColumn(
                help="Yearly fund fee. Individual stocks don't charge one."
            ),
        },
    )


def _performance(analysis: PortfolioAnalysis) -> None:
    test = analysis.backtest
    if test is None:
        return
    st.plotly_chart(charts.performance_chart(test), width="stretch", key="pf_performance")
    left_out = (
        ""
        if test.coverage >= 0.999
        else f" Holdings without a year of prices ({percent(1 - test.coverage, 0)} of the "
        "value) are left out."
    )
    st.caption(
        "A back-test: today's holdings, with today's share counts, valued over the past year. "
        "It isn't your actual past return, because it ignores trades and deposits during the "
        "year, and past results don't predict future returns." + left_out
    )


def _analysis(analysis: PortfolioAnalysis) -> None:
    _metrics(analysis)
    if analysis.missing_prices:
        st.warning(f"No price found for: {', '.join(analysis.missing_prices)}. They're left out.")
    left, right = st.columns(2)
    left.plotly_chart(
        charts.allocation_donut(analysis.asset_allocation), width="stretch", key="pf_donut"
    )
    right.plotly_chart(
        charts.allocation_bar(analysis.sector_allocation), width="stretch", key="pf_sectors"
    )
    _performance(analysis)
    if analysis.observations:
        st.markdown("**What stands out**")
        st.markdown(md("\n".join(f"- {line}" for line in analysis.observations)))
    # The tab always passes the user's risk profile, so there are always gaps to show.
    st.markdown(f"**Compared with a typical {analysis.profile} mix**")
    st.caption(
        "This typical mix is based on your risk tolerance only. Your time horizon matters "
        "too: money you'll need within a few years has less time to recover from a market drop."
    )
    gaps = pd.DataFrame(
        [
            {
                "Group": asset_class_label(g.group),
                "Yours": g.current,
                "Typical": g.target,
                "Difference": g.difference,
            }
            for g in analysis.profile_gaps
        ]
    )
    st.dataframe(
        gaps,
        hide_index=True,
        width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="percent")
            for c in ("Yours", "Typical", "Difference")
        },
    )
    st.markdown("**Past-year risk**")
    _risk(analysis)
    if analysis.correlation:
        st.plotly_chart(
            charts.correlation_heatmap(analysis.correlation), width="stretch", key="pf_corr"
        )
    with st.expander("Holdings detail"):
        _holdings_table(analysis)
    caption = freshness_caption(analysis.freshness)
    if caption:
        st.caption(caption)


def render() -> None:
    """Draw the Portfolio tab: holdings input, then the analysis of the saved holdings."""
    st.subheader("Your portfolio")
    st.caption("Holdings saved here are used in chat and on the Goals tab.")
    _inputs()
    holdings = state.portfolio()
    if not holdings:
        st.info("Add holdings above, load a sample, or upload a CSV to see an analysis.")
        return
    payload = json.dumps([h.model_dump() for h in holdings], sort_keys=True)
    try:
        analysis = analyze(payload, state.profile().risk_tolerance)
    except PortfolioError as exc:
        st.error(str(exc))
        return
    st.button(
        "Explain my portfolio in chat",
        icon=":material/chat:",
        key="pf_explain",
        on_click=state.queue_prompt,
        args=(EXPLAIN_PROMPT,),
    )
    _analysis(analysis)
