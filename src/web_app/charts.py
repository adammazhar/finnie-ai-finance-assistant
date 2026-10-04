"""Plotly figure builders. Pure functions of data, so they are tested without Streamlit.

Every chart has a title, labelled axes with units, currency or percent formatting, hover
text, and colors from the Okabe-Ito palette, which stays distinguishable for colorblind
viewers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

from src.core.indicators import MoverSummary, rsi, sma
from src.core.monte_carlo import SimulationResult
from src.core.portfolio import Backtest
from src.core.reference import asset_class_label
from src.data.models import PriceHistory

# Okabe-Ito colorblind-safe palette, led by Finnie navy
PALETTE = ["#0B2545", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
UP, DOWN, NEUTRAL = "#0072B2", "#D55E00", "#999999"
FONT = dict(family="Inter, Segoe UI, sans-serif", size=13)


def _layout(fig: go.Figure, title: str, **kwargs: object) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left"),
        font=FONT,
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        **kwargs,
    )
    return fig


def allocation_donut(mix: Mapping[str, float], title: str = "Asset allocation") -> go.Figure:
    """Donut of the portfolio by asset class (equity, bond, cash, ...), in the given order."""
    labels = [asset_class_label(k) for k in mix]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=list(mix.values()),
            hole=0.55,
            sort=False,
            marker=dict(colors=PALETTE[: len(labels)]),
            texttemplate="%{label}<br>%{percent:.0%}",
            hovertemplate="%{label}: %{percent:.1%}<extra></extra>",
        )
    )
    return _layout(fig, title, showlegend=False)


def allocation_bar(
    mix: Mapping[str, float], title: str = "Sector allocation", axis: str = "Share of portfolio"
) -> go.Figure:
    """Horizontal bars of a mix (sectors by default), largest at the top."""
    items = sorted(mix.items(), key=lambda kv: kv[1])
    fig = go.Figure(
        go.Bar(
            x=[v for _, v in items],
            y=[k for k, _ in items],
            orientation="h",
            marker_color=PALETTE[0],
            texttemplate="%{x:.0%}",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_xaxes(title=axis, tickformat=".0%")
    return _layout(fig, title, height=max(260, 40 * len(items) + 90))


def correlation_heatmap(matrix: Mapping[str, Mapping[str, float]]) -> go.Figure:
    """How closely holdings' daily returns moved together (-1 to 1)."""
    tickers = list(matrix)
    values = [[matrix[a][b] for b in tickers] for a in tickers]
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=tickers,
            y=tickers,
            zmin=-1,
            zmax=1,
            colorscale=[[0, DOWN], [0.5, "#FFFFFF"], [1, UP]],
            texttemplate="%{z:.2f}",
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(title="Correlation"),
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Correlation of daily returns (past year)")


def fan_chart(result: SimulationResult) -> go.Figure:
    """P10-P90 band with the median, and the target as a dashed line."""
    years = [y.year for y in result.yearly]
    p10 = [y.values[10] for y in result.yearly]
    p50 = [y.values[50] for y in result.yearly]
    p90 = [y.values[90] for y in result.yearly]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=years, y=p90, line=dict(width=0), hoverinfo="skip", showlegend=False)
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=p10,
            fill="tonexty",
            fillcolor="rgba(0,114,178,0.18)",
            line=dict(width=0),
            name="P10 to P90 range",
            hoverinfo="skip",
        )
    )
    for values, name, dash in (
        (p90, "P90 (good markets)", "dot"),
        (p10, "P10 (poor markets)", "dot"),
    ):
        fig.add_trace(
            go.Scatter(
                x=years,
                y=values,
                name=name,
                line=dict(color=PALETTE[0], width=1, dash=dash),
                hovertemplate="Year %{x}: $%{y:,.0f}<extra>" + name + "</extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=p50,
            name="Median",
            line=dict(color=PALETTE[0], width=3),
            hovertemplate="Year %{x}: $%{y:,.0f}<extra>Median</extra>",
        )
    )
    fig.add_hline(
        y=result.inputs.target_amount,
        line=dict(color=PALETTE[1], dash="dash", width=2),
        annotation_text=f"Target ${result.inputs.target_amount:,.0f}",
        annotation_position="top left",
    )
    fig.update_xaxes(title="Years from now", dtick=max(1, len(years) // 10))
    fig.update_yaxes(title=f"Balance ({result.dollars} dollars)", tickprefix="$", tickformat=",.0f")
    return _layout(fig, "Projected balance: range of outcomes")


def probability_gauge(probability: float, title: str = "Chance of reaching the goal") -> go.Figure:
    """The Monte Carlo chance of success as a 0-100% gauge, with the 50% mark."""
    shown = round(probability * 100)
    prefix = ""
    if probability < 0.01:
        shown, prefix = 1, "<"
    elif probability > 0.99:
        shown, prefix = 99, ">"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=shown,
            number=dict(prefix=prefix, suffix="%", valueformat=".0f"),
            gauge=dict(
                axis=dict(range=[0, 100], ticksuffix="%"),
                bar=dict(color=PALETTE[0]),
                steps=[
                    dict(range=[0, 50], color="#F3E0D5"),
                    dict(range=[50, 80], color="#F7EFD2"),
                    dict(range=[80, 100], color="#D6EBE3"),
                ],
            ),
        )
    )
    return _layout(fig, title, height=260)


def _closes(history: PriceHistory) -> pd.Series:
    return pd.Series(
        [bar.close for bar in history.bars],
        index=pd.to_datetime([bar.date for bar in history.bars]),
        dtype=float,
    )


def price_chart(history: PriceHistory) -> go.Figure:
    """Daily closes with the 50- and 200-day moving averages (when there's enough history)."""
    closes = _closes(history)
    index = history.ticker.startswith("^")  # an index level is points, not dollars
    unit = "" if index else "$"
    fig = go.Figure(
        go.Scatter(
            x=closes.index,
            y=closes,
            name="Close",
            line=dict(color=PALETTE[0], width=2),
            hovertemplate="%{x|%b %d, %Y}: " + unit + "%{y:,.2f}<extra>Close</extra>",
        )
    )
    for window, color in ((50, PALETTE[1]), (200, PALETTE[2])):
        average = sma(closes, window)
        if average.notna().any():
            fig.add_trace(
                go.Scatter(
                    x=average.index,
                    y=average,
                    name=f"{window}-day average",
                    line=dict(color=color, width=1.5, dash="dash"),
                    hovertemplate="%{x|%b %d, %Y}: "
                    + unit
                    + "%{y:,.2f}<extra>"
                    + f"{window}-day</extra>",
                )
            )
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Index level" if index else "Price (USD)", tickprefix=unit)
    what = "level" if index else "price"
    return _layout(fig, f"{history.ticker} {what} with moving averages")


def rsi_chart(history: PriceHistory) -> go.Figure:
    """14-day RSI with the 70 (often called overbought) and 30 (oversold) bands shaded."""
    values = rsi(_closes(history))
    fig = go.Figure(
        go.Scatter(
            x=values.index,
            y=values,
            name="RSI (14-day)",
            line=dict(color=PALETTE[3], width=2),
            hovertemplate="%{x|%b %d, %Y}: %{y:.0f}<extra>RSI</extra>",
        )
    )
    fig.add_hrect(y0=70, y1=100, fillcolor=DOWN, opacity=0.08, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor=UP, opacity=0.08, line_width=0)
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="RSI (0-100)", range=[0, 100])
    return _layout(
        fig, f"{history.ticker} relative strength (above 70 often called overbought)", height=260
    )


def movers_bar(movers: Sequence[MoverSummary], title: str) -> go.Figure:
    """Today's percent change per fund, strongest first, colored up or down."""
    changes = [m.change_percent or 0.0 for m in movers]
    fig = go.Figure(
        go.Bar(
            x=changes,
            y=[f"{m.name} ({m.ticker})" for m in movers],
            orientation="h",
            marker_color=[UP if c > 0 else DOWN if c < 0 else NEUTRAL for c in changes],
            texttemplate="%{x:+.2f}%",
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
        )
    )
    # room beyond the longest bar on each side, so labels like "-0.01%" are never cut off
    low, high = min(0.0, *changes), max(0.0, *changes)
    pad = ((high - low) or 1.0) * 0.25
    fig.update_xaxes(
        title="Change today (%)", ticksuffix="%", zeroline=True, range=[low - pad, high + pad]
    )
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, height=max(260, 32 * len(movers) + 90))


def performance_chart(backtest: Backtest) -> go.Figure:
    """Today's holdings over the past year vs the benchmark, both starting at 100."""
    fig = go.Figure(
        go.Scatter(
            x=backtest.dates,
            y=backtest.portfolio,
            name="Today's holdings",
            line=dict(color=PALETTE[0], width=2.5),
            hovertemplate="%{x|%b %d, %Y}: %{y:.1f}<extra>Today's holdings</extra>",
        )
    )
    if backtest.benchmark:
        fig.add_trace(
            go.Scatter(
                x=backtest.dates,
                y=backtest.benchmark,
                name=f"{backtest.benchmark_ticker} (S&P 500)",
                line=dict(color=PALETTE[2], width=2, dash="dash"),
                hovertemplate="%{x|%b %d, %Y}: %{y:.1f}<extra>"
                + backtest.benchmark_ticker
                + "</extra>",
            )
        )
    fig.add_hline(y=100, line=dict(color=NEUTRAL, width=1, dash="dot"))
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Value (start of period = 100)")
    return _layout(
        fig, f"Back-test: today's holdings over the past year vs {backtest.benchmark_ticker}"
    )
