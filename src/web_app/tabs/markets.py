"""Markets tab: index and sector snapshot, plus a ticker lookup with trend and news."""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from src.core.indicators import MoverSummary, technical_snapshot
from src.core.market_hours import market_status, price_time_label
from src.core.models import normalize_ticker
from src.data.errors import MarketDataError
from src.data.symbols import SymbolDirectory
from src.web_app import charts, services
from src.web_app.formatting import (
    big_money,
    first_sentence,
    freshness_caption,
    md,
    money,
    percent,
    provider_name,
)

logger = logging.getLogger(__name__)

HISTORY_DAYS = 400  # enough for a 200-day average over the past year
SUGGESTED_COMPANIES = 4000  # the largest SEC companies offered while typing
TRENDS = {
    "uptrend": "Uptrend",
    "downtrend": "Downtrend",
    "mixed": "Mixed",
    "unknown": "Not enough history",
}


def _overview() -> None:
    try:
        overview = services.market_overview()
    except MarketDataError as exc:
        st.error(f"Market data is unavailable right now ({exc}).")
        return
    if overview.indices:
        st.caption(price_time_label(overview.indices[0].freshness.as_of))
        columns = st.columns(len(overview.indices))
        for column, etf in zip(columns, overview.indices, strict=True):
            _index_card(column, etf, overview.levels.get(etf.ticker))
    if overview.mood.summary:
        st.info(md(" ".join(overview.mood.summary)))
    if overview.sectors:
        st.plotly_chart(
            charts.movers_bar(overview.sectors, "Sector funds today"),
            width="stretch",
            key="mk_sectors",
        )
    caption = freshness_caption([m.freshness for m in [*overview.indices, *overview.sectors]])
    if caption:
        st.caption(caption + ". Sector moves are shown through the sector ETFs.")


def _change(mover: MoverSummary) -> str | None:
    return f"{mover.change_percent:+.2f}%" if mover.change_percent is not None else None


def _index_card(column: Any, etf: MoverSummary, level: MoverSummary | None) -> None:
    """The index level (e.g. S&P 500 at 6,745), with the ETF that tracks it underneath."""
    with column:
        if level is None:  # the index quote is unavailable: show the ETF alone
            st.metric(f"{etf.name} via {etf.ticker}", money(etf.price, cents=True), _change(etf))
            return
        st.metric(f"{etf.name} ({level.ticker})", f"{level.price:,.2f}", _change(level))
        st.caption(
            md(f"Tracked by {etf.ticker}: {money(etf.price, cents=True)} {_change(etf) or ''}")
        )


def _news(ticker: str) -> None:
    try:
        feed = services.context().market.get_news(ticker=ticker, limit=5)
    except MarketDataError:
        st.caption("News is unavailable right now.")
        return
    if not feed.articles:
        st.caption(f"No recent news found for {ticker}.")
        return
    st.markdown(f"**Recent news about {ticker}**")
    for article in feed.articles:
        title = f"[{md(article.title)}]({article.url})" if article.url else md(article.title)
        details = " · ".join(
            x
            for x in (
                article.source,
                f"{article.published_at:%b %d, %Y}" if article.published_at else None,
            )
            if x
        )
        st.markdown(f"- {title}" + (f"  \n  :gray[{md(details)}]" if details else ""))
    st.caption(freshness_caption([feed.freshness]) or "")


def _is_index(ticker: str) -> bool:
    return ticker.startswith("^")


def _amount(ticker: str, value: float | None, decimals: int = 2) -> str:
    """Dollars for a stock or fund; plain points for an index level."""
    if value is None:
        return "n/a"
    if _is_index(ticker):
        return f"{value:,.{decimals}f}"
    return money(value, cents=decimals > 0)


def _suggestions(directory: SymbolDirectory) -> list[str]:
    """What the search box offers as you type: every index and Finnie fund, plus the largest
    SEC companies. Anything else is found by pressing Enter (see ``_resolve``)."""
    shown = [e for e in directory.entries if e.source != "sec"]
    shown += [e for e in directory.entries if e.source == "sec"][:SUGGESTED_COMPANIES]
    return [e.label for e in shown]


def _resolve(text: str, directory: SymbolDirectory) -> str | None:
    """The ticker for a typed name or ticker, or ``None`` after explaining what to do.

    Searches every SEC company (not just the ones offered while typing), then Yahoo
    Finance; with several matches, the user picks one.
    """
    text = text.strip()
    exact = directory.get(text)
    if exact is not None:
        return exact.ticker
    matches = directory.search(text, limit=6) or services.online_symbol_search(text)
    if matches:
        picked = st.radio(
            md(f'Did you mean one of these for "{text}"?'),
            [m.label for m in matches],
            index=None,
            key="mk_pick",
        )
        return picked.split(" · ", 1)[0] if picked else None
    try:
        return normalize_ticker(text)  # not in any list: try it as a ticker anyway
    except ValueError:
        st.warning(
            md(f'Nothing matches "{text}". Try a company or fund name, or a ticker like AAPL.')
        )
        return None


def _clear_search() -> None:
    """Widget callback: picking from the list replaces a search typed below it."""
    st.session_state["mk_query"] = ""
    st.session_state.pop("mk_pick", None)


def _lookup() -> None:
    directory = services.symbols()
    labels = _suggestions(directory)
    default = directory.get("SPY")
    choice = st.selectbox(
        "Look up a stock, fund, or index",
        labels,
        index=labels.index(default.label) if default and default.label in labels else None,
        key="mk_symbol",
        on_change=_clear_search,
        placeholder="Type a name or ticker, e.g. Apple or S&P 500",
        help="Suggestions come from the SEC's company list and Finnie's fund and index list.",
    )
    query = st.text_input(
        "Not in the list? Search by name or ticker",
        key="mk_query",
        placeholder="e.g. Nestle, or a ticker such as NSRGY",
        help="Searches every company on the SEC's list, then Yahoo Finance.",
    )
    if query.strip():
        ticker = _resolve(query, directory)
    else:
        ticker = choice.split(" · ", 1)[0] if choice else None
    if ticker is None:
        return
    market = services.context().market
    try:
        history = market.get_daily_history(ticker, HISTORY_DAYS)
        snapshot = technical_snapshot(history)
    except (MarketDataError, ValueError) as exc:
        logger.info("Ticker lookup failed", extra={"ticker": ticker, "error": str(exc)})
        st.warning(
            md(
                f"Couldn't find price history for {ticker}. Check the symbol (for example "
                "AAPL, VTI, or ^GSPC for the S&P 500). If it's right, the data providers may "
                "be unavailable for a moment."
            )
        )
        return
    cols = st.columns(4)
    try:
        quote = market.get_quote(ticker)
    except MarketDataError:  # no quote: fall back to the last daily close
        cols[0].metric("Last close", _amount(ticker, snapshot.price))
        when = f"{snapshot.as_of:%b} {snapshot.as_of.day}, {snapshot.as_of.year}"
        st.caption(f"Last close · {when} · {market_status().label}")
    else:
        change = f"{quote.change_percent:+.2f}%" if quote.change_percent is not None else None
        cols[0].metric(
            "Level" if _is_index(ticker) else "Price", _amount(ticker, quote.price), change
        )
        st.caption(price_time_label(quote.freshness.as_of))
    cols[1].metric(
        "52-week range",
        md(f"{_amount(ticker, snapshot.low_52w, 0)} to {_amount(ticker, snapshot.high_52w, 0)}"),
    )
    cols[2].metric("Trend", TRENDS.get(snapshot.trend, snapshot.trend))
    cols[3].metric(
        "RSI (14-day)", f"{snapshot.rsi_14:.0f}" if snapshot.rsi_14 is not None else "n/a"
    )
    if snapshot.volatility_30d is not None:
        st.caption(f"30-day volatility (annualized): {percent(snapshot.volatility_30d, 0)}")
    st.plotly_chart(charts.price_chart(history), width="stretch", key="mk_price")
    st.plotly_chart(charts.rsi_chart(history), width="stretch", key="mk_rsi")
    st.markdown(md("\n".join(f"- {note}" for note in snapshot.notes)))
    st.caption(freshness_caption([history.freshness]) or "")
    with st.expander(f"About {ticker}"):
        _company(market, ticker)
    _news(ticker)


def _company(market: Any, ticker: str) -> None:
    """Structured facts with their as-of date. Provider descriptions can be years old, so
    only the first sentence is shown, labelled with where it came from."""
    try:
        company = market.get_company_overview(ticker)
    except MarketDataError:
        st.caption("No company overview available.")
        return
    st.markdown(f"**{md(company.name)}**")
    facts = [
        ("Sector", company.sector),
        ("Industry", company.industry),
        ("Market cap", big_money(company.market_cap) if company.market_cap else None),
        ("P/E ratio", f"{company.pe_ratio:.1f}" if company.pe_ratio else None),
        (
            "Dividend yield",
            percent(company.dividend_yield, 2) if company.dividend_yield is not None else None,
        ),
    ]
    shown = [(label, value) for label, value in facts if value]
    if shown:
        st.markdown(md("  \n".join(f"{label}: **{value}**" for label, value in shown)))
    fetched = company.freshness.fetched_at
    st.caption(f"As of {fetched:%b %d, %Y} · {provider_name(company.freshness)}")
    sentence = first_sentence(company.description)
    if sentence:
        st.markdown(f"> {md(sentence)}")
        st.caption(
            f"Description from {provider_name(company.freshness)}. Company descriptions can be "
            "out of date."
        )


def render() -> None:
    """Draw the Markets tab: the market overview, then the ticker lookup."""
    st.subheader("Markets today")
    _overview()
    st.divider()
    st.subheader("Look up a stock, fund, or index")
    _lookup()
    st.caption("Market data is for learning. It isn't a recommendation to buy or sell anything.")
