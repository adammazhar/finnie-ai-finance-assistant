"""Less common paths: empty market data, cash-only portfolios, and navigation callbacks."""

from streamlit.testing.v1 import AppTest

from src.core.models import Holding, Source
from src.data.models import BatchQuotes
from tests.fakes.market_service import FakeMarketService, fresh
from tests.unit.web_app.conftest import ok, texts
from tests.unit.workflow.conftest import result, route


def test_deselecting_a_tab_or_view_keeps_it_open():
    def script():
        import streamlit as st

        from src.web_app import state
        from src.web_app.tabs import knowledge

        st.session_state["nav"] = None
        st.session_state["finnie_page"] = "Goals"
        state.on_nav()
        st.session_state["kb_view"] = None
        st.session_state["kb_view_last"] = "Glossary"
        knowledge._keep_view()

    app = AppTest.from_function(script).run()
    assert app.session_state["nav"] == "Goals"
    assert app.session_state["kb_view"] == "Glossary"


def test_chat_freshness_note_and_a_source_without_an_article(ui):
    irs = Source(title="IRS: 401(k) limit", kind="knowledge_base", url="https://www.irs.gov/limits")
    agents = {
        "tax": result(
            "tax",
            "The limit is $24,500 [1].",
            sources=[irs],
            freshness=[fresh(mock=True)],
            data={"citations": {"kb": {"1": irs.url}}},
        )
    }
    app, _, _ = ui(routes=[route("tax")], agents=agents)
    ok(app.chat_input(key="chat_input").set_value("401k limit?").run())
    reply = app.chat_message[1]
    assert "Market data: demo data (live feed unavailable)" in texts(reply.caption)
    expected = (
        "**[1]** IRS: 401(k) limit  \n:gray[Original source: [irs.gov](https://www.irs.gov/limits)]"
    )
    assert expected in texts(reply.markdown)
    assert not [b for b in reply.button if b.key and "_source_" in b.key]  # nothing to open


class QuietMarket(FakeMarketService):
    """No quotes at all, short price history, and a bare company overview."""

    def get_quotes(self, tickers):
        return BatchQuotes(quotes={}, errors={t: "none" for t in tickers})

    def get_daily_history(self, ticker, days=252):
        return super().get_daily_history(ticker, 3)

    def get_company_overview(self, ticker):
        bare = super().get_company_overview(ticker)
        return bare.model_copy(
            update={"sector": None, "market_cap": None, "pe_ratio": None, "dividend_yield": None}
        )


def test_markets_with_almost_no_data(ui):
    app, _, _ = ui(page="Markets", market=QuietMarket())
    ok(app)
    assert not [m for m in app.metric if "(SPY)" in m.label]  # no index cards
    assert "mk_sectors" not in {c.key for c in app.get("plotly_chart")}
    assert not any("30-day volatility" in c for c in texts(app.caption))
    assert not any(t.startswith("Sector: ") for t in texts(app.markdown))
    assert not any(c.startswith("Description from") for c in texts(app.caption))


def test_cash_only_portfolio_and_a_csv_without_valid_rows(ui):
    app, _, _ = ui(
        page="Portfolio", session={"finnie_portfolio": [Holding(ticker="CASH", shares=500)]}
    )
    ok(app)
    charts = {c.key for c in app.get("plotly_chart")}
    assert "pf_performance" not in charts and "pf_corr" not in charts
    assert not any(c.startswith("Market data:") for c in texts(app.caption))
    app.file_uploader(key="pf_upload").upload(
        "bad.csv", b"ticker,shares\nVTI,abc\n", "text/csv"
    ).run()
    ok(app.button(key="pf_use_upload").click().run())
    assert not app.success and any(t.startswith("bad.csv: ") for t in texts(app.warning))
    assert app.session_state["finnie_portfolio"] == [Holding(ticker="CASH", shares=500)]


def test_portfolio_with_nothing_notable(ui, monkeypatch):
    from src.core.portfolio import fetch_and_analyze
    from src.web_app.tabs import portfolio

    quiet = fetch_and_analyze([Holding(ticker="VTI", shares=1)], FakeMarketService())
    quiet.observations = []
    monkeypatch.setattr(portfolio, "analyze", lambda *args: quiet)
    app, _, _ = ui(
        page="Portfolio", session={"finnie_portfolio": [Holding(ticker="VTI", shares=1)]}
    )
    ok(app)
    assert "**What stands out**" not in texts(app.markdown)


def test_missing_api_key_shows_setup_steps_not_a_stack_trace(monkeypatch):
    """A fresh clone (or a Docker run) without .env: the real wiring, no keys set."""
    import streamlit as st

    from tests.unit.web_app.conftest import APP

    monkeypatch.setattr("src.utils.logging.configure_logging", lambda *a, **k: None)
    st.cache_resource.clear()
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception, [e.value for e in app.exception]
    assert "## Finnie needs an API key to start" in texts(app.markdown)
    assert "OPENAI_API_KEY is not set" in app.error[0].value
    assert any("docker compose up" in t for t in texts(app.markdown))
    assert not app.sidebar.button  # nothing else renders


def _save_script(rows, upload_id=None, used_id=None, saved=False):
    """An app script that calls the Portfolio tab's Save with the given table and upload."""
    from types import SimpleNamespace

    import pandas as pd
    import streamlit as st

    from src.core.models import Holding
    from src.web_app import state
    from src.web_app.tabs import portfolio

    st.session_state["finnie_portfolio"] = [Holding(ticker="VTI", shares=1)] if saved else []
    if used_id:
        st.session_state[portfolio.UPLOAD_USED] = used_id
    upload = SimpleNamespace(file_id=upload_id) if upload_id else None
    portfolio._save(pd.DataFrame(rows, columns=["ticker", "shares", "cost_basis"]), upload)
    st.session_state["saved_after"] = [h.ticker for h in state.portfolio()]


def save_app(**kwargs):
    return AppTest.from_function(_save_script, kwargs=kwargs, default_timeout=60).run()


def test_save_warns_about_a_file_that_was_chosen_but_not_loaded(monkeypatch):
    monkeypatch.setattr("src.web_app.state.set_portfolio", lambda holdings: None)
    app = save_app(rows=[["AAPL", 1, None]], upload_id="new")
    assert "haven't loaded it yet" in app.warning[0].value
    assert not save_app(rows=[["AAPL", 1, None]], upload_id="new", used_id="new").warning


def test_save_explains_an_empty_table_and_rejected_rows(monkeypatch):
    monkeypatch.setattr("src.web_app.state.set_portfolio", lambda holdings: None)
    assert "The table is empty" in save_app(rows=[]).info[0].value
    bad = save_app(rows=[["AAPL", -5, None]])
    assert bad.warning[0].value.endswith("was skipped: shares must be more than 0")
    assert not save_app(rows=[], saved=True).info  # an empty table clears a saved portfolio
