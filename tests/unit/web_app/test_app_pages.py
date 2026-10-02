"""Portfolio, Markets, Goals, and Knowledge pages, run with AppTest and fakes."""

import pandas as pd

from src.core.models import Holding
from src.data.models import NewsArticle
from src.rag.retriever import RetrievalResult
from src.web_app import services
from src.web_app.services import build_assistant
from src.web_app.services import data_path as real_data_path
from src.web_app.tabs.portfolio import EXPLAIN_PROMPT, _rows_to_holdings
from src.workflow.graph import FinnieAssistant
from tests.fakes.market_service import FakeMarketService
from tests.unit.web_app.conftest import goto, ok, texts
from tests.unit.workflow.conftest import route

VTI = Holding(ticker="VTI", shares=10)


def page(ui, name, **kwargs):
    app, team, context = ui(page=name, **kwargs)
    return ok(app), team, context


# ---- portfolio ----------------------------------------------------------------------------


def test_portfolio_empty_state(ui):
    app, _, _ = page(ui, "Portfolio")
    assert any("Add holdings above" in t for t in texts(app.info))


def test_load_sample_and_see_the_analysis(ui):
    app, _, _ = page(ui, "Portfolio")
    app.selectbox(key="pf_sample").set_value("Three-fund beginner").run()
    ok(app.button(key="pf_load_sample").click().run())
    assert any("Loaded 3 holdings from Three-fund beginner" in t for t in texts(app.success))
    assert [m.label for m in app.metric] == [
        "Total value",
        "Diversification",
        "Risk level",
        "Portfolio expense ratio",
    ]
    assert app.metric[3].value == "0.03%"
    assert "Portfolio expense ratio: 0.03% (funds only; VTI and BND charge 0.03% each)." in texts(
        app.caption
    )
    assert any("No price found for: VXUS" in t for t in texts(app.warning))  # not in the fake
    charts = {c.key for c in app.get("plotly_chart")}
    assert {"pf_donut", "pf_sectors", "pf_performance"} <= charts
    assert any(c.startswith("A back-test: today's holdings") for c in texts(app.caption))
    assert app.session_state["finnie_portfolio"][0].ticker == "VTI"


def test_explain_button_sends_the_portfolio_to_chat(ui):
    app, team, _ = page(
        ui, "Portfolio", routes=[route("portfolio")], session={"finnie_portfolio": [VTI]}
    )
    ok(app.button(key="pf_explain").click().run())
    assert app.segmented_control(key="nav").value == "Chat"
    assert team["portfolio"].requests[0].query == EXPLAIN_PROMPT
    assert team["portfolio"].requests[0].portfolio == [VTI]
    assert app.chat_message[0].markdown[0].value == EXPLAIN_PROMPT


def test_portfolio_without_prices_or_history(ui):
    unpriced = {"finnie_portfolio": [Holding(ticker="ZZZZ", shares=1)]}
    app, _, _ = ui(page="Portfolio", session=unpriced)  # an error is the expected result
    assert any("No prices" in t for t in texts(app.main.error))
    app, _, _ = page(
        ui,
        "Portfolio",
        market=FakeMarketService(fail={"get_daily_history"}),
        session={"finnie_portfolio": [VTI]},
    )
    assert "Not enough price history to measure past-year risk." in texts(app.info)
    assert "pf_performance" not in {c.key for c in app.get("plotly_chart")}


def test_upload_and_save_holdings(ui):
    app, _, _ = page(ui, "Portfolio")
    app.file_uploader(key="pf_upload").upload(
        "mine$.csv", b"ticker,shares\nVTI,10\nBND,abc\n", "text/csv"
    ).run()
    ok(app.button(key="pf_use_upload").click().run())
    assert any(r"Loaded 1 holdings from mine\$.csv" in t for t in texts(app.success))
    assert any(t.startswith(r"mine\$.csv: ") for t in texts(app.warning))
    ok(app.button(key="pf_save").click().run())
    assert app.session_state["finnie_portfolio"] == [Holding(ticker="VTI", shares=10)]


def test_editor_rows_become_holdings():
    frame = pd.DataFrame(
        [
            {"ticker": "vti", "shares": 10, "cost_basis": 2500.0},
            {"ticker": "BND", "shares": 5, "cost_basis": float("nan")},
            {"ticker": "", "shares": 1, "cost_basis": None},  # blank row: ignored
            {"ticker": "BAD!", "shares": 1, "cost_basis": None},
        ]
    )
    holdings, problems = _rows_to_holdings(frame)
    assert holdings == [
        Holding(ticker="VTI", shares=10, cost_basis=2500.0),
        Holding(ticker="BND", shares=5),
    ]
    assert len(problems) == 1 and problems[0].startswith("Row 4 (BAD!) was skipped")


# ---- markets ------------------------------------------------------------------------------


class DescribedMarket(FakeMarketService):
    def get_company_overview(self, ticker):
        overview = super().get_company_overview(ticker)
        return overview.model_copy(
            update={
                "industry": "Consumer Electronics",
                "description": "Apple designs phones. In fiscal 2020 it earned $274.5 billion.",
            }
        )


def test_market_overview_and_ticker_lookup(ui):
    plain = NewsArticle(title="Plain headline")
    app, _, _ = page(ui, "Markets", market=DescribedMarket(news=[plain]))
    labels = [m.label for m in app.metric]
    assert "S&P 500 (^GSPC)" in labels and "Price" in labels and "52-week range" in labels
    sp500 = next(m for m in app.metric if m.label == "S&P 500 (^GSPC)")
    assert sp500.value == "6,745.12"
    assert r"Tracked by SPY: \$600.00 +1.01%" in texts(app.caption)
    # the price's time, whether it's live or delayed, and whether the market is open
    timing = [c for c in texts(app.caption) if " ET · Market " in c]
    assert len(timing) == 2  # under "Markets today" and on the lookup result
    assert {c.key for c in app.get("plotly_chart")} == {"mk_sectors", "mk_price", "mk_rsi"}
    captions = texts(app.caption)
    assert any(c.startswith("Market data: ") for c in captions)
    assert any(c.startswith("News fetched ") for c in captions)
    markdown = texts(app.markdown)
    facts = (
        "Sector: **Technology**  \nIndustry: **Consumer Electronics**  \n"
        "Market cap: **\\$3.4T**  \nP/E ratio: **33.2**  \nDividend yield: **0.44%**"
    )
    assert facts in markdown
    assert "> Apple designs phones." in markdown  # first sentence only, not the dated revenue
    assert any(c.startswith("Description from Yahoo Finance") for c in captions)
    assert any(c.startswith("As of ") for c in captions)
    assert "- Plain headline" in markdown


def test_ticker_lookup_handles_bad_input_and_outages(ui):
    app, _, _ = page(ui, "Markets", market=FakeMarketService(news=[]))
    assert "No recent news found for SPY." in texts(app.caption)
    app.text_input(key="mk_ticker").set_value("not a ticker!").run()
    assert any("Enter a ticker symbol" in t for t in texts(app.warning))
    app.text_input(key="mk_ticker").set_value("ZZZZ").run()
    assert any("Couldn't load price history for ZZZZ" in t for t in texts(app.warning))
    market = FakeMarketService(fail={"get_quotes", "get_news", "get_company_overview"})
    app, _, _ = ui(page="Markets", market=market)
    assert any("Market data is unavailable right now" in t for t in texts(app.main.error))
    assert "News is unavailable right now." in texts(app.caption)
    assert "No company overview available." in texts(app.caption)


# ---- goals --------------------------------------------------------------------------------


def test_goal_projection_states_the_chance_in_words(ui):
    app, _, _ = page(ui, "Goals")
    assert app.checkbox(key="goal_include_portfolio").disabled
    app.number_input(key="goal_target").set_value(5_000_000.0)
    app.slider(key="goal_years").set_value(5)
    ok(app.button(key="goal_run").click().run())
    markdown = texts(app.markdown)
    assert r"### Retirement: \$5,000,000 in 5 years" in markdown
    assert "#### Chance of reaching this goal: under 1%" in markdown
    assert any(c.startswith("How to read the chance: a median") for c in texts(app.caption))
    assert any("more time to save, a higher monthly contribution" in t for t in texts(app.info))
    assert {c.key for c in app.get("plotly_chart")} == {"goal_gauge", "goal_fan"}
    assert [m.label for m in app.metric][:3] == [
        "Poor markets (P10)",
        "Median",
        "Good markets (P90)",
    ]


def test_include_saved_portfolio_with_an_editable_amount(ui):
    app, _, _ = page(ui, "Goals", session={"finnie_portfolio": [VTI]})  # $3,000 in the fake
    assert r"Your saved portfolio is worth \$3,000.00; it isn't counted." in texts(app.caption)
    app.checkbox(key="goal_include_portfolio").check().run()
    amount = app.number_input(key="goal_portfolio_amount")
    assert amount.value == 3000.0
    amount.set_value(1200.0).run()
    app.number_input(key="goal_other").set_value(800.0).run()
    assert r"Starting balance for the projection: \$2,000.00" in texts(app.caption)
    app.number_input(key="goal_target").set_value(20_000.0)
    ok(app.button(key="goal_run").click().run())
    assumptions = " ".join(texts(app.markdown))
    assert r"Starting balance \$2,000.00, \$500 a month for 25 years" in assumptions
    assert not any("odds are low" in t for t in texts(app.info))


# ---- knowledge ----------------------------------------------------------------------------


def test_search_is_sorted_labelled_and_links_to_the_article(ui):
    app, _, _ = page(ui, "Knowledge")
    app.text_input(key="kb_query").set_value("what is an exchange-traded fund?").run()
    titles = [t for t in texts(app.markdown) if t.startswith("**")]
    assert titles[0].startswith("**Exchange-Traded Funds")
    captions = texts(app.caption)
    assert any(c.split(" · ")[0] in {"Strong match", "Good match", "Related"} for c in captions)
    assert not any("relevance" in c for c in captions)
    hits = [b for b in app.button if b.key and b.key.startswith("kb_hit_")]
    article = next(b for b in hits if b.label == "Read article")
    ok(article.click().run())
    assert app.segmented_control(key="kb_view").value == "Browse"
    assert app.selectbox(key="kb_article").value == "funds_etfs-002"


def test_search_edge_cases(ui):
    app, _, context = page(ui, "Knowledge", with_retriever=False)
    app.text_input(key="kb_query").set_value("bonds").run()
    assert any("index isn't built" in t for t in texts(app.warning))

    class Empty:
        def retrieve(self, query, k):
            return RetrievalResult(query=query, categories=None, chunks=[])

    context.retriever = Empty()
    app.text_input(key="kb_query").set_value("lasagna").run()
    assert any("Nothing in the knowledge base matches" in t for t in texts(app.info))


def test_glossary_hit_opens_the_glossary(ui):
    app, _, _ = page(ui, "Knowledge")
    app.text_input(key="kb_query").set_value("yearly fee a fund charges").run()
    term = next(
        b
        for b in app.button
        if b.key and b.key.startswith("kb_hit_") and b.label == "Open in Glossary"
    )
    ok(term.click().run())
    assert app.segmented_control(key="kb_view").value == "Glossary"
    assert app.text_input(key="kb_term").value == "Expense ratio"


def test_browse_and_glossary(ui):
    app, _, _ = page(ui, "Knowledge")
    app.segmented_control(key="kb_view").set_value("Browse").run()
    app.selectbox(key="kb_category").set_value("investing_basics").run()
    markdown = texts(app.markdown)
    assert any(t.startswith("## ") for t in markdown)  # an article is open
    assert not any("$5,500" in t and "\\$" not in t for t in markdown)  # dollars escaped
    app.segmented_control(key="kb_view").set_value(None).run()  # re-clicking keeps the view
    assert app.segmented_control(key="kb_view").value == "Browse"
    app.segmented_control(key="kb_view").set_value("Glossary").run()
    app.text_input(key="kb_term").set_value("coupon").run()
    markdown = texts(app.markdown)
    assert any(t.startswith("**Coupon") and r"\$1,000" in t for t in markdown)
    assert "**Glossary sources**" in markdown
    assert sum(t.startswith("- [Investor.gov") for t in markdown) <= 1  # listed once


def test_category_without_articles(ui, monkeypatch):
    monkeypatch.setattr(services, "load_articles", lambda: [])
    app, _, _ = page(ui, "Knowledge", session={"kb_view": "Browse"})
    assert "No articles in this category yet." in texts(app.info)


# ---- app-level ----------------------------------------------------------------------------


def test_a_failing_page_shows_a_message(ui, monkeypatch):
    def broken():
        raise RuntimeError("boom")

    monkeypatch.setattr("src.web_app.tabs.markets.render", broken)
    app, _, _ = ui(page="Markets")
    assert "This page couldn't load just now. Please try again in a moment." in texts(
        app.main.error
    )
    goto(app, "Goals")
    assert not app.main.error


def test_default_assistant_builder(ui):
    _, _, context = ui()
    assert isinstance(build_assistant(context), FinnieAssistant)  # the unpatched builder


def test_saved_data_goes_to_the_configured_git_ignored_file(ui):
    from src.core.config import PROJECT_ROOT

    ui()
    assert real_data_path() == PROJECT_ROOT / "data" / "app" / "finnie.sqlite"


def test_market_fallbacks_without_index_levels_or_quotes(ui, monkeypatch):
    from tests.fakes import market_service

    for index in ("^GSPC", "^NDX", "^DJI", "^RUT"):
        monkeypatch.delitem(market_service.PRICES, index)
    app, _, _ = page(ui, "Markets", market=FakeMarketService(fail={"get_quote"}))
    labels = [m.label for m in app.metric]
    assert "S&P 500 via SPY" in labels  # the ETF alone when the index quote is missing
    assert "Last close" in labels  # the daily close when the live quote fails
    assert any(c.startswith("Last close · ") for c in texts(app.caption))


def scroll_scripts(app):
    return [e.proto.srcdoc for e in app.get("iframe") if "finnie-answer-" in e.proto.srcdoc]


def test_buttons_that_ask_in_chat_scroll_to_the_start_of_the_answer(ui):
    app, _, _ = page(
        ui, "Portfolio", routes=[route("portfolio")], session={"finnie_portfolio": [VTI]}
    )
    ok(app.button(key="pf_explain").click().run())
    # after the answer, the page scrolls to where it starts (not the bottom)
    [script] = scroll_scripts(app)
    assert "'finnie-answer-' + 1" in script and "scrollIntoView({block: 'start'})" in script
    assert '<div id="finnie-answer-1"></div>' in texts(app.markdown)
    # typing in the chat box doesn't jump anywhere, and the scroll happens once
    app.chat_input(key="chat_input").set_value("And bonds?").run()
    assert not scroll_scripts(app)
