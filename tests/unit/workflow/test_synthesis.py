from src.workflow.synthesis import MERGE_PROMPT, synthesize, tidy_citations, unify_citations
from tests.fakes.llm import FakeChatModel
from tests.unit.workflow.conftest import freshness, kb_source, market_source, news_source, result


def cited(agent, answer, kb=(), news=(), extra=()):
    """An agent result whose local [n]/[N#] markers map to the given sources."""
    sources = [*kb, *news, *extra]
    citations = {
        "kb": {str(i): s.article_id for i, s in enumerate(kb, 1)},
        "news": {str(i): s.url for i, s in enumerate(news, 1)},
    }
    return result(agent, answer, sources=sources, data={"citations": citations})


def test_markers_renumbered_into_one_list():
    etf, fees = kb_source("funds-001"), kb_source("funds-002")
    article = news_source("https://example.com/a")
    a = cited("finance_qa", "ETFs pool money [1]. Fees matter [2].", kb=[etf, fees])
    b = cited("news", "Funds saw inflows [N1]. Fees again [1].", kb=[fees], news=[article])
    unified = unify_citations([a, b])
    assert [s.article_id or s.url for s in unified.sources] == [
        "funds-001",
        "funds-002",
        "https://example.com/a",
    ]
    assert unified.answers == {
        "finance_qa": "ETFs pool money [1]. Fees matter [2].",
        "news": "Funds saw inflows [3]. Fees again [2].",
    }


def test_unresolvable_markers_dropped_and_market_data_not_listed():
    """Market data goes in the freshness note, not the sources list."""
    quote = market_source()
    r = cited("market", "TSLA is up [1] and [N4].", extra=[quote])
    unified = unify_citations([r])
    assert unified.answers["market"] == "TSLA is up  and ."
    assert unified.sources == []


def test_tidy_citations_dedupes_and_renumbers_by_first_use():
    a, b, c = kb_source("a"), kb_source("b"), kb_source("c")
    text, sources = tidy_citations("Fees [3][1][3] matter. Also [1][1] and [9].", [a, b, c])
    assert text == "Fees [1][2] matter. Also [2] and ."
    assert sources == [c, a]  # b was never cited
    assert tidy_citations("No markers.", [a]) == ("No markers.", [])


def test_merged_answer_lists_only_what_it_cites():
    llm = FakeChatModel(responses=["Taxes [2][2][2]."])  # the merge dropped the ETF citation
    a = cited("finance_qa", "ETFs [1].", kb=[kb_source("funds-001")])
    b = cited("tax", "Taxes [1].", kb=[kb_source("taxes-002")])
    out = synthesize([a, b], llm)
    assert out.text == "Taxes [1]."
    assert [s.article_id for s in out.sources] == ["taxes-002"]


def test_results_without_citation_maps():
    unified = unify_citations([result("tax", "Plain answer [1].")])
    assert unified.answers["tax"] == "Plain answer ."
    assert unified.sources == []


def test_single_result_text_is_tidied():
    out = synthesize([result("tax", "Plain answer [1].")], None)
    assert out is not None and out.text == "Plain answer."


def test_no_successful_results():
    assert synthesize([result("tax", error="boom")], FakeChatModel()) is None
    assert synthesize([], None) is None


def test_single_result_passes_through_without_llm():
    llm = FakeChatModel()
    fresh = freshness()
    r = cited("finance_qa", "An ETF is a fund [1].", kb=[kb_source("funds-001")])
    r.freshness = [fresh]
    out = synthesize([r, result("news", error="down")], llm)
    assert out is not None
    assert out.text == "An ETF is a fund [1]."
    assert out.agents == ["finance_qa"] and out.freshness == [fresh]
    assert out.merged_by_llm is False
    assert llm.calls == []


def test_several_results_merged_by_llm():
    llm = FakeChatModel(responses=["Merged: ETFs [1], taxes [2], invented [9]."])
    a = cited("finance_qa", "ETFs [1].", kb=[kb_source("funds-001")])
    b = cited("tax", "Taxes [1].", kb=[kb_source("taxes-002")])
    out = synthesize([a, b], llm)
    assert out is not None and out.merged_by_llm
    assert out.text == "Merged: ETFs [1], taxes [2], invented."
    system, human = llm.calls[0]
    assert system.content == MERGE_PROMPT
    assert human.content == "## Concepts\nETFs [1].\n\n## Taxes\nTaxes [2]."


def test_merge_failure_or_blank_falls_back_to_sections():
    a, b = result("market", "Up 1%."), result("news", "Earnings beat.")
    for llm in (
        FakeChatModel(responses=[RuntimeError("down")]),
        FakeChatModel(responses=[" "]),
        None,
    ):
        out = synthesize([a, b], llm)
        assert out is not None and not out.merged_by_llm
        assert out.text == "## Markets\nUp 1%.\n\n## News\nEarnings beat."
