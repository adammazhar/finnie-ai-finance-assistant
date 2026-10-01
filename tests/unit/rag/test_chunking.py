import pytest

from src.core.config import RAGConfig
from src.rag.chunking import INTRO_SECTION, build_chunks, chunk_article, chunk_glossary
from src.rag.knowledge_base import load_articles, load_glossary
from tests.unit.rag.conftest import write_article

# Real chunking imports LangChain's text splitters (and with them torch), which takes about
# 20 s per process. One xdist worker runs all such tests, so only it pays that cost.
pytestmark = pytest.mark.xdist_group("text_splitters")


def test_article_chunks_follow_sections(kb, rag_config):
    article = next(a for a in load_articles(kb) if a.meta.id == "bonds_fixed_income-001")
    chunks = chunk_article(article, rag_config)
    assert [c.section for c in chunks] == [INTRO_SECTION, "Bond basics", "Bond prices and rates"]
    assert [c.id for c in chunks] == [f"bonds_fixed_income-001#{i}" for i in range(3)]
    first = chunks[0]
    assert first.kind == "article" and first.category == "bonds_fixed_income"
    assert first.difficulty == "beginner" and first.sources[0].url.endswith(
        "bonds_fixed_income-001"
    )
    assert (
        first.embed_text == "What Is a Bond? > Overview: Bonds are loans investors make to issuers."
    )
    assert first.category_label == "Bonds and Fixed Income"
    assert "##" not in chunks[1].text  # headings are stripped from the text itself


def test_long_sections_are_split_with_overlap(tmp_path):
    sentence = "Diversification spreads money across many investments to reduce risk. "
    write_article(
        tmp_path,
        "portfolio_management-001",
        "portfolio_management",
        "Diversification",
        {"Why it works": sentence * 20, "Sub": "Short."},
    )
    config = RAGConfig(chunk_size=300, chunk_overlap=60, top_k=1, fetch_k=1)
    chunks = chunk_article(load_articles(tmp_path)[0], config)
    long_parts = [c for c in chunks if c.section == "Why it works"]
    assert len(long_parts) >= 4 and all(len(c.text) <= 300 for c in long_parts)
    # consecutive pieces overlap so a sentence cut at a boundary still appears whole somewhere
    assert any(long_parts[0].text[-30:] in long_parts[1].text for _ in [0])


def test_subsections_are_labelled(tmp_path):
    path = write_article(tmp_path, "stocks-002", "stocks", "Order Types", {"Types": "Intro text."})
    path.write_text(
        path.read_text(encoding="utf-8") + "\n### Limit orders\n\nA limit order sets a price.\n",
        encoding="utf-8",
    )
    chunks = chunk_article(load_articles(tmp_path)[0], RAGConfig(top_k=1, fetch_k=1))
    assert chunks[-1].section == "Types / Limit orders"


def test_glossary_chunks(kb):
    chunks = chunk_glossary(load_glossary(kb))
    assert [c.id for c in chunks] == ["glossary:coupon", "glossary:bond"]
    coupon = chunks[0]
    assert coupon.kind == "glossary" and coupon.category == "glossary"
    assert coupon.category_label == "Glossary" and coupon.section == "Definition"
    assert coupon.text.endswith("Related terms: Bond.")
    assert chunks[1].text == "A loan an investor makes to a borrower for interest."


def test_build_chunks(kb, rag_config, tmp_path):
    chunks = build_chunks(rag_config, root=kb)
    assert len([c for c in chunks if c.kind == "glossary"]) == 2
    assert {c.article_id for c in chunks if c.kind == "article"} == {
        "bonds_fixed_income-001",
        "stocks-001",
        "taxes-001",
    }
    (kb / "glossary.yaml").unlink()
    assert all(c.kind == "article" for c in build_chunks(rag_config, root=kb))
    assert build_chunks(rag_config, articles=[], root=tmp_path / "empty") == []


def test_whitespace_only_pieces_are_skipped(kb, rag_config, monkeypatch):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    monkeypatch.setattr(RecursiveCharacterTextSplitter, "split_text", lambda self, t: ["  ", t])
    article = next(a for a in load_articles(kb) if a.meta.id == "stocks-001")
    chunks = chunk_article(article, rag_config)
    assert all(c.text.strip() for c in chunks)
    assert [c.id for c in chunks] == [f"stocks-001#{i}" for i in range(len(chunks))]
