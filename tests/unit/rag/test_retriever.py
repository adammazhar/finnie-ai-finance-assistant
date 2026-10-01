import numpy as np
import pytest

from src.core.config import RAGConfig
from src.rag import retriever as retriever_module
from src.rag.chunking import Chunk, build_chunks
from src.rag.index import VectorIndex
from src.rag.knowledge_base import SourceRef
from src.rag.retriever import Retriever, build_retriever
from tests.fakes.embeddings import FakeEmbeddings


@pytest.fixture
def retriever(kb, rag_config, embeddings):
    index = VectorIndex.build(build_chunks(rag_config, root=kb), embeddings, rag_config)
    return Retriever(index, embeddings, rag_config)


def ids(result):
    return [rc.chunk.id for rc in result.chunks]


def test_best_match_first_with_scores(retriever):
    result = retriever.retrieve("  When is a capital gain long-term?  ")
    assert result.query == "When is a capital gain long-term?"
    assert result.chunks[0].chunk.section == "Holding period"
    assert result.confident and not result.widened and result.latency_ms >= 0
    scores = [rc.score for rc in result.chunks]
    assert all(s >= 0.2 for s in scores)


def test_category_filter_keeps_glossary_by_default(retriever):
    result = retriever.retrieve("bond coupon interest", categories=["bonds_fixed_income"])
    categories = {rc.chunk.category for rc in result.chunks}
    assert categories <= {"bonds_fixed_income", "glossary"} and "glossary" in categories
    no_glossary = retriever.retrieve(
        "bond coupon interest", categories=["bonds_fixed_income"], include_glossary=False
    )
    assert {rc.chunk.category for rc in no_glossary.chunks} == {"bonds_fixed_income"}


def test_widens_when_category_has_too_few_matches(retriever, caplog):
    caplog.set_level("INFO")
    result = retriever.retrieve(
        "wash sale loss thirty days", categories=["stocks"], include_glossary=False
    )
    assert result.widened and result.chunks[0].chunk.category == "taxes"
    assert "Widened retrieval" in caplog.text


def test_nothing_relevant_returns_empty(retriever):
    result = retriever.retrieve("pizza dough recipe with basil")
    assert result.chunks == [] and not result.confident


def test_per_article_cap_and_k(kb, embeddings):
    config = RAGConfig(
        chunk_size=200,
        chunk_overlap=20,
        top_k=4,
        fetch_k=20,
        score_threshold=0.0,
        mmr_lambda=1.0,
        max_chunks_per_article=1,
    )
    index = VectorIndex.build(build_chunks(config, root=kb), embeddings, config)
    result = Retriever(index, embeddings, config).retrieve("bond prices interest rates", k=3)
    articles = [rc.chunk.article_id for rc in result.chunks]
    assert len(articles) == 3 and len(set(articles)) == 3


def make_chunk(i: int, article: str) -> Chunk:
    return Chunk(
        id=f"{article}#{i}",
        kind="article",
        article_id=article,
        title=article,
        category="stocks",
        section="S",
        text=f"text {i}",
        sources=[SourceRef(name="Src name", url="https://www.sec.gov/x")],
    )


def test_mmr_prefers_diverse_results():
    # chunks 0 and 1 are near-duplicates; chunk 2 is a little less relevant but different
    vectors = np.array([[1.0, 0.0, 0.0], [0.99, 0.14, 0.0], [0.8, 0.0, 0.6]])
    chunks = [make_chunk(0, "a"), make_chunk(1, "b"), make_chunk(2, "c")]
    index = VectorIndex(chunks, vectors, {})

    class Fixed(FakeEmbeddings):
        def embed_query(self, text):
            return [1.0, 0.0, 0.0]

    base = dict(top_k=2, fetch_k=3, score_threshold=0.0, max_chunks_per_article=2)
    diverse = Retriever(index, Fixed(), RAGConfig(mmr_lambda=0.3, **base)).retrieve("q")
    greedy = Retriever(index, Fixed(), RAGConfig(mmr_lambda=1.0, **base)).retrieve("q")
    assert ids(greedy) == ["a#0", "b#1"]
    assert ids(diverse) == ["a#0", "c#2"]


def test_input_validation(retriever):
    with pytest.raises(ValueError, match="empty"):
        retriever.retrieve("   ")
    with pytest.raises(ValueError, match="Unknown categories: astrology"):
        retriever.retrieve("bonds", categories=["astrology"])
    assert retriever.retrieve("bond", categories=["glossary"]).chunks


def test_build_retriever_and_cache(kb, rag_config, embeddings, tmp_path, monkeypatch):
    monkeypatch.setattr("src.rag.index.KB_ROOT", kb)
    monkeypatch.setattr("src.rag.chunking.KB_ROOT", kb)
    built = retriever_module.build_retriever(rag_config, embeddings, index_dir=tmp_path / "vs")
    assert len(built.index) > 0 and (tmp_path / "vs" / "meta.json").is_file()

    retriever_module.get_retriever.cache_clear()
    monkeypatch.setattr(retriever_module, "build_retriever", lambda: "sentinel")
    assert retriever_module.get_retriever() == "sentinel"
    assert retriever_module.get_retriever() == "sentinel"
    retriever_module.get_retriever.cache_clear()


def test_build_retriever_uses_global_defaults(kb, monkeypatch, tmp_path):
    from src.core import config as config_module

    monkeypatch.setattr("src.rag.index.KB_ROOT", kb)
    monkeypatch.setattr("src.rag.chunking.KB_ROOT", kb)
    monkeypatch.setattr(retriever_module, "get_embeddings", lambda: FakeEmbeddings())
    settings = config_module.load_settings()
    monkeypatch.setattr(retriever_module, "get_settings", lambda: settings)
    monkeypatch.setattr(type(settings), "resolve_path", lambda self, p: tmp_path / "vs")
    assert build_retriever().config == settings.rag
