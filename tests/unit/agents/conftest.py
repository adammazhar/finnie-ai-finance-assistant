from pathlib import Path

import pytest
import yaml
from langchain_core.messages import AIMessage

from src.agents.base import AgentRequest, RunState
from src.agents.context import AgentContext
from src.core.config import RAGConfig, load_settings
from src.rag.chunking import build_chunks
from src.rag.index import VectorIndex
from src.rag.retriever import Retriever
from tests.fakes.embeddings import FakeEmbeddings
from tests.fakes.llm import FakeChatModel
from tests.fakes.market_service import FakeMarketService
from tests.unit.rag.conftest import write_article


@pytest.fixture
def tiny_kb(tmp_path) -> Path:
    root = tmp_path / "kb"
    write_article(
        root,
        "portfolio_management-001",
        "portfolio_management",
        "Diversification",
        {
            "Why it matters": "Diversification spreads money across many investments "
            "so one loss hurts less."
        },
    )
    write_article(
        root,
        "taxes-002",
        "taxes",
        "Capital Gains Basics",
        {
            "Holding period": "A capital gain is long-term when held more than one "
            "year, sold after the anniversary."
        },
    )
    write_article(
        root,
        "stocks-004",
        "stocks",
        "Price to Earnings Ratio",
        {
            "What it is": "The price to earnings ratio compares a stock price with "
            "company earnings per share."
        },
    )
    write_article(
        root,
        "market_economics-003",
        "market_economics",
        "Bull and Bear Markets",
        {"Definitions": "A bear market is a decline of twenty percent or more from a recent high."},
    )
    glossary = {
        "sources": [{"name": "Investor.gov glossary", "url": "https://www.investor.gov/glossary"}],
        "last_reviewed": "2026-09-30",
        "terms": [
            {
                "term": "Expense ratio",
                "definition": "The yearly fee a fund charges, as a share of assets.",
            }
        ],
    }
    (root / "glossary.yaml").write_text(yaml.safe_dump(glossary), encoding="utf-8")
    return root


@pytest.fixture
def retriever(tiny_kb) -> Retriever:
    config = RAGConfig(
        chunk_size=300,
        chunk_overlap=30,
        top_k=3,
        fetch_k=10,
        score_threshold=0.2,
        max_chunks_per_article=2,
    )
    embeddings = FakeEmbeddings()
    index = VectorIndex.build(build_chunks(config, root=tiny_kb), embeddings, config)
    return Retriever(index, embeddings, config)


@pytest.fixture
def market() -> FakeMarketService:
    return FakeMarketService()


@pytest.fixture
def make_context(retriever, market):
    settings = load_settings()

    def factory(*responses, retriever_override="default", **kwargs) -> AgentContext:
        llm = FakeChatModel(responses=list(responses) or ["ok"])
        return AgentContext(
            llm=llm,
            market=kwargs.get("market", market),
            settings=settings,
            retriever=retriever if retriever_override == "default" else retriever_override,
        )

    return factory


@pytest.fixture
def state():
    return RunState(request=AgentRequest(query="test question"), agent="finance_qa")


def tool_call(name: str, args: dict | None = None, call_id: str = "call-1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": call_id}])
