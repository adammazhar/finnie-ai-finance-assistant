import pytest
from langchain_core.messages import AIMessage

from src.agents.base import AgentRequest, RunState
from src.agents.context import AgentContext
from src.core.config import RAGConfig, load_settings
from src.rag.retriever import Retriever
from tests.fakes import chunks as kb
from tests.fakes.llm import FakeChatModel
from tests.fakes.market_service import FakeMarketService


@pytest.fixture
def retriever() -> Retriever:
    """Four short articles and one glossary term (see tests/fakes/chunks.py)."""
    chunks = [
        *kb.article(
            "portfolio_management-001",
            "portfolio_management",
            "Diversification",
            {
                "Why it matters": "Diversification spreads money across many investments "
                "so one loss hurts less."
            },
        ),
        *kb.article(
            "taxes-002",
            "taxes",
            "Capital Gains Basics",
            {
                "Holding period": "A capital gain is long-term when held more than one "
                "year, sold after the anniversary."
            },
        ),
        *kb.article(
            "stocks-004",
            "stocks",
            "Price to Earnings Ratio",
            {
                "What it is": "The price to earnings ratio compares a stock price with "
                "company earnings per share."
            },
        ),
        *kb.article(
            "market_economics-003",
            "market_economics",
            "Bull and Bear Markets",
            {
                "Definitions": "A bear market is a decline of twenty percent or more from a "
                "recent high."
            },
        ),
        kb.glossary("Expense ratio", "The yearly fee a fund charges, as a share of assets."),
    ]
    config = RAGConfig(
        chunk_size=300,
        chunk_overlap=30,
        top_k=3,
        fetch_k=10,
        score_threshold=0.2,
        max_chunks_per_article=2,
    )
    return kb.retriever(chunks, config)


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
