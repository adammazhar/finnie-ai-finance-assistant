from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from src.core.config import RAGConfig, load_settings
from src.mcp_server.server import Services
from src.rag.retriever import Retriever
from tests.fakes import chunks as kb
from tests.fakes.market_service import FakeMarketService


@pytest.fixture
def retriever() -> Retriever:
    chunks = [
        *kb.article(
            "funds_etfs-001",
            "funds_etfs",
            "What Is an ETF?",
            {
                "How it works": "An exchange-traded fund holds many investments and trades "
                "on an exchange like a stock."
            },
        ),
        *kb.article(
            "taxes-002",
            "taxes",
            "Capital Gains Basics",
            {"Holding period": "A capital gain is long-term when held more than one year."},
        ),
        kb.glossary("Expense ratio", "The yearly fee a fund charges, as a share of assets."),
    ]
    config = RAGConfig(
        chunk_size=300,
        chunk_overlap=30,
        top_k=3,
        fetch_k=10,
        score_threshold=0.0,
        max_chunks_per_article=2,
    )
    return kb.retriever(chunks, config)


@pytest.fixture
def make_services(retriever: Retriever) -> Callable[..., Services]:
    def factory(market: Any = None, *, search: bool = True) -> Services:
        return Services(
            market=market or FakeMarketService(),
            settings=load_settings(),
            retriever=lambda: retriever if search else None,
        )

    return factory


@pytest.fixture
def services(make_services: Callable[..., Services]) -> Services:
    return make_services()
