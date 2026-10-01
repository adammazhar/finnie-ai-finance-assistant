from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.agents.context import AgentContext
from src.core.config import RAGConfig, load_settings
from src.core.models import AGENT_NAMES
from src.rag.retriever import Retriever
from src.web_app import services
from src.workflow.graph import FinnieAssistant
from tests.fakes import chunks as kb
from tests.fakes.llm import FakeChatModel
from tests.fakes.market_service import FakeMarketService
from tests.unit.workflow.conftest import ScriptedAgent, result

APP = str(Path(__file__).resolve().parents[3] / "src" / "web_app" / "app.py")
TIMEOUT_S = 60


def tiny_retriever() -> Retriever:
    chunks = [
        *kb.article(
            "funds_etfs-002",
            "funds_etfs",
            "Exchange-Traded Funds",
            {"What it is": "An exchange-traded fund is a basket of investments that trades."},
        ),
        kb.glossary("Expense ratio", "The yearly fee a fund charges."),
    ]
    config = RAGConfig(chunk_size=300, chunk_overlap=30, top_k=3, fetch_k=10, score_threshold=0.0)
    return kb.retriever(chunks, config)


@pytest.fixture(autouse=True)
def fresh_caches() -> Iterator[None]:
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture
def ui(monkeypatch, tmp_path):
    """Start the app with fakes: ``ui(routes=[...], agents={...})`` -> (AppTest, team, context)."""

    def factory(
        routes: list[Any] | None = None,
        agents: dict[str, Any] | None = None,
        market: Any | None = None,
        with_retriever: bool = True,
        assistant: Any | None = None,
        page: str = "Chat",
        onboarded: bool = True,
        session: dict[str, Any] | None = None,
    ) -> tuple[AppTest, dict[str, ScriptedAgent], AgentContext]:
        context = AgentContext(
            llm=FakeChatModel(responses=["Merged answer."]),
            fast_llm=FakeChatModel(responses=["summary"], structured_responses=list(routes or [])),
            market=market or FakeMarketService(),
            settings=load_settings(),
            retriever=tiny_retriever() if with_retriever else None,
        )
        team = {
            name: ScriptedAgent(name, (agents or {}).get(name) or result(name))
            for name in AGENT_NAMES
        }
        monkeypatch.setattr(services, "build_context", lambda: context)
        # keep the app from reconfiguring the test session's root logger
        monkeypatch.setattr("src.utils.logging.configure_logging", lambda *a, **k: None)
        monkeypatch.setattr(
            services,
            "build_assistant",
            lambda ctx: assistant or FinnieAssistant(ctx, agents=team),
        )
        st.cache_resource.clear()  # each app start builds its services from these fakes
        st.cache_data.clear()
        app = AppTest.from_file(APP, default_timeout=TIMEOUT_S)
        app.session_state["finnie_onboarded"] = onboarded
        app.session_state["finnie_page"] = page
        for key, value in (session or {}).items():
            app.session_state[key] = value
        app.run()
        assert not app.exception, [e.value for e in app.exception]
        return app, team, context

    return factory


def texts(elements: Any) -> list[str]:
    return [str(e.value) for e in elements]


def goto(app: AppTest, page: str) -> AppTest:
    """Click a page in the tab bar."""
    app.segmented_control(key="nav").set_value(page).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def ok(app: AppTest) -> AppTest:
    assert not app.exception, [e.value for e in app.exception]
    assert not app.main.error, texts(app.main.error)
    return app
