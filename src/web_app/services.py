"""Long-lived objects shared by every session, created once per process.

``build_context`` and ``build_assistant`` are module attributes so tests can replace them
with fakes before the app runs.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from src.agents.context import AgentContext, build_agent_context
from src.core.indicators import MarketOverview, build_market_overview
from src.rag.knowledge_base import Article, Glossary, load_articles, load_glossary
from src.workflow.graph import FinnieAssistant

logger = logging.getLogger(__name__)

build_context = build_agent_context


def build_assistant(context: AgentContext) -> FinnieAssistant:
    return FinnieAssistant(context)


@st.cache_resource(show_spinner="Starting Finnie…")
def context() -> AgentContext:
    return build_context()


@st.cache_resource(show_spinner=False)
def assistant() -> FinnieAssistant:
    return build_assistant(context())


@st.cache_resource(show_spinner=False)
def articles() -> list[Article]:
    return load_articles()


@st.cache_resource(show_spinner=False)
def glossary() -> Glossary:
    return load_glossary()


@st.cache_data(ttl=300, show_spinner="Loading market data…")
def market_overview() -> MarketOverview:
    """Indices, sectors, and the S&P 500 read. The market service caches each quote too."""
    return build_market_overview(context().market)


def market_status() -> list[Any]:
    status = getattr(context().market, "provider_status", None)
    return list(status()) if callable(status) else []
