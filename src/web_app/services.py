"""Long-lived objects shared by every session, created once per process.

``build_context`` and ``build_assistant`` are module attributes so tests can replace them
with fakes before the app runs.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agents.context import AgentContext, build_agent_context
from src.core.indicators import MarketOverview, build_market_overview
from src.data.symbols import SymbolDirectory, SymbolMatch, get_symbol_directory, yahoo_search
from src.rag.knowledge_base import Article, Glossary, load_articles, load_glossary
from src.web_app.storage import AppStore
from src.workflow.graph import FinnieAssistant

logger = logging.getLogger(__name__)

build_context = build_agent_context
symbol_fallback = yahoo_search  # tests replace it, so they never reach the network


def data_path() -> Path:
    """The SQLite file for per-browser data and workflow memory (git-ignored)."""
    settings = context().settings
    return settings.resolve_path(settings.app.data_path)


def build_assistant(context: AgentContext) -> FinnieAssistant:
    """Workflow memory in SQLite, so conversations continue after an app restart."""
    path = data_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(path, check_same_thread=False))
    saver.setup()
    return FinnieAssistant(context, checkpointer=saver)


@st.cache_resource(show_spinner=False)
def store() -> AppStore:
    """The per-browser SQLite store, shared by all sessions."""
    return AppStore(data_path())


@st.cache_resource(
    show_spinner="Starting Finnie… The first start after a restart loads the AI models and "
    "the knowledge base, which takes about 20 seconds."
)
def context() -> AgentContext:
    """The models, market data service, retriever, and settings, built once per process."""
    return build_context()


@st.cache_resource(show_spinner=False)
def assistant() -> FinnieAssistant:
    """The multi-agent workflow, with its memory in the same SQLite file."""
    return build_assistant(context())


@st.cache_resource(show_spinner=False)
def symbols() -> SymbolDirectory:
    """Indexes, Finnie's funds, and the SEC's company list, for the Markets name search."""
    return get_symbol_directory()


@st.cache_data(ttl=24 * 3600, show_spinner="Searching…")
def online_symbol_search(query: str) -> list[SymbolMatch]:
    """Yahoo Finance's search, for names the local list doesn't have (cached for a day)."""
    return symbol_fallback(query)


@st.cache_resource(show_spinner=False)
def articles() -> list[Article]:
    """The knowledge base articles, loaded once."""
    return load_articles()


@st.cache_resource(show_spinner=False)
def glossary() -> Glossary:
    """The glossary of financial terms, loaded once."""
    return load_glossary()


@st.cache_data(ttl=60, show_spinner="Loading market data…")
def market_overview() -> MarketOverview:
    """Indices, sectors, and the S&P 500 read. The market service caches each quote too."""
    return build_market_overview(context().market)


def provider_status() -> list[Any]:
    """Each market data provider's status, for the sidebar (empty if not reported)."""
    status = getattr(context().market, "provider_status", None)
    return list(status()) if callable(status) else []
