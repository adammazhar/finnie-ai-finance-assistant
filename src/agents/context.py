"""Dependencies shared by every agent and tool, built once per process."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.core.config import Settings, get_settings
from src.core.models import RiskTolerance
from src.core.reference import RiskProfile, SecurityCatalog, get_catalog, get_risk_profiles
from src.core.tax import TaxReference, get_tax_reference
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    llm: Any  # main chat model (BaseChatModel or RunnableWithFallbacks)
    market: Any  # MarketDataService or a test double with the same methods
    settings: Settings
    retriever: Retriever | None = None
    fast_llm: Any | None = None  # used by guardrail rewrites
    catalog: SecurityCatalog = field(default_factory=get_catalog)
    risk_profiles: dict[RiskTolerance, RiskProfile] = field(default_factory=get_risk_profiles)
    tax: TaxReference = field(default_factory=get_tax_reference)


def build_agent_context(settings: Settings | None = None) -> AgentContext:
    """Production wiring. A missing knowledge base index degrades to no retrieval."""
    from src.core.llm import get_llm
    from src.data.service import get_market_data_service
    from src.rag.retriever import get_retriever

    settings = settings or get_settings()
    # Models first: a missing API key should fail in milliseconds, not after the
    # knowledge base's slow load.
    llm = get_llm("main", settings=settings)
    fast_llm = get_llm("fast", settings=settings)
    try:
        retriever: Retriever | None = get_retriever()
    except Exception:
        logger.exception("Knowledge base unavailable; agents will answer without it")
        retriever = None
    return AgentContext(
        llm=llm,
        fast_llm=fast_llm,
        market=get_market_data_service(),
        retriever=retriever,
        settings=settings,
    )
