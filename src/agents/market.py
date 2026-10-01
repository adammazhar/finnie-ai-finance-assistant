"""Market data and plain-English market context."""

from src.agents.base import BaseAgent


class MarketAgent(BaseAgent):
    name = "market"
    description = "Explains prices, trends, sectors, and the overall market with live data."
    rag_categories = ("market_economics", "stocks")
    tool_names = (
        "get_quotes",
        "get_market_overview",
        "get_technical_snapshot",
        "get_company_overview",
        "search_knowledge_base",
        "request_handoff",
    )
