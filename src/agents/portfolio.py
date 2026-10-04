"""Portfolio analysis: allocation, diversification, fees, and risk."""

from src.agents.base import BaseAgent


class PortfolioAgent(BaseAgent):
    """Analyzes the user's holdings (from the message or saved) and explains the results."""

    name = "portfolio"
    description = "Analyzes a user's holdings and explains what the numbers mean."
    rag_categories = ("portfolio_management", "funds_etfs", "risk_behavioral")
    tool_names = (
        "analyze_portfolio",
        "get_quotes",
        "get_company_overview",
        "search_knowledge_base",
        "request_handoff",
    )
