"""Tax education: account types, capital gains, and official figures."""

from src.agents.base import BaseAgent


class TaxAgent(BaseAgent):
    name = "tax"
    description = "Explains how investments and accounts are taxed, using official IRS figures."
    rag_categories = ("taxes", "retirement_planning")
    tool_names = (
        "get_tax_figures",
        "compare_tax_accounts",
        "illustrate_capital_gains",
        "search_knowledge_base",
        "request_handoff",
    )
