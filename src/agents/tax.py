"""Tax education: account types, capital gains, and official figures."""

from src.agents.base import BaseAgent


class TaxAgent(BaseAgent):
    """Explains account types and capital gains using the official tax-year figures."""

    name = "tax"
    description = "Explains how investments and accounts are taxed, using official IRS figures."
    rag_categories = ("taxes", "retirement_planning")
    tool_names = (
        "get_tax_figures",
        "compare_tax_accounts",
        "get_withdrawal_rules",
        "illustrate_capital_gains",
        "search_knowledge_base",
        "request_handoff",
    )
