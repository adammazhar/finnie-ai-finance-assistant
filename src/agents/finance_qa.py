"""General financial education questions; the default route."""

from src.agents.base import BaseAgent


class FinanceQAAgent(BaseAgent):
    name = "finance_qa"
    description = "Explains financial concepts and terms in plain language."
    rag_categories = None  # search the whole knowledge base
    tool_names = ("search_knowledge_base", "lookup_glossary_term", "request_handoff")
