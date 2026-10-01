"""Recent news, summarized with context for learners."""

from src.agents.base import BaseAgent


class NewsAgent(BaseAgent):
    name = "news"
    description = "Summarizes recent financial news and why it matters for a learner."
    rag_categories = ("market_economics",)
    tool_names = ("get_news", "get_quotes", "request_handoff")
