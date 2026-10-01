"""All six specialist agents, keyed by name."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.context import AgentContext
from src.agents.finance_qa import FinanceQAAgent
from src.agents.goal_planning import GoalPlanningAgent
from src.agents.market import MarketAgent
from src.agents.news import NewsAgent
from src.agents.portfolio import PortfolioAgent
from src.agents.tax import TaxAgent
from src.core.models import AgentName

AGENT_CLASSES: dict[AgentName, type[BaseAgent]] = {
    cls.name: cls
    for cls in (FinanceQAAgent, PortfolioAgent, MarketAgent, GoalPlanningAgent, NewsAgent, TaxAgent)
}


def build_agents(context: AgentContext) -> dict[AgentName, BaseAgent]:
    return {name: cls(context) for name, cls in AGENT_CLASSES.items()}
