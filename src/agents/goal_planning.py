"""Savings goals with Monte Carlo projections."""

from src.agents.base import BaseAgent


class GoalPlanningAgent(BaseAgent):
    name = "goal_planning"
    description = "Projects savings goals and explains the probability of reaching them."
    rag_categories = ("retirement_planning", "financial_planning_goals", "investing_basics")
    tool_names = ("project_goal", "search_knowledge_base", "request_handoff")
