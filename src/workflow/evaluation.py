"""Routing accuracy on a labelled question set (``tests/evals/routing_cases.yaml``).

A case is correct when every expected specialist is chosen (or one of the listed
alternative sets is) and out-of-scope questions are declined. Choosing an extra
specialist isn't counted as wrong, because it costs latency rather than correctness,
but extras are reported, alongside the stricter exact-match rate.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from src.core.models import AgentName
from src.workflow.router import RouteDecision, keyword_route, route_with_llm

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "tests" / "evals" / "routing_cases.yaml"
LLM_TARGET = 0.90  # design target (docs/DESIGN.md section 11)
KEYWORD_FLOOR = 0.75  # regression floor for the fallback router (docs/BENCHMARKS.md)


class RoutingCase(BaseModel):
    question: str
    agents: list[AgentName] = Field(default_factory=list)
    alternatives: list[list[AgentName]] = Field(default_factory=list)
    out_of_scope: bool = False
    has_portfolio: bool = False
    history: list[str] = Field(default_factory=list, description="user, Finnie, user, ...")

    def messages(self) -> list[BaseMessage]:
        return [
            HumanMessage(content=text) if i % 2 == 0 else AIMessage(content=text)
            for i, text in enumerate(self.history)
        ]

    def accepts(self, decision: RouteDecision) -> bool:
        if self.out_of_scope or decision.out_of_scope:
            return self.out_of_scope == decision.out_of_scope
        chosen = set(decision.agents)
        return any(set(wanted) <= chosen for wanted in [self.agents, *self.alternatives])

    def exact(self, decision: RouteDecision) -> bool:
        if self.out_of_scope or decision.out_of_scope:
            return self.accepts(decision)
        chosen = set(decision.agents)
        return any(set(wanted) == chosen for wanted in [self.agents, *self.alternatives])


def load_cases(path: Path = DEFAULT_CASES) -> list[RoutingCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [RoutingCase.model_validate(case) for case in raw["cases"]]


class RoutingReport(BaseModel):
    router: str
    cases: int
    accuracy: float
    exact: float
    extra_agents: int = Field(description="agents chosen beyond what a case needed")
    keyword_fallbacks: int = Field(description="LLM calls that failed and used keywords")
    latency_ms_p50: float
    latency_ms_p95: float
    misses: list[str] = Field(default_factory=list)


Router = Callable[[RoutingCase], RouteDecision]


def llm_router(llm: Any, *, max_agents: int, min_confidence: float) -> Router:
    def route(case: RoutingCase) -> RouteDecision:
        return route_with_llm(
            llm,
            case.question,
            history=case.messages(),
            summary=None,
            has_portfolio=case.has_portfolio,
            max_agents=max_agents,
            min_confidence=min_confidence,
        )

    return route


def keyword_router(max_agents: int) -> Router:
    """The fallback router, scored on its own as a baseline (it can't use history)."""
    return lambda case: keyword_route(case.question, max_agents)


def evaluate(router: Router, cases: list[RoutingCase], name: str) -> RoutingReport:
    correct = exact = extra = fallbacks = 0
    latencies: list[float] = []
    misses: list[str] = []
    for case in cases:
        started = time.perf_counter()
        decision = router(case)
        latencies.append((time.perf_counter() - started) * 1000)
        fallbacks += name != "keyword" and decision.source == "keyword"
        if case.accepts(decision):
            correct += 1
            exact += case.exact(decision)
            needed = set().union(*map(set, [case.agents, *case.alternatives]))
            extra += len(set(decision.agents) - needed)
        else:
            got = "out_of_scope" if decision.out_of_scope else ",".join(decision.agents)
            wanted = "out_of_scope" if case.out_of_scope else ",".join(case.agents)
            misses.append(f"{case.question!r}: wanted {wanted}, got {got}")
    ordered = sorted(latencies) or [0.0]
    return RoutingReport(
        router=name,
        cases=len(cases),
        accuracy=correct / len(cases) if cases else 0.0,
        exact=exact / len(cases) if cases else 0.0,
        extra_agents=extra,
        keyword_fallbacks=fallbacks,
        latency_ms_p50=round(statistics.median(ordered), 1),
        latency_ms_p95=round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1),
        misses=misses,
    )
