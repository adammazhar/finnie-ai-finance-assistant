"""Graph nodes. Each is a plain function of state, built with the shared dependencies."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.types import Send
from pydantic import BaseModel, Field

from src.agents.base import AgentRequest, BaseAgent, message_text
from src.agents.context import AgentContext
from src.core.guardrails import (
    ADVICE_REFRAME,
    INJECTION_NOTE,
    InputScreen,
    blocked_response,
    enforce_output,
    finalize,
    screen_input,
    strip_disclaimer,
)
from src.core.models import AgentResult, Freshness, Holding, Source, UserProfile
from src.core.portfolio import holding_values, portfolio_total
from src.workflow import progress
from src.workflow.planner import build_plan, plan_handoff
from src.workflow.router import HOLDINGS, RouteDecision, route_with_llm
from src.workflow.savings import (
    DEFAULT_GOAL,
    GoalSavings,
    goal_key,
    grounded_savings,
    is_short_reply,
    parse_savings_reply,
    savings_fact,
    savings_question,
    savings_reprompt,
    stated_savings,
)
from src.workflow.state import RESET, AgentTask, FinnieState
from src.workflow.synthesis import Synthesis, synthesize, tidy_citations

logger = logging.getLogger(__name__)

OUT_OF_SCOPE_REPLY = (
    "I'm Finnie, and I focus on personal finance, investing, markets, and taxes on "
    "investments, so I can't help with that one. Try asking me about saving, investing, or "
    "how markets work!"
)
FALLBACK_REPLY = (
    "Sorry, I couldn't put an answer together just now because my specialists ran into a "
    "problem. Please try again in a moment, or rephrase the question."
)
SUMMARY_PROMPT = (
    "Summarize this earlier part of a conversation between a user and Finnie, a financial "
    "education assistant, in under 120 words. Keep facts the user shared about themselves "
    "(goals, holdings, ages, amounts) and the topics covered. Don't add advice."
)


class TurnOutput(BaseModel):
    """What the UI, CLI, and tests get back for one turn."""

    answer: str
    sources: list[Source] = Field(default_factory=list)
    agents: list[str] = Field(default_factory=list)
    status: str = Field(
        description="answered, blocked, out_of_scope, fallback, or needs_input (Finnie asked "
        "the user something before answering)"
    )
    guardrail: str = Field(default="unchanged", description="unchanged/rewritten/neutralized")
    screen: str = "ok"
    route: dict[str, Any] | None = None
    data: dict[str, dict[str, Any]] = Field(default_factory=dict, description="per-agent data")
    freshness: list[Freshness] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)
    merged_by_llm: bool = False


@dataclass
class Deps:
    context: AgentContext
    agents: Mapping[Any, BaseAgent]

    @property
    def workflow(self) -> Any:
        return self.context.settings.workflow


def _latest_question(state: FinnieState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message_text(message).strip()
    return ""


def _history(state: FinnieState, limit: int) -> list[Any]:
    """Messages before the current question, most recent ``limit``."""
    return state.get("messages", [])[:-1][-limit:]


def _guidance(screen: InputScreen, route: RouteDecision | None) -> list[str]:
    notes = []
    if screen.category == "advice_seeking":
        notes.append(ADVICE_REFRAME)
    if screen.category == "injection_suspected":
        notes.append(INJECTION_NOTE)
    if route and route.tickers:
        notes.append(f"Ticker symbols in the question: {', '.join(route.tickers)}.")
    return notes


def _run_with_deadline(agent: BaseAgent, request: AgentRequest, deadline: float) -> AgentResult:
    """Run an agent, giving up when the turn's time budget runs out.

    A late agent becomes an error result, so the turn still answers with what the other
    specialists found. Its thread is left to finish on its own (each LLM and HTTP call
    has its own timeout) and its result is discarded.
    """
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"agent-{agent.name}")
    try:
        return pool.submit(agent.run, request).result(timeout=max(0.0, deadline - time.time()))
    except FutureTimeout:
        logger.warning("Agent %s ran out of time", agent.name, extra={"agent": agent.name})
        return AgentResult(agent=agent.name, error="TimeoutError: the turn's time budget ran out")
    finally:
        pool.shutdown(wait=False)


def _team_note(name: str, plan: list[list[str]]) -> str | None:
    """Tell an agent which other specialists cover the rest of this turn's question."""
    others = [a for stage in plan for a in stage if a != name]
    if not others:
        return None
    return (
        f"Other Finnie specialists ({', '.join(dict.fromkeys(others))}) are answering the other "
        "parts of this question in this turn. Cover only your own specialty and don't hand off "
        "to them."
    )


ADVICE_CONCEPTS = (
    "How investors evaluate adding to or selling one investment: concentration risk, "
    "diversification, and volatility."
)


def _user_context(state: FinnieState, route: RouteDecision | None, deps: Deps) -> list[str]:
    """For "should I buy X?": the user's own position, so the answer is about them."""
    if InputScreen.model_validate(state["screen"]).category != "advice_seeking":
        return []
    facts = _holding_facts(state, route, deps)
    return [facts] if facts else []


def _retrieval_query(state: FinnieState) -> str | None:
    """A "should I buy X?" question names no concepts, so knowledge base search finds
    nothing to cite. Search for the concepts investors weigh instead."""
    if InputScreen.model_validate(state["screen"]).category != "advice_seeking":
        return None
    return f"{state.get('query') or state['question']} {ADVICE_CONCEPTS}"


def _holding_facts(state: FinnieState, route: RouteDecision | None, deps: Deps) -> str | None:
    """For questions about specific tickers: how much of the saved portfolio each one is."""
    holdings = [Holding.model_validate(h) for h in state.get("portfolio") or []]
    if not holdings or route is None or not route.tickers:
        return None
    values = holding_values(holdings, deps.context.market, deps.context.catalog)
    total = sum(values.values())
    if not total:
        return None
    facts = []
    for ticker in route.tickers:
        if ticker in values:
            facts.append(
                f"{ticker} is already {values[ticker] / total:.0%} of my saved portfolio "
                f"(${values[ticker]:,.0f} of ${total:,.0f} at current prices)."
            )
        else:
            facts.append(f"My saved portfolio doesn't hold {ticker}.")
    return " ".join(facts)


def _portfolio_value(state: FinnieState, deps: Deps) -> float | None:
    """The portfolio's value: the Portfolio specialist's total from this turn, else quotes."""
    results = state.get("results") or {}
    analysis = (results.get("portfolio") or {}).get("data", {}).get("portfolio_analysis") or {}
    if analysis.get("total_value"):
        return float(analysis["total_value"])
    holdings = [Holding.model_validate(h) for h in state.get("portfolio") or []]
    if not holdings:
        return None
    return portfolio_total(holdings, deps.context.market, deps.context.catalog)


def _savings_guidance(state: FinnieState, route: RouteDecision | None, deps: Deps) -> str | None:
    """For the goal agent: the savings to use, from the user's per-goal choice."""
    stored = (state.get("goal_savings") or {}).get(goal_key(route.goal if route else None))
    if stored is None:
        return None
    savings = GoalSavings.model_validate(stored)
    value = _portfolio_value(state, deps) if savings.choice == "all" else None
    return savings_fact(savings, value)


# ---- nodes ----------------------------------------------------------------------------------


def make_nodes(deps: Deps) -> dict[str, Any]:
    def ingest(state: FinnieState) -> dict[str, Any]:
        progress.status("Reading your question…")
        question = _latest_question(state)
        screen = screen_input(question)
        logger.info("Turn started", extra={"screen": screen.category})
        update: dict[str, Any] = {
            "question": question,
            "query": question,
            "screen": screen.model_dump(),
            "route": None,
            "plan": [],
            "stage": 0,
            "handoff_used": False,
            "deadline": time.time() + deps.workflow.turn_timeout_s,
            "savings_prompt": None,
            "results": {RESET: True},
            "output": None,
            "profile": state.get("profile") or UserProfile().model_dump(),
        }
        pending = state.get("pending_savings")
        if pending and not screen.blocked:
            update |= _answer_pending(state, pending, question)
        return update

    def _answer_pending(state: FinnieState, pending: dict[str, Any], reply: str) -> dict[str, Any]:
        """Read the reply to the savings question, then resume the original goal question."""
        value = pending["portfolio_value"]
        savings = parse_savings_reply(reply, value)
        if savings is None:
            if is_short_reply(reply):
                return {"savings_prompt": savings_reprompt(value)}  # ask again
            return {"pending_savings": None}  # a new question: move on
        logger.info("Savings answered", extra={"choice": savings.choice})
        return {
            "pending_savings": None,
            "goal_savings": {
                **(state.get("goal_savings") or {}),
                pending["goal"]: savings.model_dump(),
            },
            "question": pending["question"],
            "query": pending["query"],
            "screen": pending["screen"],
            "route": pending["route"],
            "plan": pending["plan"],
        }

    def router(state: FinnieState) -> dict[str, Any]:
        progress.status("Choosing the right specialists…")
        workflow = deps.workflow
        decision = route_with_llm(
            deps.context.fast_llm or deps.context.llm,
            state["question"],
            history=_history(state, 6),
            summary=state.get("summary"),
            has_portfolio=bool(state.get("portfolio")),
            max_agents=workflow.max_agents_per_turn,
            min_confidence=workflow.router_min_confidence,
            goals=[g for g in state.get("goal_savings") or {} if g != DEFAULT_GOAL],
        )
        plan = build_plan(decision.agents, decision.depends_on, max_stages=workflow.max_stages)
        logger.info("Routed", extra={"agents": decision.agents, "route_source": decision.source})
        return {"route": decision.model_dump(), "query": decision.standalone_query, "plan": plan}

    def check_savings(state: FinnieState) -> dict[str, Any]:
        """Before a goal question runs: settle how much of a saved portfolio counts."""
        route = RouteDecision.model_validate(state["route"])
        if route.out_of_scope or "goal_planning" not in {a for s in state["plan"] for a in s}:
            return {}
        key = goal_key(route.goal)
        saved = state.get("goal_savings") or {}
        if key in saved:
            return {}  # already settled for this goal
        stated = grounded_savings(route.current_savings, state["question"])
        if stated is None:
            stated = stated_savings(state["question"])
        if stated is not None:
            choice = GoalSavings(choice="amount", amount=stated, source="stated")
        elif HOLDINGS.search(state["question"]):  # holdings listed in the question count
            choice = GoalSavings(choice="all", source="stated")
        else:
            value = _portfolio_value(state, deps) if state.get("portfolio") else None
            if not value:
                return {}  # nothing to ask about: the goal agent asks or assumes
            pending = {
                "goal": key,
                "question": state["question"],
                "query": state["query"],
                "screen": state["screen"],
                "route": state["route"],
                "plan": state["plan"],
                "portfolio_value": value,
            }
            return {"pending_savings": pending, "savings_prompt": savings_question(value)}
        return {"goal_savings": {**saved, key: choice.model_dump()}}

    def ask_savings(state: FinnieState) -> dict[str, Any]:
        output = TurnOutput(
            answer=state["savings_prompt"] or "",
            status="needs_input",
            screen=state["screen"]["category"],
            route=state.get("route"),
        )
        return {
            "output": output.model_dump(mode="json"),
            "messages": [AIMessage(content=output.answer)],
        }

    def make_agent_node(name: str) -> Any:
        agent = deps.agents[name]

        def run_agent(task: AgentTask) -> dict[str, Any]:
            state = task["state"]
            route = RouteDecision.model_validate(state["route"]) if state.get("route") else None
            request = AgentRequest(
                query=state.get("query") or state["question"],
                profile=UserProfile.model_validate(state.get("profile") or {}),
                portfolio=[Holding.model_validate(h) for h in state.get("portfolio") or []] or None,
                history=_history(state, deps.workflow.history_window),
                prior_results={
                    k: AgentResult.model_validate(v)
                    for k, v in (state.get("results") or {}).items()
                },
                tickers=route.tickers if route else [],
                guidance=[
                    *_guidance(InputScreen.model_validate(state["screen"]), route),
                    *filter(
                        None,
                        [
                            _team_note(name, state.get("plan") or []),
                            _savings_guidance(state, route, deps)
                            if name == "goal_planning"
                            else None,
                        ],
                    ),
                ],
                allow_handoff=task["allow_handoff"],
                retrieval_query=_retrieval_query(state),
                user_context=_user_context(state, route, deps),
            )
            progress.agent_started(name)
            result = _run_with_deadline(agent, request, state["deadline"])
            progress.agent_finished(name, result.ok)
            return {"results": {name: result.model_dump(mode="json")}}

        return run_agent

    def collect(state: FinnieState) -> dict[str, Any]:
        """Finish a stage: advance, persist new holdings, and schedule at most one hand-off."""
        update: dict[str, Any] = {"stage": state.get("stage", 0) + 1}
        results = state.get("results") or {}
        holdings = (results.get("portfolio") or {}).get("data", {}).get("holdings")
        if holdings:
            update["portfolio"] = holdings
        if not state.get("handoff_used"):
            requested = [h for r in results.values() for h in r.get("handoff", [])]
            if requested:
                plan = plan_handoff(
                    state["plan"],
                    requested[:1],
                    list(results),
                    max_stages=deps.workflow.max_stages,
                )
                update["handoff_used"] = True  # the one hand-off is spent, used or not
                if plan is not None:
                    update["plan"] = plan
                    logger.info("Hand-off scheduled", extra={"agents": plan[-1]})
        return update

    def synthesize_node(state: FinnieState) -> dict[str, Any]:
        order = list(dict.fromkeys(a for stage in state.get("plan", []) for a in stage))
        results = [
            AgentResult.model_validate(state["results"][a])
            for a in order
            if a in (state.get("results") or {})
        ]
        llm = deps.context.llm if sum(r.ok for r in results) > 1 else None
        if llm is not None:
            progress.status("Combining the specialists' answers…")
        synthesis = synthesize(results, llm)
        errors: dict[str, str] = {r.agent: r.error for r in results if r.error}
        data: dict[str, Any] = {r.agent: r.data for r in results}
        screen = state["screen"]["category"]
        if synthesis is None:
            output = TurnOutput(
                answer=FALLBACK_REPLY,
                status="fallback",
                errors=errors,
                screen=screen,
                route=state.get("route"),
                data=data,
            )
        else:
            output = _draft_output(synthesis, errors, data, screen, state.get("route"))
        return {"output": output.model_dump(mode="json")}

    def guard(state: FinnieState) -> dict[str, Any]:
        output = TurnOutput.model_validate(state["output"])
        remembered = output.answer
        if output.status == "answered":
            progress.status("Checking the answer…")
            guarded = enforce_output(strip_disclaimer(output.answer), deps.context.fast_llm)
            output.guardrail = guarded.action
            text, output.sources = tidy_citations(guarded.text, output.sources)
            output.answer = finalize(text, output.freshness)
            remembered = text  # history keeps the answer without the disclaimer
        return {
            "output": output.model_dump(mode="json"),
            "messages": [AIMessage(content=remembered)],
        }

    def respond_blocked(state: FinnieState) -> dict[str, Any]:
        screen = InputScreen.model_validate(state["screen"])
        output = TurnOutput(
            answer=blocked_response(screen), status="blocked", screen=screen.category
        )
        return {
            "output": output.model_dump(mode="json"),
            "messages": [AIMessage(content=output.answer)],
        }

    def respond_out_of_scope(state: FinnieState) -> dict[str, Any]:
        output = TurnOutput(
            answer=OUT_OF_SCOPE_REPLY,
            status="out_of_scope",
            screen=state["screen"]["category"],
            route=state.get("route"),
        )
        return {
            "output": output.model_dump(mode="json"),
            "messages": [AIMessage(content=output.answer)],
        }

    def summarize(state: FinnieState) -> dict[str, Any]:
        """Fold older turns into a running summary once the history gets long."""
        messages = state.get("messages", [])
        keep = deps.workflow.history_window
        if len(messages) <= deps.workflow.summarize_after:
            return {}
        old = messages[:-keep]
        transcript = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Finnie'}: {message_text(m)[:500]}"
            for m in old
        )
        if state.get("summary"):
            transcript = f"Earlier summary: {state['summary']}\n\n{transcript}"
        try:
            reply = (deps.context.fast_llm or deps.context.llm).invoke(
                [SystemMessage(content=SUMMARY_PROMPT), HumanMessage(content=transcript)]
            )
            summary = message_text(reply).strip()
        except Exception:
            logger.exception("Summarizing history failed; keeping full history for now")
            return {}
        return {"summary": summary, "messages": [RemoveMessage(id=m.id) for m in old if m.id]}

    nodes: dict[str, Any] = {
        "ingest": ingest,
        "router": router,
        "check_savings": check_savings,
        "ask_savings": ask_savings,
        "collect": collect,
        "synthesize": synthesize_node,
        "guard": guard,
        "respond_blocked": respond_blocked,
        "respond_out_of_scope": respond_out_of_scope,
        "summarize": summarize,
    }
    nodes.update({name: make_agent_node(name) for name in deps.agents})
    return nodes


def _draft_output(
    synthesis: Synthesis,
    errors: dict[str, str],
    data: dict[str, Any],
    screen: str,
    route: dict[str, Any] | None,
) -> TurnOutput:
    return TurnOutput(
        answer=synthesis.text,
        sources=synthesis.sources,
        agents=synthesis.agents,
        status="answered",
        screen=screen,
        route=route,
        data=data,
        freshness=synthesis.freshness,
        errors=errors,
        merged_by_llm=synthesis.merged_by_llm,
    )


# ---- edges ----------------------------------------------------------------------------------


def after_ingest(state: FinnieState) -> str:
    if InputScreen.model_validate(state["screen"]).blocked:
        return "respond_blocked"
    if state.get("savings_prompt"):
        return "ask_savings"
    if state.get("route"):
        return "check_savings"  # resuming a goal question after the savings answer
    return "router"


def fan_out(state: FinnieState) -> list[Send] | str:
    """Send the current stage's agents in parallel, or move on when the plan is done."""
    route = state.get("route") or {}
    if route.get("out_of_scope"):
        return "respond_out_of_scope"
    if state.get("savings_prompt"):
        return "ask_savings"
    plan, stage = state.get("plan", []), state.get("stage", 0)
    if stage >= len(plan):
        return "synthesize"
    allow_handoff = not state.get("handoff_used", False)
    return [
        Send(agent, AgentTask(agent=agent, allow_handoff=allow_handoff, state=state))
        for agent in plan[stage]
    ]
