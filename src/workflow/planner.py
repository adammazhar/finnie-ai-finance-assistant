"""Turn the router's agents and dependencies into execution stages.

Agents in the same stage run in parallel; a stage runs after every stage before it, so an
agent can use earlier agents' results. Some dependencies always apply (a goal projection
or a tax question about the user's holdings benefits from the portfolio analysis first),
on top of any the router reports. Cycles are broken by dropping dependency edges.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

DEFAULT_DEPENDENCIES: dict[str, set[str]] = {
    "goal_planning": {"portfolio"},
    "tax": {"portfolio"},
}


def build_plan(
    agents: Sequence[str],
    depends_on: Mapping[str, Sequence[str]] | None = None,
    *,
    max_stages: int,
) -> list[list[str]]:
    """Group the agents into stages that respect their dependencies, at most ``max_stages``.

    Duplicate agents are dropped, and stages beyond the budget are merged into the last one.
    """
    chosen = list(dict.fromkeys(agents))
    if not chosen:
        return []
    deps: dict[str, set[str]] = {a: set() for a in chosen}
    for agent in chosen:
        wanted = set(DEFAULT_DEPENDENCIES.get(agent, ())) | set((depends_on or {}).get(agent, ()))
        deps[agent] = {d for d in wanted if d in deps and d != agent}

    stages: list[list[str]] = []
    placed: set[str] = set()
    while len(placed) < len(chosen):
        ready = [a for a in chosen if a not in placed and deps[a] <= placed]
        if not ready:  # a cycle: run whatever is left together rather than loop forever
            ready = [a for a in chosen if a not in placed]
        stages.append(ready)
        placed.update(ready)

    # Respect the stage budget by merging the overflow into the last allowed stage.
    if len(stages) > max_stages:
        head, tail = stages[: max_stages - 1], stages[max_stages - 1 :]
        stages = [*head, [a for stage in tail for a in stage]]
    return stages


def plan_handoff(
    plan: list[list[str]], requested: Sequence[str], already_run: Sequence[str], *, max_stages: int
) -> list[list[str]] | None:
    """Append one hand-off stage, or ``None`` if there's nothing valid to add.

    At most one hand-off per question: the caller passes only the first request, and the
    agent that runs it can't hand off again. Agents already run or already scheduled are
    never added twice.
    """
    planned = {a for stage in plan for a in stage} | set(already_run)
    for agent in requested:
        if agent not in planned and len(plan) < max_stages:
            return [*plan, [agent]]
    return None
