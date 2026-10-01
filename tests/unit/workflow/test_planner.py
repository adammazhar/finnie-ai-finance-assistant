from src.workflow.planner import build_plan, plan_handoff
from src.workflow.state import RESET, merge_results


def test_independent_agents_share_one_stage():
    assert build_plan(["market", "news"], max_stages=3) == [["market", "news"]]


def test_empty_and_duplicate_agents():
    assert build_plan([], max_stages=3) == []
    assert build_plan(["market", "market"], max_stages=3) == [["market"]]


def test_default_dependencies_put_portfolio_first():
    plan = build_plan(["goal_planning", "market", "portfolio"], max_stages=3)
    assert plan == [["market", "portfolio"], ["goal_planning"]]
    assert build_plan(["tax", "portfolio"], max_stages=3) == [["portfolio"], ["tax"]]


def test_default_dependency_ignored_when_agent_not_chosen():
    assert build_plan(["goal_planning"], max_stages=3) == [["goal_planning"]]


def test_router_dependencies_add_to_defaults():
    plan = build_plan(["news", "market"], {"news": ["market"]}, max_stages=3)
    assert plan == [["market"], ["news"]]


def test_self_and_unknown_dependencies_are_ignored():
    plan = build_plan(["news"], {"news": ["news", "tax"]}, max_stages=3)
    assert plan == [["news"]]


def test_cycle_runs_the_rest_together():
    plan = build_plan(["news", "market"], {"news": ["market"], "market": ["news"]}, max_stages=3)
    assert plan == [["news", "market"]]


def test_overflow_stages_merge_into_the_last_allowed():
    deps = {"market": ["portfolio"], "news": ["market"]}
    assert build_plan(["portfolio", "market", "news"], deps, max_stages=3) == [
        ["portfolio"],
        ["market"],
        ["news"],
    ]
    assert build_plan(["portfolio", "market", "news"], deps, max_stages=2) == [
        ["portfolio"],
        ["market", "news"],
    ]


def test_handoff_appends_one_stage():
    assert plan_handoff([["tax"]], ["portfolio"], ["tax"], max_stages=3) == [
        ["tax"],
        ["portfolio"],
    ]


def test_handoff_skips_agents_already_run_or_scheduled():
    """Regression: a hand-off to an agent planned for a later stage ran it twice."""
    plan = [["portfolio", "market"], ["goal_planning"]]
    assert plan_handoff(plan, ["goal_planning"], ["portfolio", "market"], max_stages=3) is None
    assert plan_handoff(plan, ["market"], ["portfolio", "market"], max_stages=3) is None


def test_handoff_respects_stage_budget():
    assert plan_handoff([["a"], ["b"]], ["tax"], ["a", "b"], max_stages=2) is None


def test_handoff_takes_first_valid_request():
    assert plan_handoff([["tax"]], ["tax", "news"], ["tax"], max_stages=3) == [["tax"], ["news"]]


def test_merge_results_reducer():
    assert merge_results(None, None) == {}
    assert merge_results({"a": 1}, None) == {"a": 1}
    assert merge_results({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_results(None, {"b": 2}) == {"b": 2}
    assert merge_results({"a": 1}, {RESET: True}) == {}
