"""Evaluate the router on the labelled set: python scripts/eval_routing.py.

Uses the configured fast model (one API call per case) and scores the keyword fallback
router on the same cases. Exits non-zero when the LLM router is below its target or the
keyword router falls below its regression floor.
"""

import sys

from src.core.config import get_settings
from src.core.llm import get_llm
from src.workflow.evaluation import (
    KEYWORD_FLOOR,
    LLM_TARGET,
    evaluate,
    keyword_router,
    llm_router,
    load_cases,
)

if __name__ == "__main__":
    settings = get_settings()
    workflow = settings.workflow
    cases = load_cases()
    llm = get_llm("fast", settings=settings)
    reports = [
        evaluate(
            llm_router(
                llm,
                max_agents=workflow.max_agents_per_turn,
                min_confidence=workflow.router_min_confidence,
            ),
            cases,
            "llm",
        ),
        evaluate(keyword_router(workflow.max_agents_per_turn), cases, "keyword"),
    ]
    print(f"{len(cases)} cases")
    print("router    accuracy  exact   extra agents  fallbacks  p50 ms   p95 ms")
    for r in reports:
        print(
            f"{r.router:<8}  {r.accuracy:>7.1%}  {r.exact:>6.1%}  {r.extra_agents:>12}  "
            f"{r.keyword_fallbacks:>9}  {r.latency_ms_p50:>7.1f}  {r.latency_ms_p95:>7.1f}"
        )
    for r in reports:
        for miss in r.misses:
            print(f"{r.router} miss: {miss}")

    failed = False
    for report, target, label in (
        (reports[0], LLM_TARGET, "target"),
        (reports[1], KEYWORD_FLOOR, "floor"),
    ):
        ok = report.accuracy >= target
        failed |= not ok
        print(f"{report.router} router {'meets' if ok else 'is BELOW'} its {target:.0%} {label}.")
    sys.exit(1 if failed else 0)
