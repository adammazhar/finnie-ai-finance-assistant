"""End-to-end workflow latency with live models and data: python scripts/bench_workflow.py.

Runs one question per specialist and several multi-specialist questions, each in its own
conversation with a saved 3-holding portfolio, through ``FinnieAssistant.stream`` as the
chat tab does. Questions state their savings (or list holdings) so none stops to ask how
much of the portfolio counts. Exits non-zero when a turn doesn't produce an answer, the
first progress event is slow, or the multi-specialist median misses its target.
"""

import statistics
import sys
import time

from src.core.models import Holding, UserProfile
from src.workflow.graph import FinnieAssistant
from src.workflow.nodes import TurnOutput
from src.workflow.progress import Progress

MULTI_P50_TARGET_S = 15.0  # docs/BENCHMARKS.md
FIRST_FEEDBACK_P95_TARGET_S = 1.0  # first progress event the chat shows (docs/BENCHMARKS.md)
SINGLE_P50_TARGET_S = 6.0  # docs/DESIGN.md section 14 (reported, not enforced)

PORTFOLIO = [
    Holding(ticker="VTI", shares=40),
    Holding(ticker="TSLA", shares=30),
    Holding(ticker="BND", shares=50),
]
SINGLE = [
    "What is an ETF?",
    "How diversified is my portfolio?",
    "How is AAPL doing today?",
    "I'm 30 with $20,000 saved. Could I reach $500,000 by 55 adding $800 a month?",
    "Any news on Nvidia?",
    "How are long-term capital gains taxed?",
]
MULTI = [
    "How is TSLA doing today and is there any news about it?",
    "Explain what an ETF is, and how is VOO doing today?",
    "I hold 40 VTI, 30 TSLA and 50 BND. Am I on track for $400,000 in 20 years if I add $700 "
    "a month?",
    "What are the tax implications of selling my holdings, and how concentrated am I?",
]


FIRST_FEEDBACK: list[float] = []


def run(assistant: FinnieAssistant, questions: list[str], label: str) -> list[float]:
    times = []
    for i, question in enumerate(questions):
        started = time.perf_counter()
        first: float | None = None
        out: TurnOutput | None = None
        for event in assistant.stream(
            question,
            thread_id=f"bench-{label}-{i}",
            portfolio=PORTFOLIO,
            profile=UserProfile(age=35),
        ):
            if isinstance(event, Progress) and first is None:
                first = time.perf_counter() - started
            elif isinstance(event, TurnOutput):
                out = event
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        FIRST_FEEDBACK.append(first if first is not None else elapsed)
        assert out is not None
        print(
            f"{elapsed:6.1f}s  first {FIRST_FEEDBACK[-1] * 1000:5.0f} ms  {out.status:<11} "
            f"{','.join(out.agents):<28} {question[:50]}"
        )
        if out.status != "answered":
            print(f"FAILED: expected an answer, got {out.status}")
            sys.exit(1)
    return times


if __name__ == "__main__":
    assistant = FinnieAssistant()
    single = run(assistant, SINGLE, "single")
    multi = run(assistant, MULTI, "multi")
    single_p50, multi_p50 = statistics.median(single), statistics.median(multi)
    print(
        f"single specialist: n={len(single)} p50 {single_p50:.1f}s max {max(single):.1f}s "
        f"(design target < {SINGLE_P50_TARGET_S:.0f}s)"
    )
    print(
        f"multi specialist:  n={len(multi)} p50 {multi_p50:.1f}s max {max(multi):.1f}s "
        f"(target < {MULTI_P50_TARGET_S:.0f}s)"
    )
    ordered = sorted(FIRST_FEEDBACK)
    first_p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    print(
        f"first progress:    n={len(ordered)} p50 {statistics.median(ordered) * 1000:.0f} ms "
        f"p95 {first_p95 * 1000:.0f} ms (target p95 < {FIRST_FEEDBACK_P95_TARGET_S:.0f}s)"
    )
    checks = [
        ("Multi-specialist p50", multi_p50 < MULTI_P50_TARGET_S, f"{MULTI_P50_TARGET_S:.0f}s"),
        ("First progress p95", first_p95 < FIRST_FEEDBACK_P95_TARGET_S, "1s"),
    ]
    for name, ok, target in checks:
        print(f"{name} {'meets' if ok else 'MISSES'} the {target} target.")
    sys.exit(0 if all(ok for _, ok, _ in checks) else 1)
