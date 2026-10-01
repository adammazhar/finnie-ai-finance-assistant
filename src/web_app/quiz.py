"""A five-question risk tolerance quiz (pure, no Streamlit).

Each answer scores 1 (cautious) to 3 (comfortable with risk). The total (5 to 15) maps to
the three risk profiles Finnie uses for portfolio comparisons and goal projections. It is
an educational starting point, not a suitability assessment.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.core.models import RiskTolerance

QUESTIONS: list[tuple[str, list[tuple[str, int]]]] = [
    (
        "If your investments dropped 20% in a month, what would you most likely do?",
        [
            ("Sell to avoid further losses", 1),
            ("Wait and see", 2),
            ("Keep investing as planned", 3),
        ],
    ),
    (
        "When will you need most of this money?",
        [("Within 3 years", 1), ("In 3 to 10 years", 2), ("More than 10 years from now", 3)],
    ),
    (
        "Which matters more to you?",
        [
            ("Avoiding losses, even if growth is slow", 1),
            ("A balance of growth and stability", 2),
            ("Maximum long-term growth, even with big swings", 3),
        ],
    ),
    (
        "How much investing experience do you have?",
        [("None or very little", 1), ("Some", 2), ("A lot", 3)],
    ),
    (
        "If you lost your job, how long could your emergency savings cover expenses?",
        [("Less than 3 months", 1), ("3 to 6 months", 2), ("More than 6 months", 3)],
    ),
]


def tolerance_from_scores(scores: Sequence[int]) -> RiskTolerance:
    if len(scores) != len(QUESTIONS):
        raise ValueError(f"Expected {len(QUESTIONS)} answers, got {len(scores)}")
    total = sum(scores)
    if total <= 8:
        return "conservative"
    if total <= 11:
        return "moderate"
    return "aggressive"
