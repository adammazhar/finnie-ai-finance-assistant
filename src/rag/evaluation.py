"""Retrieval quality evaluation: hit@k, MRR, and off-topic rejection across thresholds.

A case is a hit when any expected article appears among the top-k retrieved chunks
(glossary chunks neither help nor hurt). Off-topic questions should retrieve nothing.
Run ``python scripts/eval_retrieval.py`` against the real index; results are recorded in
``docs/BENCHMARKS.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.core.config import PROJECT_ROOT
from src.rag.retriever import Retriever

CASES_FILE = PROJECT_ROOT / "tests" / "evals" / "retrieval_cases.yaml"


class EvalCase(BaseModel):
    """One retrieval test question and the article ids that count as a hit."""

    question: str
    expected: list[str] = Field(min_length=1)

    @property
    def category(self) -> str:
        """Category of the first expected article (its id minus the number), for filtered runs."""
        return self.expected[0].rsplit("-", 1)[0]


class EvalSet(BaseModel):
    """The evaluation cases, plus off-topic questions that should retrieve nothing."""

    cases: list[EvalCase]
    off_topic: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    """Scores for one threshold and filter setting.

    ``misses`` lists each missed question with the articles that were retrieved instead.
    """

    threshold: float
    filtered: bool
    k: int
    hit_rate: float
    mrr: float
    off_topic_rejected: float
    misses: list[str]


def load_eval_set(path: Path = CASES_FILE) -> EvalSet:
    """The retrieval evaluation set (``tests/evals/retrieval_cases.yaml``)."""
    return EvalSet(**yaml.safe_load(path.read_text(encoding="utf-8")))


def evaluate(
    retriever: Retriever,
    eval_set: EvalSet,
    *,
    threshold: float | None = None,
    filtered: bool = False,
    k: int | None = None,
) -> EvalResult:
    """Score one configuration. ``filtered`` passes each case's category, as agents do."""
    original = retriever.config
    if threshold is not None:
        retriever.config = original.model_copy(update={"score_threshold": threshold})
    k = k or retriever.config.top_k
    try:
        hits, reciprocal, misses = 0, 0.0, []
        for case in eval_set.cases:
            result = retriever.retrieve(
                case.question, categories=[case.category] if filtered else None, k=k
            )
            articles = [rc.chunk.article_id for rc in result.chunks if rc.chunk.kind == "article"]
            rank = next((i for i, a in enumerate(articles, 1) if a in case.expected), None)
            if rank:
                hits += 1
                reciprocal += 1 / rank
            else:
                misses.append(f"{case.question} -> {articles[:k]}")
        rejected = sum(not retriever.retrieve(q, k=k).chunks for q in eval_set.off_topic)
        total = len(eval_set.cases)
        return EvalResult(
            threshold=retriever.config.score_threshold,
            filtered=filtered,
            k=k,
            hit_rate=round(hits / total, 4) if total else 0.0,
            mrr=round(reciprocal / total, 4) if total else 0.0,
            off_topic_rejected=(
                round(rejected / len(eval_set.off_topic), 4) if eval_set.off_topic else 1.0
            ),
            misses=misses,
        )
    finally:
        retriever.config = original


def sweep(retriever: Retriever, eval_set: EvalSet, thresholds: Sequence[float]) -> list[EvalResult]:
    """Evaluate each threshold with and without category filtering."""
    return [
        evaluate(retriever, eval_set, threshold=t, filtered=f)
        for t in thresholds
        for f in (False, True)
    ]
