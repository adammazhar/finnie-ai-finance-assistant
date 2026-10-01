import yaml

from src.rag.chunking import build_chunks
from src.rag.evaluation import EvalCase, EvalSet, evaluate, load_eval_set, sweep
from src.rag.index import VectorIndex
from src.rag.retriever import Retriever

EVAL = EvalSet(
    cases=[
        EvalCase(question="wash sale loss thirty days", expected=["taxes-001"]),
        EvalCase(
            question="bond prices interest rates duration", expected=["bonds_fixed_income-001"]
        ),
        EvalCase(question="dividends stockholders profits", expected=["bonds_fixed_income-001"]),
    ],
    off_topic=["pizza dough basil oven", "bond"],
)


def make_retriever(kb, rag_config, embeddings):
    index = VectorIndex.build(build_chunks(rag_config, root=kb), embeddings, rag_config)
    return Retriever(index, embeddings, rag_config)


def test_evaluate_counts_hits_rank_and_off_topic(kb, rag_config, embeddings):
    retriever = make_retriever(kb, rag_config, embeddings)
    result = evaluate(retriever, EVAL)
    assert result.k == rag_config.top_k and not result.filtered
    assert result.hit_rate == round(2 / 3, 4)
    assert result.mrr == round(2 / 3, 4)  # both hits ranked first
    assert result.misses[0].startswith("dividends stockholders profits ->")
    assert result.off_topic_rejected == 0.5  # "bond" legitimately matches


def test_threshold_override_is_temporary_and_filtering(kb, rag_config, embeddings):
    retriever = make_retriever(kb, rag_config, embeddings)
    strict = evaluate(retriever, EVAL, threshold=0.99, filtered=True, k=2)
    assert strict.threshold == 0.99 and strict.filtered and strict.k == 2
    assert strict.hit_rate == 0 and strict.off_topic_rejected == 1.0
    assert retriever.config.score_threshold == rag_config.score_threshold


def test_sweep_covers_both_modes(kb, rag_config, embeddings):
    results = sweep(make_retriever(kb, rag_config, embeddings), EVAL, [0.2, 0.5])
    assert [(r.threshold, r.filtered) for r in results] == [
        (0.2, False),
        (0.2, True),
        (0.5, False),
        (0.5, True),
    ]


def test_empty_sets(kb, rag_config, embeddings):
    result = evaluate(make_retriever(kb, rag_config, embeddings), EvalSet(cases=[]))
    assert (result.hit_rate, result.mrr, result.off_topic_rejected) == (0.0, 0.0, 1.0)


def test_case_category_and_real_eval_file(tmp_path):
    assert EvalCase(question="q", expected=["bonds_fixed_income-004"]).category == (
        "bonds_fixed_income"
    )
    real = load_eval_set()
    assert len(real.cases) >= 40 and len(real.off_topic) >= 5
    path = tmp_path / "cases.yaml"
    path.write_text(
        yaml.safe_dump({"cases": [{"question": "q", "expected": ["taxes-001"]}]}), encoding="utf-8"
    )
    assert load_eval_set(path).off_topic == []
