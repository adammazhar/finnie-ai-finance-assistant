"""Retrieval quality with the real MiniLM model on the real knowledge base.

Slow and excluded from the default run. It needs the model in the local Hugging Face
cache (downloaded once by ``python scripts/build_index.py``) and runs offline:

    pytest -m slow --no-cov tests/evals
"""

import pytest

pytestmark = pytest.mark.slow

TARGET_HIT_RATE = 0.85  # design target (docs/DESIGN.md section 11)


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    from src.core.config import get_settings
    from src.rag.embeddings import EmbeddingModelUnavailableError, build_embeddings
    from src.rag.retriever import build_retriever

    try:
        embeddings = build_embeddings(get_settings().rag, offline=True)
    except EmbeddingModelUnavailableError as exc:
        pytest.skip(f"embedding model not available offline: {exc}")
    return build_retriever(embeddings=embeddings, index_dir=tmp_path_factory.mktemp("vs"))


@pytest.mark.parametrize("filtered", [False, True])
def test_hit_rate_and_off_topic_rejection(retriever, filtered):
    from src.rag.evaluation import evaluate, load_eval_set

    result = evaluate(retriever, load_eval_set(), filtered=filtered)
    assert result.hit_rate >= TARGET_HIT_RATE, "\n".join(result.misses)
    assert result.off_topic_rejected == 1.0
