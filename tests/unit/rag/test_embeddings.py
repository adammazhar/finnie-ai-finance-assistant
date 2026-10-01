import sys
import types

import pytest

from src.core.config import RAGConfig
from src.rag import embeddings as embeddings_module


class Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def fake_huggingface(monkeypatch):
    """Stand-in module so tests never import torch or download a model."""
    module = types.ModuleType("langchain_huggingface")
    module.HuggingFaceEmbeddings = Recorder
    monkeypatch.setitem(sys.modules, "langchain_huggingface", module)
    embeddings_module.get_embeddings.cache_clear()
    yield
    embeddings_module.get_embeddings.cache_clear()


def test_build_embeddings_normalizes():
    built = embeddings_module.build_embeddings(RAGConfig(embedding_model="some/model"))
    assert built.kwargs == {"model": "some/model", "encode_kwargs": {"normalize_embeddings": True}}


def test_get_embeddings_is_cached():
    first = embeddings_module.get_embeddings()
    assert first is embeddings_module.get_embeddings()
    assert first.kwargs["model"].endswith("all-MiniLM-L6-v2")
