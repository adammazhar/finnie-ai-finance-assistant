import os
import sys
import types

import pytest

from src.core.config import RAGConfig
from src.rag import embeddings as embeddings_module
from src.rag.embeddings import EmbeddingModelUnavailableError, build_embeddings


class Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class MissingModel:
    def __init__(self, **kwargs):
        raise OSError("We couldn't connect to huggingface.co and the files aren't cached")


@pytest.fixture
def fake_huggingface(monkeypatch):
    """Stand-in module so tests never import torch or download a model."""
    module = types.ModuleType("langchain_huggingface")
    module.HuggingFaceEmbeddings = Recorder
    monkeypatch.setitem(sys.modules, "langchain_huggingface", module)
    # register the variables so monkeypatch restores them after the test
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    embeddings_module.get_embeddings.cache_clear()
    yield module
    embeddings_module.get_embeddings.cache_clear()


def test_build_embeddings_normalizes(fake_huggingface):
    built = build_embeddings(RAGConfig(embedding_model="some/model"))
    assert built.kwargs == {"model": "some/model", "encode_kwargs": {"normalize_embeddings": True}}
    assert os.environ["HF_HUB_OFFLINE"] == "0"  # online unless asked


def test_offline_mode_sets_env_and_live_hub_setting(fake_huggingface, monkeypatch):
    hub = types.ModuleType("huggingface_hub")
    hub.constants = types.SimpleNamespace(HF_HUB_OFFLINE=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    build_embeddings(RAGConfig(), offline=True)
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert hub.constants.HF_HUB_OFFLINE is True


def test_offline_mode_without_cached_model_explains_the_fix(fake_huggingface, monkeypatch):
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    fake_huggingface.HuggingFaceEmbeddings = MissingModel
    with pytest.raises(EmbeddingModelUnavailableError, match="HF_HUB_OFFLINE=0 python scripts"):
        build_embeddings(RAGConfig(), offline=True)
    with pytest.raises(OSError, match="couldn't connect"):
        build_embeddings(RAGConfig(), offline=False)


@pytest.mark.parametrize(("env", "offline"), [(None, False), ("1", True), ("0", False)])
def test_get_embeddings_uses_the_setting(fake_huggingface, monkeypatch, env, offline):
    calls = []
    monkeypatch.setattr(
        embeddings_module,
        "build_embeddings",
        lambda config, offline: calls.append(offline) or "model",
    )
    if env is None:
        monkeypatch.delenv("HF_HUB_OFFLINE")
    else:
        monkeypatch.setenv("HF_HUB_OFFLINE", env)
    from src.core.config import get_settings

    get_settings.cache_clear()
    assert embeddings_module.get_embeddings() == "model"
    assert embeddings_module.get_embeddings() == "model"  # cached
    assert calls == [offline]
    get_settings.cache_clear()
