import json

import numpy as np
import pytest

from src.core.config import RAGConfig
from src.rag.chunking import build_chunks
from src.rag.index import (
    CHUNKS_FILE,
    INDEX_FILE,
    META_FILE,
    VectorIndex,
    VectorIndexError,
    content_hash,
    ensure_index,
)

# Real chunking imports LangChain's text splitters (and with them torch), which takes about
# 20 s per process. One xdist worker runs all such tests, so only it pays that cost.
pytestmark = pytest.mark.xdist_group("text_splitters")


def test_build_and_score(kb, rag_config, embeddings):
    chunks = build_chunks(rag_config, root=kb)
    index = VectorIndex.build(chunks, embeddings, rag_config, kb_hash="h")
    assert len(index) == len(chunks) and index.dimensions == 256
    assert index.meta["kb_hash"] == "h" and index.meta["embedding_model"].endswith("MiniLM-L6-v2")
    scores = index.scores(embeddings.embed_query("wash sale loss"))
    best = index.chunks[int(np.argmax(scores))]
    assert best.section == "Wash sale"
    assert scores.max() <= 1.0001 and scores.shape == (len(chunks),)


def test_scores_reject_wrong_dimensions(kb, rag_config, embeddings):
    index = VectorIndex.build(build_chunks(rag_config, root=kb), embeddings, rag_config)
    with pytest.raises(VectorIndexError, match="dimensions"):
        index.scores([1.0, 0.0])


def test_constructor_and_build_validation(kb, rag_config, embeddings):
    chunks = build_chunks(rag_config, root=kb)
    with pytest.raises(VectorIndexError, match="count"):
        VectorIndex(chunks, np.ones((1, 4)), {})
    with pytest.raises(VectorIndexError, match="no chunks"):
        VectorIndex.build([], embeddings, rag_config)


def test_zero_vectors_are_safe(kb, rag_config):
    chunks = build_chunks(rag_config, root=kb)[:2]
    index = VectorIndex(chunks, np.zeros((2, 8)), {})
    assert np.allclose(index.vectors, 0)


def test_save_and_load_round_trip_without_pickle(kb, rag_config, embeddings, tmp_path):
    index = VectorIndex.build(build_chunks(rag_config, root=kb), embeddings, rag_config)
    directory = tmp_path / "vs"
    index.save(directory)
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        [INDEX_FILE, CHUNKS_FILE, META_FILE]
    )
    json.loads((directory / CHUNKS_FILE).read_text(encoding="utf-8"))  # plain JSON, not pickle
    loaded = VectorIndex.load(directory)
    assert [c.id for c in loaded.chunks] == [c.id for c in index.chunks]
    query = embeddings.embed_query("bond coupon")
    assert np.allclose(loaded.scores(query), index.scores(query), atol=1e-6)


def test_load_errors(kb, rag_config, embeddings, tmp_path):
    with pytest.raises(VectorIndexError, match="can't load"):
        VectorIndex.load(tmp_path / "missing")
    directory = tmp_path / "vs"
    VectorIndex.build(build_chunks(rag_config, root=kb), embeddings, rag_config).save(directory)
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    (directory / META_FILE).write_text(json.dumps(meta | {"format_version": 0}), encoding="utf-8")
    with pytest.raises(VectorIndexError, match="format version"):
        VectorIndex.load(directory)
    (directory / CHUNKS_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(VectorIndexError):
        VectorIndex.load(directory)


def test_content_hash_tracks_changes_but_not_line_endings(kb):
    original = content_hash(kb)
    path = kb / "stocks" / "stocks-001.md"
    text = path.read_text(encoding="utf-8")
    path.write_bytes(text.replace("\n", "\r\n").encode())
    assert content_hash(kb) == original
    path.write_text(text + "\nMore.\n", encoding="utf-8")
    assert content_hash(kb) != original


def test_ensure_index_builds_reuses_and_rebuilds(kb, rag_config, embeddings, tmp_path, caplog):
    caplog.set_level("INFO")
    directory = tmp_path / "vs"
    first = ensure_index(rag_config, embeddings, index_dir=directory, kb_root=kb)
    built = embeddings.documents_embedded
    assert built == len(first) and (directory / META_FILE).is_file()

    again = ensure_index(rag_config, embeddings, index_dir=directory, kb_root=kb)
    assert embeddings.documents_embedded == built and len(again) == len(first)  # reused

    (kb / "taxes" / "taxes-001.md").write_text(
        (kb / "taxes" / "taxes-001.md").read_text(encoding="utf-8") + "\nUpdated.\n",
        encoding="utf-8",
    )
    ensure_index(rag_config, embeddings, index_dir=directory, kb_root=kb)
    assert embeddings.documents_embedded > built and "rebuilding" in caplog.text

    count = embeddings.documents_embedded
    changed = rag_config.model_copy(update={"chunk_size": 150})
    ensure_index(changed, embeddings, index_dir=directory, kb_root=kb)
    assert embeddings.documents_embedded > count

    count = embeddings.documents_embedded
    ensure_index(changed, embeddings, index_dir=directory, kb_root=kb, force=True)
    assert embeddings.documents_embedded > count


def test_ensure_index_recovers_from_corruption(kb, rag_config, embeddings, tmp_path, caplog):
    directory = tmp_path / "vs"
    ensure_index(rag_config, embeddings, index_dir=directory, kb_root=kb)
    (directory / INDEX_FILE).write_bytes(b"garbage")
    index = ensure_index(rag_config, embeddings, index_dir=directory, kb_root=kb)
    assert len(index) > 0 and "Rebuilding unusable index" in caplog.text


def test_different_model_triggers_rebuild(kb, rag_config, embeddings, tmp_path):
    directory = tmp_path / "vs"
    ensure_index(rag_config, embeddings, index_dir=directory, kb_root=kb)
    count = embeddings.documents_embedded
    other = RAGConfig(**(rag_config.model_dump() | {"embedding_model": "other/model"}))
    ensure_index(other, embeddings, index_dir=directory, kb_root=kb)
    assert embeddings.documents_embedded > count


def test_content_hash_without_glossary(kb):
    with_glossary = content_hash(kb)
    (kb / "glossary.yaml").unlink()
    assert content_hash(kb) != with_glossary
