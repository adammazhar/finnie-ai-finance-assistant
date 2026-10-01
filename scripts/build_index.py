"""Build or refresh the FAISS index: python scripts/build_index.py [--force]."""

import argparse
import time

from src.core.config import get_settings
from src.rag.embeddings import get_embeddings
from src.rag.index import ensure_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if current")
    args = parser.parse_args()
    settings = get_settings()
    started = time.perf_counter()
    index = ensure_index(
        settings.rag,
        get_embeddings(),
        index_dir=settings.resolve_path(settings.rag.index_dir),
        force=args.force,
    )
    print(
        f"{len(index)} chunks, {index.dimensions} dimensions, "
        f"{time.perf_counter() - started:.1f}s ({index.meta['built_at']})"
    )
