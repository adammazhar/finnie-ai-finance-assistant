"""Evaluate retrieval quality on the real index: python scripts/eval_retrieval.py."""

from src.rag.evaluation import load_eval_set, sweep
from src.rag.retriever import build_retriever

THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)

if __name__ == "__main__":
    retriever = build_retriever()
    eval_set = load_eval_set()
    print(
        f"{len(eval_set.cases)} cases, {len(eval_set.off_topic)} off-topic questions, "
        f"k={retriever.config.top_k}"
    )
    print("threshold  filtered  hit@k   MRR    off-topic rejected")
    for r in sweep(retriever, eval_set, THRESHOLDS):
        print(
            f"{r.threshold:>9.2f}  {r.filtered!s:<8}  {r.hit_rate:>5.1%}  {r.mrr:.3f}  "
            f"{r.off_topic_rejected:>6.1%}"
        )
        for miss in r.misses:
            print(f"           miss: {miss}")
