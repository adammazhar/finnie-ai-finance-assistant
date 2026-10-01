# Finnie Benchmarks

Measured results for retrieval quality and performance. Each section names the command that reproduces it. Later phases add LLM, workflow, and end-to-end latency.

**Machine:** Windows 11, Intel Core (family 6 model 186), CPU only, Python 3.12.10. **Date:** 2026-09-30.

## Retrieval quality (Phase 5)

`python scripts/eval_retrieval.py` runs the evaluation set in `tests/evals/retrieval_cases.yaml`: 45 questions phrased the way a user would ask, each labelled with the article(s) that should be retrieved, plus 8 off-topic questions that should retrieve nothing.

- **Hit@4:** at least one expected article appears in the top 4 chunks. Glossary chunks neither help nor hurt.
- **MRR:** mean reciprocal rank of the first expected article.
- **Filtered:** the question's category is passed, as agents do. **Unfiltered:** the whole knowledge base is searched.

| Score threshold | Hit@4 unfiltered | Hit@4 filtered | MRR (unfiltered / filtered) | Off-topic rejected |
|---|---|---|---|---|
| 0.25 | 93.3% | 95.6% | 0.891 / 0.902 | 75% |
| 0.30 | 93.3% | 95.6% | 0.891 / 0.902 | 100% |
| 0.35 | 93.3% | 95.6% | 0.891 / 0.902 | 100% |
| **0.40 (chosen)** | **93.3%** | **95.6%** | **0.891 / 0.902** | **100%** |
| 0.45 | 95.6% | 95.6% | 0.896 / 0.902 | 100% |
| 0.50 | 95.6% | 95.6% | 0.898 / 0.887 | 100% |

The design target was hit@4 ≥ 85%.

**Why 0.40:**
- The highest-scoring off-topic question reaches 0.28.
- The weakest score for an expected article in any case is 0.51.
- 0.40 sits near the middle of that gap, so it doesn't overfit a 45-question set.

**Remaining misses at 0.40:** "What does it mean to own shares of a company?" and "What is an exchange-traded fund?". Both are still answered well, because the top results are the exact glossary definitions ("Share" at 0.69, "Exchange-traded fund (ETF)" at 0.80). The unfiltered run misses one more: "Why shouldn't I put all my money in one stock?".

`pytest -m slow --no-cov tests/evals` enforces the 85% target and full off-topic rejection with the real model, offline.

## Retrieval performance (Phase 5)

| Measure | Result |
|---|---|
| Corpus | 112 articles + 172 glossary terms → 1,100 chunks (928 article, 172 glossary) |
| Chunk size | median 470 characters, max 798; max embedded text (with title/section prefix) 883 characters, under MiniLM's ~1,000-character limit |
| Index build (`scripts/build_index.py`, includes model load) | 55.7 s |
| Cold start (imports, model load, load saved index) | 21.0 s, once per process |
| Retrieval latency over 135 queries (embed query + exact search + filter + MMR) | p50 27.7 ms, p95 32.7 ms, max 48.1 ms (target < 50 ms) |
