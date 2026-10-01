# Finnie Benchmarks

Measured results for retrieval quality and performance, routing accuracy, and workflow latency. Each section names the command that reproduces it.

**Machine:** Windows 11, Intel Core (family 6 model 186), CPU only, Python 3.12.10. **Date:** 2026-09-30.

## Targets

The course documents ask for "performance considerations" and "performance benchmarks" but set no numeric response-time requirement. Their one number is a 30-minute TTL for cached market data, which Finnie uses for quotes and news. The targets below are Finnie's own. The scripts that check them exit non-zero when one is missed.

| Target | Value | Measured | Enforced by |
|---|---|---|---|
| Retrieval hit@4 | ≥ 85% | 93.3% unfiltered / 95.6% filtered | `pytest -m slow tests/evals` |
| Retrieval latency | p95 < 50 ms | 32.7 ms | measured (below) |
| LLM routing accuracy | ≥ 90% | 97.0% | `scripts/eval_routing.py` |
| Keyword fallback routing accuracy (regression floor) | ≥ 75% | 78.8% | `scripts/eval_routing.py`, and a unit test in CI |
| Single-specialist turn | p50 < 6 s | 5.8 s | reported by `scripts/bench_workflow.py` |
| Multi-specialist turn | p50 < 15 s | 12.9 s | `scripts/bench_workflow.py` |
| First visible progress in chat | p95 < 1 s | 17 ms | `scripts/bench_workflow.py` |

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

## Routing accuracy (Phase 7)

`python scripts/eval_routing.py` routes the 66 labelled questions in `tests/evals/routing_cases.yaml`. Each question lists the specialists it needs. The set includes:

- every specialist
- 6 multi-agent questions
- 6 out-of-scope questions
- 5 follow-ups that only make sense with the earlier conversation

A case is **correct** when every needed specialist is chosen (or an accepted alternative, for genuinely ambiguous questions) and out-of-scope questions are declined. **Exact** also requires no extra specialists. The keyword router is the fallback used when the LLM call fails.

| Router | Correct | Exact | Extra agents | p50 / p95 latency |
|---|---|---|---|---|
| **LLM (`gpt-4o-mini`, structured output)** | **97.0%** (64/66) | 90.9% | 4 | 967 / 1,349 ms |
| Keyword fallback | 78.8% | 77.3% | 1 | < 1 ms |

The LLM router's two misses: "Give me an overview of Costco as a company" (market expected, finance_qa chosen) and "What's a wash sale?" (tax expected, finance_qa chosen). Both still get a correct educational answer, just from the general specialist. Most of the extra agents come from the model sometimes adding the portfolio agent to a goal question when a portfolio is saved. The keyword router can't recognise out-of-scope questions or resolve follow-ups, which is why the LLM router is primary. Its 75% floor guards against regressions in the fallback path.

## Workflow latency (Phase 7)

`python scripts/bench_workflow.py` runs full turns through the LangGraph workflow:

- live models: `gpt-4o` for agents and merging, `gpt-4o-mini` for routing and guardrails
- live market data and the real index
- a saved 3-holding portfolio

Turns run through `FinnieAssistant.stream`, as the chat does. Goal questions state their savings or list holdings, so none stops to ask how much of the portfolio counts. Measured on 2026-10-01 (Phase 8).

| Turn type | n | Median | Max |
|---|---|---|---|
| Single specialist | 6 | 5.8 s | 8.4 s |
| Several specialists | 4 | 12.9 s | 15.8 s |
| Savings question (asked before the goal runs) | 2 | 1.5 s | 1.7 s |
| First progress event ("Reading your question…") | 10 | 7 ms | 17 ms |

Turn times vary by a few seconds between runs with model and API latency. In the Phase 7 run, the multi-specialist median was 14.1 s with a 20.0 s maximum. The slowest turns are those where one agent needs another's results: portfolio runs before goal planning or tax, and then a merge call follows. Two independent agents running in parallel take 10–12 s.

While a turn runs, the chat shows each step and specialist as it happens. The first update appears in milliseconds, which is well under the 1 s target. The answer streams in once the output guardrail has approved it.

## Test suite run time (Phase 8)

`pytest` runs 809 tests in parallel with pytest-xdist (`-n auto --dist loadgroup`) and coverage. CI runs the unit and UI suites as separate jobs and gates coverage on their combined data.

| Where | Configuration | Wall time |
|---|---|---|
| Local (12 logical CPUs) | serial, before these changes | ~190 s |
| Local | parallel, before fixing slow imports | 160 s |
| Local | parallel, after (`pytest`) | ~125 s |
| Local | unit suite only / UI suite only (as in CI) | 106 s / 49 s |
| CI (GitHub-hosted, 4 vCPUs) | unit job / UI job / combined coverage job | see the latest run |

Most of the gain came from imports, not parallelism alone. LangChain's text-splitters package imports sentence-transformers, transformers, and torch, which takes about 20 s per process, and each xdist worker paid that during collection. That import is now lazy. Agent and UI tests build their small search indexes directly, and the chunking tests share one worker, so only that worker pays the cost. The remaining long pole is that worker: about 60 s for the chunking, index, and retrieval tests, including the import.

