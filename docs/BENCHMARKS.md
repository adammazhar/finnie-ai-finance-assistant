# Finnie Benchmarks

Measured results for the final system: retrieval quality and speed, routing accuracy, workflow latency, the MCP server, the Docker image, a fresh-clone install, and the test suite. Each section names the command that reproduces it.

**Machine:** Windows 11, Intel Core (family 6 model 186), 12 logical CPUs, 64 GB RAM, CPU only, Python 3.12.10. Docker numbers come from GitHub's hosted Linux runners (4 vCPUs). Each section gives its date; the last update is 2026-10-02.

## Targets

The course documents ask for "performance considerations" and "performance benchmarks" but set no numeric response-time requirement. Their one number is a 30-minute TTL for cached market data, which Finnie uses for quotes and news. The targets below are Finnie's own. The scripts that check them exit non-zero when one is missed.

| Target | Value | Measured | Enforced by |
|---|---|---|---|
| Retrieval hit@4 | ≥ 85% | 91.1% unfiltered / 93.3% filtered (re-run 2026-10-04) | `pytest -m slow tests/evals` |
| Retrieval latency | p95 < 50 ms | 32.7 ms | measured (below) |
| LLM routing accuracy | ≥ 90% | 97.0% (re-run 2026-10-04) | `scripts/eval_routing.py` |
| Keyword fallback routing accuracy (regression floor) | ≥ 75% | 78.8% | `scripts/eval_routing.py`, and a unit test in CI |
| Single-specialist turn | p50 < 6 s | 5.8 s | reported by `scripts/bench_workflow.py` |
| Multi-specialist turn | p50 < 15 s | 12.9 s | `scripts/bench_workflow.py` |
| First visible progress in chat | p95 < 1 s | 17 ms | `scripts/bench_workflow.py` |
| Test coverage | ≥ 90% | 100% (1,027 tests) | `pytest` and the CI `coverage` job |
| MCP tool call, server warm | < 1 s | 4–256 ms median | measured by `scripts/bench_mcp.py` |
| Docker: healthy after `docker compose up` | < 60 s | about 10 s | the CI `docker` job |
| Voice: transcription (median) | < 3 s | 1.15–1.56 s for 2–16 s questions | measured by `scripts/bench_voice.py` |

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

### Re-run after the persona-testing fixes (2026-10-04)

The knowledge base grew by one article (the Rule of 55) and one glossary term (pump-and-dump scheme), to 113 articles, 173 terms, and 1,108 chunks. At the chosen 0.40 threshold:

| | Hit@4 unfiltered | Hit@4 filtered | MRR (unfiltered / filtered) | Off-topic rejected |
|---|---|---|---|---|
| Before (Phase 5) | 93.3% | 95.6% | 0.891 / 0.902 | 100% |
| **After** | **91.1%** | **93.3%** | **0.885 / 0.896** | **100%** |

The drop is one new miss, "At what age do I have to start withdrawing from my IRA?". The new Rule of 55 article, which is also about withdrawal ages, now ranks in the top 4 ahead of the RMD article. Both are relevant, and the tax agent answers RMD-age questions from IRS reference data (`get_withdrawal_rules`) rather than from retrieval alone. The evaluation set was not changed to recover the number, and both figures stay above the 85% target that `pytest -m slow tests/evals` enforces (which passes).

## Retrieval performance (Phase 5)

| Measure | Result |
|---|---|
| Corpus | 112 articles + 172 glossary terms → 1,100 chunks (928 article, 172 glossary) |
| Chunk size | median 470 characters, max 798; max embedded text (with title/section prefix) 883 characters, under MiniLM's ~1,000-character limit |
| Index build (`scripts/build_index.py`, includes model load) | 55.7 s |
| Cold start (imports, model load, load saved index) | 21.0 s, once per process |
| Retrieval latency over 135 queries (embed query + exact search + filter + MMR) | p50 27.7 ms, p95 32.7 ms, max 48.1 ms (target < 50 ms) |

## Routing accuracy (Phase 7)

**Re-run on 2026-10-04**, after the docstrings (the router's output schema gained a description) and the stricter beginner prompt: LLM router **97.0%** correct, 90.9% exact, with the same two misses; keyword fallback 78.8%. Unchanged from the table below.


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

## MCP server (Phases 9–10)

`python scripts/bench_mcp.py` starts the Streamable HTTP server in-process on loopback, with a throwaway token, and calls each tool through a real MCP client. The full HTTP path is measured: bearer-token check, the SDK, and JSON-RPC. Measured on 2026-10-02.

**Knowledge base cold start:** the first search answered **27.1 s** after the server started.
- That is the one-time load of the embedding model and FAISS index (sentence-transformers and torch imports included).
- The server begins this load in a background thread at startup (decision 33), and answers `tools/list` right away.
- In practice the load is usually done before the first question arrives. Before the warm-up was added, a first search took 34 s from the moment it was asked (measured while the machine was also busy installing packages).

With the server warm:

| Tool | First call | Median of next 10 |
|---|---|---|
| `explain_tax_account` | 13 ms | 4 ms |
| `project_financial_goal` (10,000 Monte Carlo paths + the 80% contribution solver) | 259 ms | 256 ms |
| `search_financial_knowledge` | 27 ms | 29 ms |
| `get_stock_quote` | 16 ms | 7 ms |
| `get_market_overview` (4 index levels, 4 ETFs, 11 sectors) | 62 ms | 25 ms |
| `analyze_portfolio` (2 holdings, with a year of history) | 60 ms | 46 ms |

Market data was already in the SQLite cache from an earlier run, so first calls show cached reads. On a cold cache, a first quote took about 430 ms and the market overview about 590 ms (yfinance).

## Docker image (Phase 10)

Measured by the `docker` job in CI on GitHub's Linux runners (2026-10-02), because the development PC has no Docker.

| Measure | Result |
|---|---|
| Image size | 2.27 GB (CPU-only `torch 2.14.1+cpu`; PyTorch is most of it) |
| Build, no cache | about 4.5 min, including about 70 s to install dependencies and 30–55 s to download the model and build the index |
| Build, dependencies cached | about 3.5 min (the index layer rebuilds when code changes) |
| `docker compose up` to healthy | about 10 s |
| With networking switched off (`--network none`) | the model loads, a search returns "Expense Ratios and Why Small Differences Add Up", the setup page shows without a key, and onboarding renders with a placeholder key |
| MCP profile | 401 without the token, a valid response with it |

## Fresh-clone install (Phase 10)

The repository was cloned from GitHub into an empty folder and set up by following only the README on Windows 11 (PowerShell), with an empty Hugging Face cache (2026-10-02).

| README step | Result |
|---|---|
| 1. venv + `pip install -r requirements-dev.txt` + `pip install -e . --no-deps` | succeeded (several minutes, mostly the PyTorch download) |
| 2. `.env` from `.env.example` | as documented |
| 3. download the model and build the index | 1.4 min: 1,100 chunks, 384 dimensions |
| `pytest` | 919 passed, 100% coverage, 2.4 min |
| `scripts/validate_kb.py` | Knowledge base OK |
| 4. `python -m src.web_app` | onboarding, then a chat answer with citations (1,060 characters), then the Knowledge tab, in a real browser |

Gaps found and fixed:
- With no API key, the app showed an error instead of starting. It now shows a setup page naming the missing key (decision 32).
- The README now says to create the venv with Python 3.12 explicitly. On a machine where the `py` launcher defaults to 3.14, a bare `py -m venv` would have picked the wrong version.

## Voice transcription

`python scripts/bench_voice.py DIR --runs 5` sends each recording in `DIR` (`name.wav` plus `name.txt` with what is said) through Finnie's voice module five times. It uses the same call the chat makes: OpenAI `whisper-1` with the finance vocabulary prompt. Measured on 2026-10-04.

| Recording | Length | Median | Slowest | Word error rate |
|---|---|---|---|---|
| "What is an ETF?" | 2.1 s | 1.15 s | 1.98 s | 0% |
| "How diversified is my portfolio, and what does its expense ratio mean for me?" | 5.7 s | 1.19 s | 1.97 s | 0% |
| A 41-word question about a 401(k), a Roth IRA, RMDs, and early-withdrawal penalties | 15.6 s | 1.56 s | 2.33 s | 2% |

- The only error in the long question was "distribution" for "distributions". Numbers come back as digits ("58", "401k"), so the expected text is written the same way, and "401(k)" and "401k" count as one word.
- **End to end in a browser** (Chromium with a simulated microphone playing the 5-second recording), from "Submit recording" to the transcript sitting in the chat box: **4.3 s**. That includes the upload and Streamlit's rerun.
- **Caveat:** these recordings are synthetic speech from Windows' built-in voice, which is cleaner than a real microphone in a real room. Expect some more errors with real voices, accents, and background noise, which is why the transcript goes into the box to be checked before it's sent.
- **Reading aloud** happens in the browser and starts at once; there is no service call to measure.

To make the recordings on Windows (PowerShell), for each question:

```powershell
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile("samples\1_short.wav", $fmt); $s.Speak("What is an ETF?"); $s.Dispose()
Set-Content samples\1_short.txt "What is an ETF?"
```

## Test suite run time

`pytest` runs **1,027 tests** in parallel with pytest-xdist (`-n auto --dist loadgroup`), with branch coverage at 100%. CI runs the unit and UI suites as separate jobs and gates coverage on their combined data.

| Where | Configuration | Wall time |
|---|---|---|
| Local (12 logical CPUs) | serial, Phase 8 before the speed-ups | ~190 s |
| Local | parallel, Phase 8 before fixing slow imports | 160 s |
| Local | parallel, final (`pytest`) | ~130 s |
| CI (GitHub-hosted, 4 vCPUs) | `unit` / `ui` / `coverage` / `kb-links` / `docker` jobs | 4.1 / 2.2 / 0.2 / 2.9 / 4.6 min, in parallel except coverage |

Most of the Phase 8 gain came from imports, not parallelism alone:
- **The slow import:** LangChain's text-splitters package imports sentence-transformers, transformers, and torch, which takes about 20 s per process. Each xdist worker paid that during collection.
- **The fix:** that import is now lazy. Agent and UI tests build their small search indexes directly, and the chunking tests share one worker, so only that worker pays the cost.
- **What's left:** the slowest part is now that worker, at about 60 s for the chunking, index, and retrieval tests, including the import.
