# Finnie API Reference

Finnie's main interface is the Streamlit app. The same functionality is available to Python code and to MCP clients:

1. **Python: the assistant.** `FinnieAssistant` is the whole multi-agent workflow behind the chat.
2. **Python: the domain layer.** Market data, portfolio analytics, Monte Carlo, tax, and knowledge base search. These are pure functions that need no LLM.
3. **MCP.** Six tools, two resources, and a prompt, for Claude Desktop, Claude Code, or any MCP client.
4. **External APIs** that Finnie calls, and what happens when they fail.

Every example below was run against the real system with a `.env` holding `OPENAI_API_KEY` (2026-10-04). Run them from the project root with the virtual environment active.

## 1. The assistant (`src/workflow/graph.py`)

```python
from src.core.models import Holding, UserProfile
from src.workflow.graph import FinnieAssistant

assistant = FinnieAssistant()  # real models, market data, and knowledge base (reads .env)
profile = UserProfile(knowledge_level="beginner", risk_tolerance="moderate")
portfolio = [Holding(ticker="VTI", shares=10), Holding(ticker="BND", shares=20)]

out = assistant.ask("What is an index fund?", thread_id="demo-1", profile=profile)
print(out.status, out.agents)  # answered ['finance_qa']
for source in out.sources:
    print(source.title, source.url)  # Index Funds and How They Work https://www.investor.gov/...

# The same call, with progress events while the specialists work (what the chat shows)
for event in assistant.stream(
    "How diversified is my portfolio?", thread_id="demo-1", portfolio=portfolio
):
    # Progress(kind='agent_started', message='Consulting the portfolio specialist…'), …,
    # then the TurnOutput
    print(event)

assistant.forget("demo-1")  # delete the conversation's memory
```

| Method | What it does |
|---|---|
| `FinnieAssistant(context=None, checkpointer=None, agents=None)` | Builds the LangGraph workflow. Defaults: real models and services from `.env` and `config.yaml`, and in-memory conversation memory. The app passes a SQLite checkpointer, so memory survives restarts. Tests pass fakes. |
| `ask(question, *, thread_id, profile=None, portfolio=None) -> TurnOutput` | Answers one message. A `thread_id` is one conversation; follow-ups in the same thread use its memory. `profile` and `portfolio` update what's saved for that thread. |
| `stream(...) -> Iterator[Progress \| TurnOutput]` | Like `ask`, but first yields `Progress` events (`status`, `agent_started`, `agent_finished`), then the `TurnOutput`. |
| `state(thread_id) -> dict` | The saved conversation: messages, summary, profile, portfolio, title. |
| `update_title(thread_id)` / `lock_title(thread_id, title)` | Automatic conversation titles, and a user-chosen title that automatic titling never replaces. |
| `forget(thread_id)` | Deletes the conversation's checkpoints. |
| `mermaid() -> str` | The compiled graph as a Mermaid diagram. |

**`TurnOutput`** (`src/workflow/nodes.py`):

| Field | Meaning |
|---|---|
| `answer` | The final markdown answer, with numbered citations and the disclaimer |
| `status` | `answered`, `blocked` (input screen), `out_of_scope`, `fallback` (every specialist failed), or `needs_input` (Finnie asked a question first, such as how much of the portfolio counts toward a goal) |
| `agents` | The specialists that answered |
| `sources` | `Source(title, kind, category, url, article_id, score, published_at)`, where `kind` is `knowledge_base`, `market_data`, or `news` |
| `data` | Structured results per agent, used for charts (allocation, projections, quotes, news) |
| `freshness` | When each piece of market data is from, and whether it's live, cached, stale, or mock |
| `guardrail` | `unchanged`, `rewritten` (advice-like wording reframed as education), or `neutralized` |
| `route`, `errors`, `merged_by_llm` | The routing decision, any specialist errors, and whether answers were merged by the model |

**`UserProfile`**: `knowledge_level` (`beginner` / `intermediate` / `advanced`), `risk_tolerance` (`conservative` / `moderate` / `aggressive`), plus optional `age` and `investment_horizon_years`. **`Holding`**: `ticker`, `shares`, and optional `cost_basis` (total cost).

## 2. The domain layer (no LLM needed)

```python
from src.core.config import get_settings
from src.core.monte_carlo import inputs_for_profile, required_monthly_contribution, simulate
from src.core.models import Holding
from src.core.portfolio import fetch_and_analyze
from src.core.reference import get_risk_profiles
from src.data.service import get_market_data_service
from src.rag.retriever import get_retriever

market = get_market_data_service()  # provider chain with cache, stale, and mock fallbacks
quote = market.get_quote("VTI")
# 377.99 Cached · just now · prices as of Oct 2, 04:00 PM ET
print(quote.price, quote.freshness.label())

moderate = get_risk_profiles()["moderate"]
holdings = [Holding(ticker="VTI", shares=10), Holding(ticker="BND", shares=20)]
analysis = fetch_and_analyze(holdings, market, profile=moderate)
# 5178.7 87.7 Moderate
print(analysis.total_value, analysis.diversification_score, analysis.risk_level)

inputs = inputs_for_profile(
    moderate,
    get_settings().analytics.monte_carlo,
    current_balance=5_000,
    monthly_contribution=300,
    years=10,
    target_amount=50_000,
    seed=42,
)
# 0.28 420.49
print(simulate(inputs).success_probability, required_monthly_contribution(inputs, 0.8))

hits = get_retriever().retrieve("How do expense ratios work?", categories=["funds_etfs"], k=3)
for item in hits.chunks:
    # 0.74 Expense Ratios and Why Small Differences Add Up
    print(round(item.score, 2), item.chunk.title)
```

| Module | Main functions |
|---|---|
| `src/data/service.py` | `get_market_data_service()` returns a service with `get_quote`, `get_quotes`, `get_daily_history`, `get_company_overview`, `get_news`, `get_treasury_bill_yield`, and `provider_status`. Every result carries a `Freshness`. |
| `src/core/portfolio.py` | `fetch_and_analyze(holdings, market, profile=...)` returns a `PortfolioAnalysis`: value, weights, asset and sector allocation, HHI diversification score, risk level and score, expense ratio, past-year return, volatility, drawdown, beta, Sharpe, and observations. `analyze_portfolio(...)` does the same from data you've already fetched. `backtest(...)` compares the holdings with SPY. |
| `src/core/monte_carlo.py` | `inputs_for_profile(...)`, `simulate(inputs)` (success probability, percentiles, fan chart data), `required_monthly_contribution(inputs, p)`, and `deterministic_future_value(...)` |
| `src/core/indicators.py` | `build_market_overview(market)` (indexes, sectors, mood), `technical_snapshot(history)`, `sma`, `rsi`, `realized_volatility`, and `latest_cross` (moving-average crossovers) |
| `src/core/tax.py` | `get_tax_reference()` (2026 IRS figures with source URLs), `compare_accounts(reference, keys)`, `exceptions_for(reference, "plans" \| "iras")` (early-withdrawal exceptions by account type), `rmd_age_for(reference, birth_year=...)` and `rmd_age_for_current_age(...)`, and the capital gains illustration |
| `src/core/voice.py` | `transcribe(audio_bytes, settings)` returns the text of a spoken question (OpenAI `whisper-1`; raises `VoiceError` with a message safe to show); `speakable_text(answer)` returns an answer as plain sentences for reading aloud |
| `src/data/symbols.py` | `get_symbol_directory().search("Apple")` returns `SymbolMatch(ticker, name, kind, source)`, best first, from the indexes, Finnie's fund catalog, and the SEC company list; `yahoo_search(query)` is the online fallback |
| `src/rag/retriever.py` | `get_retriever().retrieve(query, categories=None, k=None)`: relevance threshold, MMR, category filter with widening, and sources on every chunk |
| `src/core/llm.py` | `get_llm("main" \| "fast")`: the configured provider with automatic fallback to the other one |

## 3. MCP server

`python -m src.mcp_server` runs over stdio, for Claude Desktop. `python -m src.mcp_server --http` serves `http://127.0.0.1:8765/mcp` and needs `Authorization: Bearer <MCP_API_TOKEN>`. It implements MCP specification 2026-07-28 with the official SDK. Every tool has a typed input and output schema.

| Tool | Input | Output |
|---|---|---|
| `get_stock_quote` | `ticker` | price, change, exact price time, live/delayed/last close, market status |
| `get_market_overview` | none | 4 index levels with their tracking ETFs, 11 sectors, mood summary |
| `analyze_portfolio` | `holdings: [{ticker, shares, cost_basis?}]`, `risk_tolerance?` | value, holdings, allocations, diversification, risk, expense ratio, past year |
| `project_financial_goal` | `target_amount`, `years`, `current_savings?`, `monthly_contribution?`, `risk_tolerance?`, `target_in_todays_dollars?` | chance of success, P10/median/P90, contribution for an 80% chance, assumptions |
| `search_financial_knowledge` | `query`, `category?`, `limit?` | passages with relevance, sources, and an article resource URI |
| `explain_tax_account` | `account_type` (`traditional_401k`, `roth_401k`, `traditional_ira`, `roth_ira`, `hsa`, `plan_529`, `taxable_brokerage`) | rules, this year's limits with IRS sources |

The resources are `finnie://articles/{article_id}` and `finnie://glossary`, and the prompt is `explain_like_beginner(topic)`. Errors come back as MCP tool errors with a plain message. `scripts/mcp_client_demo.py` is a complete Python client example. Setup for Claude Desktop and Claude Code, and the HTTP security rules, are in [MCP.md](MCP.md).

## 4. External APIs Finnie uses

| API | Used for | Key | When it fails |
|---|---|---|---|
| OpenAI (`gpt-4o`, `gpt-4o-mini`) | answers; routing, checks, titles | `OPENAI_API_KEY` | SDK retries with backoff, then Anthropic |
| Anthropic (`claude-sonnet-5-5`, `claude-haiku-4-5`) | fallback, or the primary provider | `ANTHROPIC_API_KEY` | the turn ends with a plain fallback reply |
| yfinance | quotes, history, news | none | Alpha Vantage, then the stale cache, then labelled mock data |
| Alpha Vantage | quotes, company facts, news sentiment | `ALPHA_VANTAGE_API_KEY` | rate-limit detection (including its HTTP 200 "Note" body), a daily budget, backoff |
| Tavily | news search | `TAVILY_API_KEY` | skipped |
| Hugging Face Hub | one-time download of `all-MiniLM-L6-v2` | none | offline afterwards (`HF_HUB_OFFLINE=1`; built into the Docker image) |

Market data is cached in SQLite (`config.yaml` → `market_data`):
- **Quotes:** 60 seconds while the market is open. While it's closed, up to 30 minutes, and never older than the last close.
- **News:** 30 minutes.
- **Price history:** 12 hours.

Every result says where it came from and how fresh it is.
