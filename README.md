# Finnie: AI Finance Assistant

Finnie is a multi-agent assistant that teaches beginners about investing. Six specialist agents are orchestrated with LangGraph. Answers are grounded in a 112-article knowledge base through RAG and combined with live market data. **Finnie provides education, not financial advice.**

> This README is a work in progress. The full version (architecture, usage, API reference, troubleshooting) arrives with the final phase. The design is in [docs/DESIGN.md](docs/DESIGN.md), and measured results are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Setup

**Requirements:** Python 3.12, about 2 GB of free disk (CPU-only PyTorch plus the embedding model), and an OpenAI or Anthropic API key.

### 1. Create the environment and install

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate      Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt   # runtime + test tools; CPU-only PyTorch comes from the index in requirements.txt
pip install -e . --no-deps            # makes the `src` package importable
```

For runtime only, `pip install -r requirements.txt` is enough.

### 2. Configure secrets

```bash
cp .env.example .env    # Windows: copy .env.example .env
```

Fill in `OPENAI_API_KEY` (the default provider) and optionally `ANTHROPIC_API_KEY`, which is used as an automatic fallback. Alpha Vantage and Tavily keys are optional: without them, Finnie uses yfinance for market data and news. Never commit `.env`; it is git-ignored. Non-secret settings are in `config.yaml`.

### 3. Download the embedding model and build the search index (one time)

`.env.example` sets `HF_HUB_OFFLINE=1`, so Finnie loads the embedding model (`all-MiniLM-L6-v2`, about 90 MB) only from the local Hugging Face cache and never contacts the Hub at startup. The first time, while the cache is empty, allow the download once:

```bash
# macOS/Linux
HF_HUB_OFFLINE=0 python scripts/build_index.py
```

```powershell
# Windows PowerShell
$env:HF_HUB_OFFLINE = "0"; python scripts/build_index.py; Remove-Item Env:HF_HUB_OFFLINE
```

This downloads the model into the Hugging Face cache (`~/.cache/huggingface` by default) and builds the FAISS index in `data/vectorstore/`. After that, everything runs offline. The index rebuilds automatically whenever knowledge base articles change. If the model is missing while `HF_HUB_OFFLINE=1`, Finnie stops with a message showing this command.

### 4. Check the installation

```bash
pytest                                   # all tests, in parallel; the network is blocked, so no API calls
python scripts/validate_kb.py            # knowledge base rules
python scripts/check_kb_links.py         # every source link (needs internet)
python scripts/eval_retrieval.py         # retrieval quality on the evaluation set
python scripts/eval_routing.py           # routing accuracy (calls the fast model once per case)
python scripts/bench_workflow.py         # end-to-end latency with live models and data
```

### 5. Run the app

```bash
streamlit run src/web_app/app.py
```

Open http://localhost:8501. On a first visit Finnie asks for your knowledge level and risk tolerance; there's a 5-question quiz if you're unsure. The sidebar has **New conversation**, your recent conversations (rename or delete one from its **⋯** menu), a **Profile** button to change those answers, and a system status icon.

Finnie remembers you without a login. Your profile, portfolio, and conversations are saved in `data/app/finnie.sqlite`, which is git-ignored, under a random ID stored in a browser cookie. They're still there after a refresh or a restart. **Delete my data** on the Profile page removes everything for this browser. The tabs:

- **Chat**: ask anything in the box pinned at the bottom. You see which specialists are working while they run. The answer then streams in, with the specialists who wrote it, the sources it cites (open any knowledge base article in Knowledge), and charts when relevant. When you ask about a savings goal and have a saved portfolio, Finnie first asks how much of the portfolio counts toward that goal.
- **Portfolio**: enter holdings in the table, upload a CSV (`ticker,shares,cost_basis`), or load a sample. You get allocation, diversification, risk, the portfolio expense ratio, past-year risk measures, a back-test of today's holdings against SPY, and a comparison with a typical mix for your risk tolerance. Holdings saved here are used in chat too.
- **Markets**: index and sector moves today, plus a lookup for any ticker with its price trend, moving averages, RSI, company facts, and recent English-language news.
- **Goals**: a Monte Carlo projection for one goal, with the chance of reaching it. Tick **Include saved portfolio** and edit the amount to count some or all of your portfolio.
- **Knowledge**: search the knowledge base, browse articles by category, and look up glossary terms.
