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
pytest                                   # unit and integration tests; the network is blocked, so no API calls
python scripts/validate_kb.py            # knowledge base rules
python scripts/check_kb_links.py         # every source link (needs internet)
python scripts/eval_retrieval.py         # retrieval quality on the evaluation set
```
