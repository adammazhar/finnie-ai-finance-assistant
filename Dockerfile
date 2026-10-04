# Finnie: the Streamlit app (and, with a different command, the MCP server).
#
#   docker compose up --build          # http://localhost:8501
#
# The embedding model and the knowledge base search index are built into the image, so a
# container runs offline except for the LLM and market data APIs. Secrets are never baked
# in: they come from .env at run time (see docker-compose.yml).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

# Dependencies first, so code changes don't reinstall them. CPU-only PyTorch comes from
# the extra index named in requirements.txt.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml config.yaml ./
COPY .streamlit .streamlit
COPY src src
COPY scripts scripts
COPY data/knowledge_base data/knowledge_base
COPY data/reference data/reference
COPY data/sample_portfolios data/sample_portfolios

# Download the embedding model into the image and build the FAISS index, then switch
# Hugging Face to offline mode for good.
RUN HF_HUB_OFFLINE=0 python scripts/build_index.py
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Run as a non-root user. data/app (saved conversations) and data/cache (market data) are
# volumes in docker-compose.yml.
RUN useradd --create-home --uid 1000 finnie \
    && mkdir -p data/app data/cache \
    && chown -R finnie:finnie /app/data "$HF_HOME"
USER finnie

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

CMD ["python", "-m", "src.web_app", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
