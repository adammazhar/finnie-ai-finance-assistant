"""Shared fixtures. Every test runs isolated from the developer's real .env."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from src.core.config import DEFAULT_CONFIG_PATH, Settings, get_settings, load_settings

ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "TAVILY_API_KEY",
    "MCP_API_TOKEN",
    "LLM_PROVIDER",
    "LLM_FALLBACK_PROVIDER",
    "FINNIE_CONFIG",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Clear Finnie env vars and point .env at a file that doesn't exist."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FINNIE_ENV_FILE", str(tmp_path / "no.env"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def make_settings(monkeypatch: pytest.MonkeyPatch):
    """Build Settings from the real config.yaml with the given env vars."""

    def factory(**env: str) -> Settings:
        for name, value in env.items():
            monkeypatch.setenv(name.upper(), value)
        return load_settings(DEFAULT_CONFIG_PATH)

    return factory
