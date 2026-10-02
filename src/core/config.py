"""Application settings.

Secrets and the LLM provider choice come from environment variables (or ``.env``).
Everything else comes from ``config.yaml``. Environment variables take precedence
over YAML values, so a deployment can override a setting without editing files.

Use :func:`get_settings` for the process-wide cached instance, and
:func:`load_settings` when a fresh, independently configured instance is needed.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

ProviderName = Literal["openai", "anthropic"]
Tier = Literal["main", "fast"]

_DISABLED_VALUES = {"", "none", "off", "false", "disabled"}


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSpec(_Section):
    """A model name plus optional per-model overrides of the LLM defaults."""

    model: str = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)


class ProviderModels(_Section):
    main: ModelSpec
    fast: ModelSpec

    def for_tier(self, tier: Tier) -> ModelSpec:
        return self.main if tier == "main" else self.fast


class LLMConfig(_Section):
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=2048, gt=0)
    timeout_s: float = Field(default=60, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    providers: dict[ProviderName, ProviderModels]


class AppConfig(_Section):
    name: str = "Finnie"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"
    data_path: Path = Path("data/app/finnie.sqlite")  # per-browser saved data (git-ignored)
    disclaimer: str = (
        "Finnie provides educational information only, not financial, investment, tax, "
        "or legal advice. Consult a qualified professional before making financial decisions."
    )


class RAGConfig(_Section):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    knowledge_base_dir: Path = Path("data/knowledge_base")
    index_dir: Path = Path("data/vectorstore")
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    top_k: int = Field(default=4, gt=0)
    fetch_k: int = Field(default=40, gt=0)
    score_threshold: float = Field(default=0.40, ge=-1, le=1)
    mmr_lambda: float = Field(default=0.7, ge=0, le=1)
    max_chunks_per_article: int = Field(default=2, gt=0)

    @model_validator(mode="after")
    def _check_sizes(self) -> RAGConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("rag.chunk_overlap must be smaller than rag.chunk_size")
        if self.fetch_k < self.top_k:
            raise ValueError("rag.fetch_k must be at least rag.top_k")
        return self


class AlphaVantageConfig(_Section):
    requests_per_minute: int = Field(default=5, gt=0)
    daily_budget: int = Field(default=25, ge=0)


class BackoffConfig(_Section):
    max_attempts: int = Field(default=3, ge=1)
    initial_s: float = Field(default=1.0, ge=0)
    multiplier: float = Field(default=2.0, ge=1)
    max_s: float = Field(default=10.0, ge=0)


class MarketDataConfig(_Section):
    cache_path: Path = Path("data/cache/market.sqlite")
    quote_ttl_minutes: int = Field(default=30, gt=0)  # while the market is closed
    quote_live_ttl_seconds: int = Field(default=60, gt=0)  # while it's open
    news_ttl_minutes: int = Field(default=30, gt=0)
    history_ttl_hours: int = Field(default=12, gt=0)
    request_timeout_s: float = Field(default=10, gt=0)
    batch_threshold: int = Field(default=3, ge=1)
    alpha_vantage: AlphaVantageConfig = Field(default_factory=AlphaVantageConfig)
    backoff: BackoffConfig = Field(default_factory=BackoffConfig)


class WorkflowConfig(_Section):
    max_stages: int = Field(default=3, ge=1)
    max_agents_per_turn: int = Field(default=3, ge=1)
    agent_max_iterations: int = Field(default=4, ge=1)
    history_window: int = Field(default=20, ge=1)
    summarize_after: int = Field(default=30, ge=1)
    turn_timeout_s: float = Field(default=120, gt=0)
    router_min_confidence: float = Field(default=0.5, ge=0, le=1)


class MonteCarloConfig(_Section):
    simulations: int = Field(default=10_000, ge=100, le=200_000)
    t_degrees_of_freedom: float | None = Field(default=5, gt=2)
    inflation: float = Field(default=0.025, ge=-0.05, le=0.5)
    target_success_probability: float = Field(default=0.8, gt=0, lt=1)


class AnalyticsConfig(_Section):
    risk_free_rate: float = Field(default=0.042, ge=-0.05, le=0.5)  # fallback only
    trading_days_per_year: int = Field(default=252, gt=0)
    concentration_threshold: float = Field(default=0.20, gt=0, le=1)
    sector_concentration_threshold: float = Field(default=0.40, gt=0, le=1)
    high_expense_ratio: float = Field(default=0.005, ge=0)
    fee_growth_rate: float = Field(default=0.06, ge=-0.5, le=0.5)
    min_history_days: int = Field(default=60, ge=2)
    history_days: int = Field(default=252, ge=2, le=5000)
    benchmark: str = "SPY"
    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)


YAML_SECTIONS = frozenset({"app", "llm", "rag", "market_data", "workflow", "analytics"})


class Settings(BaseSettings):
    """All runtime settings. Secrets are ``SecretStr`` so they never appear in reprs or logs."""

    model_config = SettingsConfigDict(
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
        frozen=True,
    )

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    alpha_vantage_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None

    llm_provider: ProviderName = "openai"
    llm_fallback_provider: ProviderName | None = "anthropic"
    hf_hub_offline: bool = False  # HF_HUB_OFFLINE: load the embedding model from cache only

    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig
    rag: RAGConfig = Field(default_factory=RAGConfig)
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # YAML values arrive as init kwargs; give real environment variables priority over them.
        return env_settings, dotenv_settings, init_settings, file_secret_settings

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("llm_fallback_provider", mode="before")
    @classmethod
    def _normalize_fallback(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip().lower()
            return None if value in _DISABLED_VALUES else value
        return value

    @model_validator(mode="after")
    def _check_providers_configured(self) -> Settings:
        for name in (self.llm_provider, self.llm_fallback_provider):
            if name is not None and name not in self.llm.providers:
                raise ValueError(f"No models configured for provider '{name}' under llm.providers")
        return self

    @property
    def fallback_provider(self) -> ProviderName | None:
        """The fallback provider, or ``None`` when disabled or identical to the primary."""
        fallback = self.llm_fallback_provider
        return None if fallback == self.llm_provider else fallback

    def api_key_for(self, provider: ProviderName) -> SecretStr | None:
        return {"openai": self.openai_api_key, "anthropic": self.anthropic_api_key}[provider]

    def resolve_path(self, path: Path) -> Path:
        """Resolve a config path relative to the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    unknown = set(data) - YAML_SECTIONS
    if unknown:
        raise ConfigError(
            f"Unknown section(s) in {path}: {', '.join(sorted(unknown))}. "
            f"Expected: {', '.join(sorted(YAML_SECTIONS))}"
        )
    return data


def load_settings(config_path: Path | None = None, env_file: Path | None = None) -> Settings:
    """Build settings from YAML plus environment.

    Paths default to ``FINNIE_CONFIG`` / ``FINNIE_ENV_FILE`` when set, otherwise to
    ``config.yaml`` and ``.env`` in the project root. A missing ``.env`` is not an error.
    """
    config_path = Path(config_path or os.environ.get("FINNIE_CONFIG") or DEFAULT_CONFIG_PATH)
    env_file = Path(env_file or os.environ.get("FINNIE_ENV_FILE") or DEFAULT_ENV_FILE)
    data = _read_yaml(config_path)
    try:
        return Settings(_env_file=env_file, **data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration ({config_path}):\n{exc}") from exc


@cache
def get_settings() -> Settings:
    """Process-wide settings, loaded once. Call ``get_settings.cache_clear()`` to reload."""
    return load_settings()
