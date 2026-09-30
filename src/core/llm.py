"""LLM provider factory.

The provider is chosen with the ``LLM_PROVIDER`` environment variable and model names
come from ``config.yaml``, so switching between OpenAI and Anthropic needs no code
changes. Transient errors (rate limits, 5xx, timeouts) are retried by the provider SDKs
with exponential backoff (``llm.max_retries``). If retries are exhausted and
``LLM_FALLBACK_PROVIDER`` is configured with a valid key, the call is re-run on the
fallback provider.

Adding a provider means adding one builder function to ``_BUILDERS``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableWithFallbacks
from pydantic import SecretStr

from src.core.config import LLMConfig, ModelSpec, ProviderName, Settings, Tier, get_settings

logger = logging.getLogger(__name__)

ChatModel = BaseChatModel | RunnableWithFallbacks
"""What :func:`get_llm` returns. Both support ``invoke``, ``stream``, ``bind_tools``
and ``with_structured_output``."""

API_KEY_ENV_VARS: dict[ProviderName, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class LLMConfigurationError(RuntimeError):
    """Raised when the selected provider cannot be built (unknown provider or missing key)."""


def _build_openai(spec: ModelSpec, cfg: LLMConfig, api_key: SecretStr) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=spec.model,
        api_key=api_key,
        temperature=spec.temperature if spec.temperature is not None else cfg.temperature,
        max_tokens=spec.max_tokens or cfg.max_tokens,
        timeout=cfg.timeout_s,
        max_retries=cfg.max_retries,
    )


def _build_anthropic(spec: ModelSpec, cfg: LLMConfig, api_key: SecretStr) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=spec.model,
        api_key=api_key,
        temperature=spec.temperature if spec.temperature is not None else cfg.temperature,
        max_tokens=spec.max_tokens or cfg.max_tokens,
        timeout=cfg.timeout_s,
        max_retries=cfg.max_retries,
    )


Builder = Callable[[ModelSpec, LLMConfig, SecretStr], BaseChatModel]

_BUILDERS: dict[str, Builder] = {
    "openai": _build_openai,
    "anthropic": _build_anthropic,
}


def build_chat_model(
    provider: str, tier: Tier = "main", settings: Settings | None = None
) -> BaseChatModel:
    """Build a single provider's chat model, without any fallback."""
    settings = settings or get_settings()
    if provider not in _BUILDERS:
        raise LLMConfigurationError(
            f"Unknown LLM provider '{provider}'. Supported: {', '.join(sorted(_BUILDERS))}"
        )
    name: ProviderName = provider  # type: ignore[assignment]
    api_key = settings.api_key_for(name)
    if api_key is None or not api_key.get_secret_value().strip():
        raise LLMConfigurationError(
            f"{API_KEY_ENV_VARS[name]} is not set. Add it to .env, or set LLM_PROVIDER "
            f"to a provider whose key is configured."
        )
    spec = settings.llm.providers[name].for_tier(tier)
    return _BUILDERS[provider](spec, settings.llm, api_key)


def get_llm(
    tier: Tier = "main",
    *,
    provider: str | None = None,
    settings: Settings | None = None,
    use_fallback: bool = True,
) -> ChatModel:
    """Return the chat model for ``tier`` ("main" for answers, "fast" for routing/checks).

    The primary provider must be usable; a missing key raises :class:`LLMConfigurationError`.
    A fallback provider that can't be built is skipped with a warning, so a missing
    fallback key never blocks the app.
    """
    settings = settings or get_settings()
    primary_name = provider or settings.llm_provider
    primary = build_chat_model(primary_name, tier, settings)

    fallback_name = settings.fallback_provider
    if not use_fallback or fallback_name is None or fallback_name == primary_name:
        return primary
    try:
        fallback = build_chat_model(fallback_name, tier, settings)
    except LLMConfigurationError as exc:
        logger.warning("LLM fallback disabled: %s", exc)
        return primary
    return primary.with_fallbacks([fallback])


@dataclass(frozen=True)
class LLMInfo:
    """Summary of the active LLM setup, for the UI sidebar and startup checks."""

    provider: ProviderName
    main_model: str
    fast_model: str
    primary_ready: bool
    fallback_provider: ProviderName | None
    fallback_ready: bool


def describe_llm(settings: Settings | None = None) -> LLMInfo:
    settings = settings or get_settings()

    def ready(name: ProviderName | None) -> bool:
        key = settings.api_key_for(name) if name else None
        return bool(key and key.get_secret_value().strip())

    models = settings.llm.providers[settings.llm_provider]
    fallback = settings.fallback_provider
    return LLMInfo(
        provider=settings.llm_provider,
        main_model=models.main.model,
        fast_model=models.fast.model,
        primary_ready=ready(settings.llm_provider),
        fallback_provider=fallback,
        fallback_ready=ready(fallback),
    )
