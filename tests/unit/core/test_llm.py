import logging

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableWithFallbacks
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.core import llm as llm_module
from src.core.llm import (
    LLMConfigurationError,
    build_chat_model,
    describe_llm,
    get_llm,
)
from tests.fakes.llm import FakeChatModel

OPENAI_KEY = "sk-test-openai-000000000000"
ANTHROPIC_KEY = "sk-ant-test-000000000000"


def test_default_provider_is_openai_gpt4o(make_settings):
    s = make_settings(OPENAI_API_KEY=OPENAI_KEY, LLM_FALLBACK_PROVIDER="none")
    model = get_llm(settings=s)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o"
    assert model.temperature == 0.2
    assert model.max_retries == 3
    assert model.request_timeout == 60


def test_fast_tier_uses_mini_model(make_settings):
    s = make_settings(OPENAI_API_KEY=OPENAI_KEY, LLM_FALLBACK_PROVIDER="none")
    assert get_llm("fast", settings=s).model_name == "gpt-4o-mini"


def test_switch_to_anthropic_by_env_only(make_settings):
    s = make_settings(
        LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY=ANTHROPIC_KEY, LLM_FALLBACK_PROVIDER="none"
    )
    main, fast = get_llm(settings=s), get_llm("fast", settings=s)
    assert isinstance(main, ChatAnthropic)
    assert main.model == "claude-sonnet-5-5"
    assert fast.model == "claude-haiku-4-5-20251001"
    assert main.max_tokens == 2048


def test_uses_global_settings_when_none_passed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "none")
    assert isinstance(get_llm(), ChatOpenAI)


def test_fallback_wraps_primary_when_both_keys_present(make_settings):
    s = make_settings(OPENAI_API_KEY=OPENAI_KEY, ANTHROPIC_API_KEY=ANTHROPIC_KEY)
    model = get_llm(settings=s)
    assert isinstance(model, RunnableWithFallbacks)
    assert isinstance(model.runnable, ChatOpenAI)
    assert isinstance(model.fallbacks[0], ChatAnthropic)
    assert model.fallbacks[0].model == "claude-sonnet-5-5"


def test_fallback_skipped_with_warning_when_key_missing(make_settings, caplog):
    s = make_settings(OPENAI_API_KEY=OPENAI_KEY)
    with caplog.at_level(logging.WARNING):
        model = get_llm(settings=s)
    assert isinstance(model, ChatOpenAI)
    assert "ANTHROPIC_API_KEY" in caplog.text


def test_use_fallback_false(make_settings):
    s = make_settings(OPENAI_API_KEY=OPENAI_KEY, ANTHROPIC_API_KEY=ANTHROPIC_KEY)
    assert isinstance(get_llm(settings=s, use_fallback=False), ChatOpenAI)


def test_explicit_provider_equal_to_fallback_has_no_wrapper(make_settings):
    s = make_settings(OPENAI_API_KEY=OPENAI_KEY, ANTHROPIC_API_KEY=ANTHROPIC_KEY)
    assert isinstance(get_llm(provider="anthropic", settings=s), ChatAnthropic)


@pytest.mark.parametrize("key", [None, "   "])
def test_missing_primary_key_fails_fast(make_settings, key):
    env = {"LLM_FALLBACK_PROVIDER": "none"}
    if key is not None:
        env["OPENAI_API_KEY"] = key
    s = make_settings(**env)
    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY is not set"):
        get_llm(settings=s)


def test_unknown_provider(make_settings):
    with pytest.raises(LLMConfigurationError, match="Unknown LLM provider 'gemini'"):
        build_chat_model("gemini", settings=make_settings())


def test_per_model_overrides(tmp_path, monkeypatch):
    from src.core.config import load_settings

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "llm:\n  providers:\n"
        "    openai: {main: {model: m, temperature: 0.9, max_tokens: 99}, fast: {model: f}}\n"
        "    anthropic: {main: {model: a, temperature: 0}, fast: {model: b}}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    s = load_settings(cfg)
    oai = build_chat_model("openai", settings=s)
    assert (oai.temperature, oai.max_tokens) == (0.9, 99)
    assert build_chat_model("anthropic", settings=s).temperature == 0


class Answer(BaseModel):
    text: str


def test_runtime_fallback_on_primary_failure(make_settings, monkeypatch):
    """When the primary raises (after SDK retries), the fallback answers."""
    primary = FakeChatModel(responses=[RuntimeError("503 from primary")])
    backup = FakeChatModel(responses=["from backup"], structured_responses=[{"text": "sb"}])
    monkeypatch.setitem(llm_module._BUILDERS, "openai", lambda *a: primary)
    monkeypatch.setitem(llm_module._BUILDERS, "anthropic", lambda *a: backup)
    s = make_settings(OPENAI_API_KEY=OPENAI_KEY, ANTHROPIC_API_KEY=ANTHROPIC_KEY)
    model = get_llm(settings=s)

    reply = model.invoke("hi")
    assert isinstance(reply, AIMessage) and reply.content == "from backup"
    assert len(primary.calls) == 1 and len(backup.calls) == 1

    # structured output and tool binding also fall through to the backup
    primary.structured_responses = [RuntimeError("down")]
    assert model.with_structured_output(Answer).invoke("q") == Answer(text="sb")
    assert isinstance(model.bind_tools([]), RunnableWithFallbacks)


def test_describe_llm(make_settings):
    info = describe_llm(make_settings(OPENAI_API_KEY=OPENAI_KEY))
    assert (info.provider, info.main_model, info.fast_model) == ("openai", "gpt-4o", "gpt-4o-mini")
    assert info.primary_ready is True
    assert (info.fallback_provider, info.fallback_ready) == ("anthropic", False)


def test_describe_llm_nothing_configured(make_settings):
    info = describe_llm(make_settings(LLM_FALLBACK_PROVIDER="none"))
    assert info.primary_ready is False
    assert info.fallback_provider is None and info.fallback_ready is False


def test_describe_llm_uses_global_settings():
    assert describe_llm().provider == "openai"
