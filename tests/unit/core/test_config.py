from pathlib import Path

import pytest

from src.core.config import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    ConfigError,
    get_settings,
    load_settings,
)


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


MINIMAL_LLM = """
llm:
  providers:
    openai: {main: {model: m1}, fast: {model: m2}}
    anthropic: {main: {model: a1}, fast: {model: a2}}
"""


def test_real_config_loads_with_approved_defaults():
    s = load_settings(DEFAULT_CONFIG_PATH)
    assert s.llm_provider == "openai"
    assert s.llm_fallback_provider == "anthropic"
    assert s.llm.providers["openai"].main.model == "gpt-4o"
    assert s.llm.providers["openai"].fast.model == "gpt-4o-mini"
    assert s.llm.providers["anthropic"].main.model == "claude-sonnet-5-5"
    assert s.llm.providers["anthropic"].fast.model == "claude-haiku-4-5-20251001"
    assert s.market_data.quote_ttl_minutes == 30
    assert s.rag.embedding_model.endswith("all-MiniLM-L6-v2")
    assert "educational" in s.app.disclaimer


def test_env_vars_override_provider_and_are_normalized(make_settings):
    s = make_settings(LLM_PROVIDER=" Anthropic ", LLM_FALLBACK_PROVIDER="OPENAI")
    assert s.llm_provider == "anthropic"
    assert s.fallback_provider == "openai"


@pytest.mark.parametrize("value", ["none", "off", "NONE", "disabled", "false"])
def test_fallback_can_be_disabled(make_settings, value):
    assert make_settings(LLM_FALLBACK_PROVIDER=value).fallback_provider is None


def test_empty_env_values_fall_back_to_defaults(make_settings):
    s = make_settings(LLM_PROVIDER="", OPENAI_API_KEY="")
    assert s.llm_provider == "openai"
    assert s.openai_api_key is None


def test_fallback_same_as_primary_is_ignored(make_settings):
    s = make_settings(LLM_PROVIDER="anthropic", LLM_FALLBACK_PROVIDER="anthropic")
    assert s.fallback_provider is None


def test_unknown_provider_rejected(make_settings):
    with pytest.raises(ConfigError, match="llm_provider"):
        make_settings(LLM_PROVIDER="gemini")


def test_secrets_are_masked(make_settings):
    s = make_settings(OPENAI_API_KEY="sk-test-1234567890abcdef")
    assert "sk-test" not in repr(s)
    assert s.api_key_for("openai").get_secret_value() == "sk-test-1234567890abcdef"
    assert s.api_key_for("anthropic") is None


def test_dotenv_file_is_read(tmp_path):
    env = write(
        tmp_path / ".env", "ANTHROPIC_API_KEY=sk-ant-test-fromfile\nLLM_PROVIDER=anthropic\n"
    )
    s = load_settings(DEFAULT_CONFIG_PATH, env_file=env)
    assert s.llm_provider == "anthropic"
    assert s.anthropic_api_key.get_secret_value() == "sk-ant-test-fromfile"


def test_real_env_beats_dotenv_and_yaml(tmp_path, monkeypatch):
    env = write(tmp_path / ".env", "LLM_PROVIDER=anthropic\n")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert load_settings(DEFAULT_CONFIG_PATH, env_file=env).llm_provider == "openai"


def test_finnie_config_env_var_selects_file(tmp_path, monkeypatch):
    cfg = write(tmp_path / "c.yaml", MINIMAL_LLM + "workflow: {max_stages: 2}\n")
    monkeypatch.setenv("FINNIE_CONFIG", str(cfg))
    s = load_settings()
    assert s.workflow.max_stages == 2
    assert s.llm.providers["openai"].main.model == "m1"
    # sections omitted from YAML get defaults
    assert s.rag.top_k == 4


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "missing.yaml")


def test_invalid_yaml(tmp_path):
    with pytest.raises(ConfigError, match="Could not parse"):
        load_settings(write(tmp_path / "bad.yaml", "llm: [unclosed"))


def test_non_mapping_yaml(tmp_path):
    with pytest.raises(ConfigError, match="mapping"):
        load_settings(write(tmp_path / "list.yaml", "- a\n- b\n"))


def test_empty_yaml_reports_missing_llm_section(tmp_path):
    with pytest.raises(ConfigError, match="llm"):
        load_settings(write(tmp_path / "empty.yaml", ""))


def test_unknown_section_rejected(tmp_path):
    with pytest.raises(ConfigError, match="Unknown section"):
        load_settings(write(tmp_path / "c.yaml", MINIMAL_LLM + "rga: {}\n"))


def test_typo_inside_section_rejected(tmp_path):
    with pytest.raises(ConfigError, match="top_kk"):
        load_settings(write(tmp_path / "c.yaml", MINIMAL_LLM + "rag: {top_kk: 3}\n"))


def test_provider_without_models_rejected(tmp_path):
    text = "llm:\n  providers:\n    openai: {main: {model: m1}, fast: {model: m2}}\n"
    with pytest.raises(ConfigError, match="anthropic"):
        load_settings(write(tmp_path / "c.yaml", text))


def test_provider_without_models_ok_when_fallback_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "none")
    text = "llm:\n  providers:\n    openai: {main: {model: m1}, fast: {model: m2}}\n"
    assert load_settings(write(tmp_path / "c.yaml", text)).fallback_provider is None


@pytest.mark.parametrize(
    "section",
    [
        "rag: {chunk_size: 100, chunk_overlap: 100}",
        "rag: {top_k: 10, fetch_k: 5}",
        "llm: {temperature: 3, providers: {openai: {main: {model: m}, fast: {model: m}}}}",
        "market_data: {quote_ttl_minutes: 0}",
    ],
)
def test_invalid_values_rejected(tmp_path, section, monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "none")
    text = section if section.startswith("llm") else MINIMAL_LLM + section + "\n"
    with pytest.raises(ConfigError):
        load_settings(write(tmp_path / "c.yaml", text))


def test_tier_selection_and_path_resolution():
    s = load_settings(DEFAULT_CONFIG_PATH)
    models = s.llm.providers["openai"]
    assert models.for_tier("main").model == "gpt-4o"
    assert models.for_tier("fast").model == "gpt-4o-mini"
    assert s.resolve_path(Path("data/cache")) == PROJECT_ROOT / "data/cache"
    absolute = PROJECT_ROOT / "x"
    assert s.resolve_path(absolute) == absolute


def test_fallback_accepts_none_programmatically():
    s = load_settings(DEFAULT_CONFIG_PATH).model_copy()
    rebuilt = type(s)(llm=s.llm, llm_fallback_provider=None)
    assert rebuilt.fallback_provider is None
