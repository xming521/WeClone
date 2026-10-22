from __future__ import annotations

import pytest
from weclone_cloud.config import ApiSettings, FinetuneSettings


def test_defaults_target_deepseek():
    settings = ApiSettings()
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.model == "deepseek-flash"
    assert settings.resolved_thinking == "disabled"


def test_thinking_auto_only_for_deepseek():
    other = ApiSettings(base_url="https://api.openai.com/v1", model="gpt-4o-mini", api_key="k")
    assert other.resolved_thinking == "omit"

    forced = ApiSettings(base_url="https://api.openai.com/v1", thinking="enabled", api_key="k")
    assert forced.resolved_thinking == "enabled"


def test_estimate_cost_with_prices():
    settings = ApiSettings(
        api_key="k",
        price_input_per_million=2.0,
        price_output_per_million=8.0,
    )
    cost = settings.estimate_cost({"prompt_tokens": 1_000_000, "completion_tokens": 500_000})
    assert cost == pytest.approx(6.0)
    assert settings.estimate_cost({}) is None


def test_estimate_cost_without_prices_returns_none():
    assert ApiSettings(api_key="k").estimate_cost({"prompt_tokens": 10}) is None


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("WECLONE_API_KEY", raising=False)
    settings = ApiSettings()
    with pytest.raises(Exception) as excinfo:
        settings.resolved_key
    assert "WECLONE_API_KEY" in str(excinfo.value)


def test_finetune_generic_has_no_job_requirements():
    settings = FinetuneSettings(provider="generic")
    assert settings.provider == "generic"
    assert settings.api_style == "method"
