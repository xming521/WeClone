from __future__ import annotations

import pytest
from weclone_cloud.config import ApiSettings, FinetuneSettings
from weclone_cloud.providers import API_PROVIDER_REGISTRY


def test_defaults_target_minimax():
    settings = ApiSettings()
    assert settings.provider == "minimax"
    assert settings.region == "global_en"
    assert settings.base_url == "https://api.minimax.io/v1"
    assert settings.model == "MiniMax-M3.1-Flash"
    assert settings.extra_body == {"reasoning_effort": "max"}
    assert settings.price_currency == "USD"


def test_minimax_registry_contains_current_models_and_regions():
    preset = API_PROVIDER_REGISTRY["minimax"]
    assert preset["default_model"] == "MiniMax-M3.1-Flash"
    assert set(preset["models"]) == {"MiniMax-M3.1-Flash", "MiniMax-M3"}
    assert preset["endpoints"] == {
        "global_en": "https://api.minimax.io/v1",
        "cn_zh": "https://api.minimaxi.com/v1",
    }
    assert preset["models"]["MiniMax-M3.1-Flash"]["context_window"] == 1_000_000
    assert preset["models"]["MiniMax-M3"]["pricing_tiers"]["priority_multiplier"] == 1.5


def test_minimax_china_region_uses_regional_endpoint():
    settings = ApiSettings(provider="minimax", region="cn_zh")
    assert settings.base_url == "https://api.minimaxi.com/v1"


def test_minimax_m3_does_not_send_reasoning_effort():
    settings = ApiSettings(provider="minimax", model="MiniMax-M3")
    assert settings.extra_body == {}


def test_explicit_endpoint_and_model_remain_custom():
    settings = ApiSettings(base_url="https://gateway.example/v1", model="example-model")
    assert settings.provider == "custom"
    assert settings.base_url == "https://gateway.example/v1"
    assert settings.model == "example-model"


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
    settings = ApiSettings(
        provider="custom",
        base_url="https://gateway.example/v1",
        model="example-model",
        api_key="k",
    )
    assert settings.estimate_cost({"prompt_tokens": 10}) is None


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
