"""Provider presets for cloud inference."""

from __future__ import annotations

from typing import Any

MINIMAX_PROVIDER = "minimax"
MINIMAX_DEFAULT_REGION = "global_en"
MINIMAX_DEFAULT_MODEL = "MiniMax-M3.1-Flash"

MINIMAX_MODELS: dict[str, dict[str, Any]] = {
    "MiniMax-M3.1-Flash": {
        "context_window": 1_000_000,
        "input_modalities": ("text", "image", "video"),
        "thinking": "adaptive",
        "thinking_always_on": True,
        "reasoning_effort": ("low", "medium", "high", "xhigh", "max"),
        "reasoning_effort_default": "max",
        "service_tier_supported": False,
        "pricing_usd_per_million_tokens": {
            "input": 0.15,
            "output": 0.60,
            "cache_read": 0.015,
            "cache_write": None,
        },
    },
    "MiniMax-M3": {
        "context_window": 1_000_000,
        "input_modalities": ("text", "image", "video"),
        "thinking": ("adaptive", "disabled"),
        "thinking_always_on": False,
        "reasoning_effort": (),
        "reasoning_effort_default": None,
        "reasoning_effort_note": "Effort fields are ignored by MiniMax-M3.",
        "service_tier_supported": True,
        "pricing_usd_per_million_tokens": {
            "input": 0.30,
            "output": 1.20,
            "cache_read": 0.06,
            "cache_write": None,
        },
        "pricing_tiers": {
            "selection_basis": "request_input_tokens",
            "input_token_threshold": "512k",
            "standard": (
                {
                    "input_token_condition": "<=512k",
                    "pricing_usd_per_million_tokens": {
                        "input": 0.30,
                        "output": 1.20,
                        "cache_read": 0.06,
                        "cache_write": None,
                    },
                },
                {
                    "input_token_condition": ">512k",
                    "pricing_usd_per_million_tokens": {
                        "input": 0.60,
                        "output": 2.40,
                        "cache_read": 0.12,
                        "cache_write": None,
                    },
                },
            ),
            "default_service_tier": "standard",
            "priority_multiplier": 1.5,
            "priority_multiplier_applies_to": ("input", "output", "cache_read"),
        },
    },
}

MINIMAX_ENDPOINTS = {
    "global_en": "https://api.minimax.io/v1",
    "cn_zh": "https://api.minimaxi.com/v1",
}

API_PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    MINIMAX_PROVIDER: {
        "default_model": MINIMAX_DEFAULT_MODEL,
        "models": MINIMAX_MODELS,
        "default_region": MINIMAX_DEFAULT_REGION,
        "endpoints": MINIMAX_ENDPOINTS,
    }
}
