from __future__ import annotations

import json
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """让测试不依赖外部 shell 里的密钥，避免 CI/本机环境污染断言。"""
    for name in ("WECLONE_API_KEY", "OPENAI_API_KEY", "WECLONE_SERVE_KEY", "WECLONE_CLOUD_CONFIG"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def api_settings():
    from weclone_cloud.config import ApiSettings

    return ApiSettings(
        base_url="https://example.test/v1",
        model="test-model",
        api_key="test-key",
        json_mode="object",
        max_retries=3,
        timeout=5.0,
    )


def completion(content: str, *, model: str = "test-model") -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }


def record(index: int = 0, *, score: int | None = 4, messages: list[dict] | None = None) -> dict:
    payload = {
        "id": str(index),
        "time": "2024-05-01T10:00:00",
        "messages": messages
        or [
            {"role": "user", "content": f"你最近在忙什么（{index}）"},
            {"role": "assistant", "content": f"在改论文，导师催得紧（{index}）"},
        ],
        "images": [],
        "system": "请你扮演一名人类",
        "chat_with": "peer",
    }
    if score is not None:
        payload["score"] = score
    return payload


@pytest.fixture()
def records() -> list[dict]:
    return [record(i) for i in range(3)]


def dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
