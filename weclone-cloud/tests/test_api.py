from __future__ import annotations

import json
import time

import httpx
import pytest
from weclone_cloud.api import ChatClient, CompletionCache, parse_json_object
from weclone_cloud.errors import ApiError

from .conftest import completion


def test_parse_json_variants():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('前缀 {"a": {"b": "}"}} 后缀') == {"a": {"b": "}"}}
    assert parse_json_object("没有 JSON") is None
    assert parse_json_object("") is None


def test_chat_parses_completion(api_settings):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=completion('{"score": 4, "reason": "有风格"}'))

    with ChatClient(api_settings, transport=httpx.MockTransport(handler)) as client:
        result = client.chat([{"role": "user", "content": "hi"}])

    assert result.data == {"score": 4, "reason": "有风格"}
    assert result.usage["total_tokens"] == 14
    assert seen[0].headers["authorization"] == "Bearer test-key"
    assert str(seen[0].url) == "https://example.test/v1/chat/completions"


def test_schema_rejected_falls_back_to_json_object(api_settings):
    api_settings.json_mode = "schema"
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body.get("response_format"))
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(
                400, json={"error": {"message": "response_format json_schema is unsupported"}}
            )
        return httpx.Response(200, json=completion('{"score": 3}'))

    with ChatClient(api_settings, transport=httpx.MockTransport(handler)) as client:
        result = client.chat([{"role": "user", "content": "hi"}], json_schema={"type": "object"})

    assert result.data == {"score": 3}
    assert bodies[0]["type"] == "json_schema"
    assert bodies[1]["type"] == "json_object"


def test_retries_on_429(api_settings, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=completion('{"score": 5}'))

    with ChatClient(api_settings, transport=httpx.MockTransport(handler)) as client:
        result = client.chat([{"role": "user", "content": "hi"}])

    assert calls["n"] == 2
    assert result.data == {"score": 5}


def test_non_retryable_error_raises(api_settings):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with ChatClient(api_settings, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ApiError) as excinfo:
            client.chat([{"role": "user", "content": "hi"}])

    assert excinfo.value.status == 401
    assert "bad key" in str(excinfo.value)


def test_deepseek_disables_thinking_by_default(api_settings):
    api_settings.base_url = "https://api.deepseek.com"
    api_settings.model = "deepseek-flash"
    api_settings.max_tokens = 400
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion('{"score": 5}'))

    with ChatClient(api_settings, transport=httpx.MockTransport(handler)) as client:
        client.chat([{"role": "user", "content": "hi"}], json_schema={"type": "object"})

    assert bodies[0]["thinking"] == {"type": "disabled"}
    assert bodies[0]["response_format"] == {"type": "json_object"}
    assert bodies[0]["max_tokens"]  # 防止 JSON 被截断


def test_empty_json_content_retries_without_response_format(api_settings):
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:  # DeepSeek 文档承认会偶发返回空 content
            return httpx.Response(200, json=completion(""))
        return httpx.Response(200, json=completion('{"score": 4, "reason": "ok"}'))

    with ChatClient(api_settings, transport=httpx.MockTransport(handler)) as client:
        result = client.chat([{"role": "user", "content": "hi"}], json_schema={"type": "object"})

    assert result.data == {"score": 4, "reason": "ok"}
    assert len(bodies) == 2
    assert "response_format" in bodies[0] and "response_format" not in bodies[1]


def test_extra_body_merges_into_payload(api_settings):
    api_settings.extra_body = {"reasoning_effort": "low"}
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion('{"score": 5}'))

    with ChatClient(api_settings, transport=httpx.MockTransport(handler)) as client:
        client.chat([{"role": "user", "content": "hi"}])

    assert bodies[0]["reasoning_effort"] == "low"


def test_usage_totals_accumulate(api_settings):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion('{"score": 5}'))

    with ChatClient(api_settings, transport=httpx.MockTransport(handler)) as client:
        client.chat([{"role": "user", "content": "a"}])
        client.chat([{"role": "user", "content": "b"}])

    assert client.usage_total["requests"] == 2
    assert client.usage_total["prompt_tokens"] == 20
    assert client.usage_total["completion_tokens"] == 8


def test_cache_hit_skips_http(api_settings, tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=completion('{"score": 4}'))

    cache = CompletionCache(tmp_path)
    transport = httpx.MockTransport(handler)
    with ChatClient(api_settings, cache=cache, transport=transport) as client:
        first = client.chat([{"role": "user", "content": "same"}])
        second = client.chat([{"role": "user", "content": "same"}])

    assert calls["n"] == 1
    assert first.cached is False
    assert second.cached is True
    assert second.data == {"score": 4}
