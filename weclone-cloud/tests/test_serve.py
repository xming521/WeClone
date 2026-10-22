from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient
from weclone_cloud.config import ApiSettings, CloudConfig, ServeSettings
from weclone_cloud.serve import create_app

from .conftest import completion


def build_config(*, require_key: bool = True) -> CloudConfig:
    return CloudConfig(
        api=ApiSettings(base_url="https://up.test/v1", model="base", api_key="up-key", timeout=5.0),
        serve=ServeSettings(
            model="ft:persona",
            system_prompt="你是 B",
            require_api_key=require_key,
            api_key="serve-key",
            temperature=0.3,
        ),
    )


def make_client(config: CloudConfig, *, captured: list[dict] | None = None) -> TestClient:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if captured is not None:
            captured.append(body)
        return httpx.Response(200, json=completion("好的", model=body["model"]))

    app = create_app(config, transport=httpx.MockTransport(handler))
    return TestClient(app)  # 用 with 进入以触发 lifespan（创建 AsyncClient）


def test_models_endpoint_requires_key():
    with make_client(build_config()) as client:
        assert client.get("/v1/models").status_code == 401
        response = client.get("/v1/models", headers={"Authorization": "Bearer serve-key"})
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "ft:persona"


def test_chat_injects_system_and_model():
    captured: list[dict] = []
    with make_client(build_config(), captured=captured) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer serve-key"},
            json={"messages": [{"role": "user", "content": "在吗"}]},
        )
    assert response.status_code == 200
    body = captured[0]
    assert body["model"] == "ft:persona"
    assert body["messages"][0] == {"role": "system", "content": "你是 B"}
    assert body["temperature"] == 0.3


def test_chat_replaces_client_system_by_default():
    captured: list[dict] = []
    with make_client(build_config(), captured=captured) as client:
        client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer serve-key"},
            json={
                "messages": [
                    {"role": "system", "content": "自定义"},
                    {"role": "user", "content": "hi"},
                ]
            },
        )
    system_messages = [m for m in captured[0]["messages"] if m["role"] == "system"]
    assert system_messages == [{"role": "system", "content": "你是 B"}]


def test_chat_if_missing_keeps_client_system():
    config = build_config()
    config.serve.system_mode = "if_missing"
    captured: list[dict] = []
    with make_client(config, captured=captured) as client:
        client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer serve-key"},
            json={
                "messages": [
                    {"role": "system", "content": "自定义"},
                    {"role": "user", "content": "hi"},
                ]
            },
        )
    system_messages = [m for m in captured[0]["messages"] if m["role"] == "system"]
    assert system_messages == [{"role": "system", "content": "自定义"}]


def test_chat_rejects_empty_messages():
    with make_client(build_config()) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer serve-key"},
            json={"messages": []},
        )
    assert response.status_code == 400


def test_open_mode_without_key():
    with make_client(build_config(require_key=False)) as client:
        assert client.get("/v1/models").status_code == 200


def test_index_page_renders():
    with make_client(build_config()) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "ft:persona" in response.text
