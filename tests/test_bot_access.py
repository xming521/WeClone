import base64
import json
import sqlite3
import sys
from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from weclone.server import api_service, app, bots
from weclone.server.profile_review import create_app
from weclone.utils import secure_storage

PROFILE = {"domains": [], "dimensions": [], "facts": [], "sources": {}}


def clone_model():
    """The /v1 routes of the inference app. In encrypted mode they need the key, as loading the model does."""
    router = APIRouter()

    def unlocked():
        if secure_storage.is_encrypted_mode():
            secure_storage.get_unlocked_key()

    @router.get("/v1/models")
    def models():
        unlocked()
        return {"data": [{"id": "clone"}]}

    @router.post("/v1/chat/completions")
    def chat(body: dict):
        unlocked()
        if not body.get("stream"):
            return {"choices": [{"message": {"role": "assistant", "content": "在呢"}}]}

        def stream():
            unlocked()
            yield 'data: {"content":"在呢"}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return router


def server(tmp_path, *, inference=True):
    source = tmp_path / "profile_hierarchy.json"
    if not secure_storage.file_exists(source):
        source.write_text(json.dumps(PROFILE))
    return create_app(
        tmp_path / "profile_review.sqlite3",
        source,
        tmp_path / "static",
        inference_router=clone_model() if inference else None,
    )


@pytest.fixture
def owner(tmp_path, storage):
    """The owner's logged-in browser, on plaintext storage."""
    storage()
    client = TestClient(server(tmp_path), headers={"X-WeClone-Request": "1"})
    password = {"password": "bot-test", "confirmation": "bot-test"}
    assert client.post("/api/auth/setup", json=password).status_code == 200
    return client


def new_token(owner):
    response = owner.post("/api/bots/token")
    assert response.status_code == 201
    return response.json()["token"]


class Bot:
    """A bot platform's OpenAI provider: the API key goes in the Authorization header, no cookie."""

    def __init__(self, app, token, client=None):
        self.client = client or TestClient(app)
        self.headers = {"Authorization": f"Bearer {token}"}

    def chat(self, stream=False, **headers):
        body = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "在吗"}], "stream": stream}
        return self.client.post("/v1/chat/completions", json=body, headers={**self.headers, **headers})


def test_bots_reach_v1_with_the_shared_token_alone(owner):
    view = owner.get("/api/bots").json()
    assert view == {
        "encrypted": False,
        "inference": True,
        "api_url": "http://testserver/v1",
        "token": None,
    }
    refused = Bot(owner.app, "wcbot_unknown").chat()
    assert refused.status_code == 401 and refused.headers["www-authenticate"] == "Bearer"

    token = new_token(owner)
    assert token.startswith("wcbot_")
    bot = Bot(owner.app, token)
    reply = bot.chat()
    assert reply.status_code == 200 and reply.json()["choices"][0]["message"]["content"] == "在呢"
    assert reply.headers["cache-control"] == "no-store"
    assert bot.chat(stream=True).text.endswith("data: [DONE]\n\n")
    assert bot.client.get("/v1/models", headers=bot.headers).json()["data"][0]["id"] == "clone"
    shown = owner.get("/api/bots").json()["token"]
    assert shown["created"] and shown["last_used"] and token not in json.dumps(shown)
    # A web page that reaches the port is turned away even with the token.
    assert bot.chat(Origin="http://127.0.0.1:5175").status_code == 403
    # The owner's tools, like test-model, keep calling with the session.
    assert (
        owner.post("/v1/chat/completions", json={}, headers={"Authorization": "Bearer sk-test"}).status_code
        == 200
    )

    renewed = new_token(owner)
    assert bot.chat().status_code == 401
    assert Bot(owner.app, renewed).chat().status_code == 200
    assert owner.delete("/api/bots/token").status_code == 200
    assert Bot(owner.app, renewed).chat().status_code == 401
    assert owner.get("/api/bots").json()["token"] is None


def test_bot_and_assistant_tokens_open_only_their_own_door(owner):
    bot_token = new_token(owner)
    assistant_token = owner.post("/api/assistant/token").json()["token"]
    assert Bot(owner.app, assistant_token).chat().status_code == 401
    ping = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    mcp = TestClient(owner.app).post("/mcp", json=ping, headers={"Authorization": f"Bearer {bot_token}"})
    assert mcp.status_code == 401


def test_the_page_says_when_the_server_has_no_model(tmp_path, storage):
    storage()
    app = server(tmp_path, inference=False)
    owner = TestClient(app, headers={"X-WeClone-Request": "1"})
    password = {"password": "bot-test", "confirmation": "bot-test"}
    assert owner.post("/api/auth/setup", json=password).status_code == 200
    assert owner.get("/api/bots").json()["inference"] is False
    assert Bot(app, new_token(owner)).chat().status_code == 404


def sealed_key(tmp_path):
    with sqlite3.connect(tmp_path / "state" / "bots.sqlite3") as db:
        return db.execute("SELECT sealed_key FROM access").fetchone()[0]


def test_encrypted_clone_answers_with_the_token_alone_across_restarts(tmp_path, storage):
    storage("encrypted")
    secure_storage.ensure_initialized("bot-test")
    secure_storage.write_json(tmp_path / "profile_hierarchy.json", PROFILE)
    data_key = bytes(secure_storage.get_unlocked_key())
    # From here on only the server holds the key.
    secure_storage.lock()
    app = server(tmp_path)
    owner = TestClient(app, headers={"X-WeClone-Request": "1"})

    def login(password="bot-test"):
        assert owner.post("/api/auth/login", json={"password": password}).status_code == 200

    login()
    token = new_token(owner)
    stored = (tmp_path / "state" / "bots.sqlite3").read_bytes()
    assert token.encode() not in stored and data_key not in stored
    assert base64.b64encode(data_key) not in stored
    # Nobody needs to stay logged in, streamed replies included.
    assert owner.post("/api/auth/logout").status_code == 200
    assert Bot(app, token).chat().status_code == 200
    assert Bot(app, token).chat(stream=True).text.endswith("data: [DONE]\n\n")
    # A restarted server has no session at all.
    restarted = server(tmp_path)
    assert Bot(restarted, token).chat().status_code == 200
    # A new password keeps the same data key, so the token keeps working.
    secure_storage.change_password("bot-test", "bot-new")
    assert Bot(restarted, token).chat().status_code == 200

    # A reset starts a new key the old token never had.
    secure_storage.reset_password("bot-reset")
    secure_storage.lock()
    stale = Bot(restarted, token).chat()
    assert stale.status_code == 403 and "http://testserver/#/bots" in stale.json()["detail"]
    owner = TestClient(restarted, headers={"X-WeClone-Request": "1"})
    login("bot-reset")
    renewed = new_token(owner)
    assert Bot(restarted, renewed).chat().status_code == 200
    assert owner.delete("/api/bots/token").status_code == 200
    assert Bot(restarted, renewed).chat().status_code == 401 and sealed_key(tmp_path) is None


def test_tokens_issued_at_once_leave_the_current_token_with_its_own_key(tmp_path, monkeypatch, storage):
    storage("encrypted")
    secure_storage.ensure_initialized("bot-test")
    key = secure_storage.get_unlocked_key()
    access = bots.BotAccess(tmp_path / "state" / bots.ACCESS_FILE)
    seal, issued, sealing_first = secure_storage.seal_key, [], [True]

    def sealing(key, token):
        # A second request issues while the first one seals.
        if sealing_first:
            sealing_first.clear()
            issued.append(access.issue(key))
        return seal(key, token)

    monkeypatch.setattr(secure_storage, "seal_key", sealing)
    issued.append(access.issue(key))
    current = next(item["token"] for item in issued if access.match(item["token"]))
    assert secure_storage.open_sealed_key(access.row()["sealed_key"], current) == key


def test_a_bot_calling_after_a_restart_loads_the_model_with_the_sealed_key(tmp_path, monkeypatch, storage):
    events = []

    class ChatModel:
        def __init__(self, config):
            secure_storage.get_unlocked_key()
            events.append("load")
            self.engine = SimpleNamespace(name="clone")

    def llamafactory_api(model):
        @asynccontextmanager
        async def lifespan(_app):
            events.append("start")
            yield

        api = FastAPI(lifespan=lifespan)
        api.include_router(clone_model())
        return api

    api_module, chat_module = ModuleType("llamafactory.api.app"), ModuleType("llamafactory.chat")
    api_module.create_app, chat_module.ChatModel = llamafactory_api, ChatModel
    monkeypatch.setitem(sys.modules, "llamafactory.api.app", api_module)
    monkeypatch.setitem(sys.modules, "llamafactory.chat", chat_module)
    monkeypatch.setattr(
        api_service,
        "load_base_config",
        lambda: SimpleNamespace(common_args=SimpleNamespace(adapter_name_or_path=None)),
    )
    monkeypatch.setattr(
        api_service, "create_config_by_arg_type", Mock(return_value=Mock(model_dump=lambda **_: {}))
    )
    storage("encrypted")
    secure_storage.ensure_initialized("bot-test")
    run = tmp_path / "run"
    secure_storage.write_json(run / "profile_hierarchy" / "profile_hierarchy.json", PROFILE)
    secure_storage.lock()

    def start():
        return TestClient(app.create_app(run=run, static_dir=tmp_path / "static", inference=True))

    with start() as client:
        owner = TestClient(client.app, headers={"X-WeClone-Request": "1"})
        assert owner.post("/api/auth/login", json={"password": "bot-test"}).status_code == 200
        token = new_token(owner)
    with start() as client:
        assert events == []
        # On the running server's event loop, where the model's lifespan is later closed.
        assert Bot(client.app, token, client).chat().status_code == 200
        assert events == ["load", "start"]
