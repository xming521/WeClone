import importlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from weclone.server import api_service, app, style_view
from weclone.utils import secure_storage


@pytest.mark.parametrize("options,expected", [
    ([], {"host": "127.0.0.1", "port": 5175, "run": None, "inference": False, "demo": None}),
    (["--host", "0.0.0.0", "--port", "5180", "--run", "run", "--inference"],
     {"host": "0.0.0.0", "port": 5180, "run": Path("run"), "inference": True, "demo": None}),
    (["--demo"], {"host": "127.0.0.1", "port": 5175, "run": None, "inference": False, "demo": "zh"}),
    (["--demo", "en", "--port", "5180"],
     {"host": "127.0.0.1", "port": 5180, "run": None, "inference": False, "demo": "en"}),
])
def test_server_cli_routes_options_without_loading_model_config(monkeypatch, tmp_path, options, expected):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run").mkdir()
    module = importlib.import_module("weclone.cli")
    monkeypatch.delenv("API_PORT", raising=False)
    monkeypatch.setattr(module, "_check_project_root", lambda: None)
    load = Mock(side_effect=AssertionError("CLI must defer configuration loading to the inference factory"))
    monkeypatch.setattr(module, "load_config", load)
    monkeypatch.setattr(module, "_check_versions", load)
    serve = Mock()
    monkeypatch.setattr(app, "serve", serve)
    result = CliRunner().invoke(module.cli, ["server", *options])
    assert result.exit_code == 0, result.output
    serve.assert_called_once_with(**expected)
    load.assert_not_called()
    assert "inference-server" not in module.cli.commands


def test_server_cli_rejects_demo_with_run(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run").mkdir()
    module = importlib.import_module("weclone.cli")
    monkeypatch.setattr(module, "_check_project_root", lambda: None)
    serve = Mock()
    monkeypatch.setattr(app, "serve", serve)
    result = CliRunner().invoke(module.cli, ["server", "--demo", "--run", "run"])
    assert result.exit_code == 2
    serve.assert_not_called()


@pytest.mark.parametrize("language", ["zh", "en"])
def test_demo_serves_test_profile_apart_from_real_storage(tmp_path, monkeypatch, language):
    settings = tmp_path / "settings.jsonc"
    real_state = tmp_path / "state"
    settings.write_text(json.dumps({"security_args": {"storage_mode": "encrypted", "state_dir": str(real_state)}}))
    monkeypatch.setenv("WECLONE_CONFIG_PATH", str(settings))
    secure_storage.configure()
    profile = json.loads((app.DEMO_DATA / app.DEMO_PROFILES[language]).read_text())
    with app.demo_run(language) as run:
        assert not secure_storage.is_encrypted_mode()
        with TestClient(app.create_app(run=run, static_dir=tmp_path / "no-static")) as client:
            client.headers["X-WeClone-Request"] = "1"
            assert client.get("/api/auth/status").json()["initialized"]
            assert client.post("/api/auth/login", json={"password": app.DEMO_PASSWORD}).status_code == 200
            assert len(client.get("/api/profile").json()["facts"]) == len(profile["facts"])
            assert client.get("/api/profile/intro").json()["message_total"] == profile["_demo"]["message_total"]
    assert not run.exists()
    assert not real_state.exists()
    assert secure_storage.is_encrypted_mode()


def write_profile(run):
    source = run / "profile_hierarchy" / "profile_hierarchy.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"domains":[],"dimensions":[],"facts":[],"sources":{}}')


def use_plaintext_storage(tmp_path, monkeypatch):
    settings = tmp_path / "settings.jsonc"
    settings.write_text(
        json.dumps({"security_args": {"storage_mode": "plaintext", "state_dir": str(tmp_path / "state")}})
    )
    monkeypatch.setenv("WECLONE_CONFIG_PATH", str(settings))
    secure_storage.configure()


def test_unified_routes_lifespan_auth_and_streaming(tmp_path, monkeypatch):
    events = []

    @asynccontextmanager
    async def lifespan(_app):
        events.append("start")
        yield
        events.append("stop")

    inference = FastAPI(lifespan=lifespan)

    def auth(authorization: str | None = Header(default=None)):
        if authorization != "Bearer fixture":
            raise HTTPException(401)

    @inference.get("/v1/models", dependencies=[Depends(auth)])
    def models():
        return {"data": [{"id": "fixture"}]}

    @inference.post("/v1/chat/completions", dependencies=[Depends(auth)])
    def chat():
        return StreamingResponse(iter(['data: {"content":"ok"}\n\n', "data: [DONE]\n\n"]),
                                 media_type="text/event-stream")

    factory = Mock(return_value=inference)
    monkeypatch.setattr(api_service, "create_inference_app", factory)
    use_plaintext_storage(tmp_path, monkeypatch)
    run = tmp_path / "run"
    write_profile(run)
    static = tmp_path / "web"
    static.mkdir()
    (static / "index.html").write_text("<h1>WeClone</h1>")
    options = {"run": run, "static_dir": static}
    with TestClient(app.create_app(**options)) as client:
        client.headers["X-WeClone-Request"] = "1"
        assert client.get("/api/profile").status_code == 401
        password = {"password": "server-test", "confirmation": "server-test"}
        assert client.post("/api/auth/setup", json=password).status_code == 200
        assert client.get("/").text == "<h1>WeClone</h1>"
        assert client.get("/api/profile").json()["facts"] == []
        assert client.get("/v1/models").status_code == 404
        cookies = dict(client.cookies)
    factory.assert_not_called()
    with TestClient(app.create_app(**options, inference=True)) as client:
        client.headers["X-WeClone-Request"] = "1"
        client.cookies.update(cookies)
        assert events == ["start"]
        assert client.get("/").status_code == 200
        assert client.get("/api/profile").status_code == 200
        assert client.get("/v1/models").status_code == 401
        headers = {"Authorization": "Bearer fixture"}
        assert client.get("/v1/models", headers=headers).json()["data"][0]["id"] == "fixture"
        response = client.post("/v1/chat/completions", headers=headers)
        assert "text/event-stream" in response.headers["content-type"]
        assert response.text.endswith("data: [DONE]\n\n")
        paths = client.get("/openapi.json").json()["paths"]
        assert "/api/profile" in paths and "/v1/chat/completions" in paths
    assert events == ["start", "stop"]
    factory.assert_called_once()


def test_style_comes_from_the_served_run(tmp_path, monkeypatch):
    use_plaintext_storage(tmp_path, monkeypatch)
    monkeypatch.setattr(style_view, "build_view", lambda run: {"run": str(run)})
    with_style, without_style = tmp_path / "with-style", tmp_path / "without-style"
    for run in (with_style, without_style):
        write_profile(run)
    (with_style / "style" / "person").mkdir(parents=True)
    (with_style / "style" / "manifest.json").write_text("{}")
    (with_style / "style" / "person" / "metrics.json").write_text("{}")
    password = {"password": "server-test", "confirmation": "server-test"}
    with TestClient(app.create_app(run=with_style, static_dir=tmp_path / "no-static")) as client:
        client.headers["X-WeClone-Request"] = "1"
        assert client.post("/api/auth/setup", json=password).status_code == 200
        assert client.get("/api/style").json() == {"run": str(with_style / "style")}
    with TestClient(app.create_app(run=without_style, static_dir=tmp_path / "no-static")) as client:
        client.headers["X-WeClone-Request"] = "1"
        assert client.post("/api/auth/login", json={"password": "server-test"}).status_code == 200
        assert client.get("/api/style").status_code == 404
