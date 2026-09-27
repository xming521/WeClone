import importlib
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from weclone.server import api_service, app


@pytest.mark.parametrize("options,expected", [
    ([], {"host": "127.0.0.1", "port": 5175, "database": None, "source": None, "inference": False}),
    (["--host", "0.0.0.0", "--port", "5180", "--database", "review.db", "--source", "input.json", "--inference"],
     {"host": "0.0.0.0", "port": 5180, "database": Path("review.db"), "source": Path("input.json"), "inference": True}),
])
def test_server_cli_routes_options_without_loading_model_config(monkeypatch, options, expected):
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
    source = tmp_path / "input.json"
    source.write_text('{"dimensions":[],"facts":[],"sources":{}}')
    static = tmp_path / "web"
    static.mkdir()
    (static / "index.html").write_text("<h1>WeClone</h1>")
    options = {"database": tmp_path / "review.db", "source": source, "static_dir": static}
    with TestClient(app.create_app(**options)) as client:
        assert client.get("/").text == "<h1>WeClone</h1>"
        assert client.get("/api/profile").json()["facts"] == []
        assert client.get("/v1/models").status_code == 404
    factory.assert_not_called()
    with TestClient(app.create_app(**options, inference=True)) as client:
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
