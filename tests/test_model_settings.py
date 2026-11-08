from types import SimpleNamespace

import pyjson5
import pytest
from fastapi.testclient import TestClient

import weclone.core.inference as inference
from weclone.core.inference import LLMResponse
from weclone.data.agent import profile_pipeline
from weclone.server.profile_review import create_app

HEADERS = {"X-WeClone-Request": "1"}
KEY = "sk-fixture-0123456789"


@pytest.fixture
def web(tmp_path, storage):
    settings = storage()
    security = settings.read_text()[1:-1]
    settings.write_text(
        "{\n"
        f"    {security},\n"
        "    // which backend builds profiles\n"
        '    "agent_distill_args": {\n'
        '        "llm_provider": "codex_exec", // codex_exec or api\n'
        '        "max_samples_per_window": 8\n'
        "    },\n"
        '    "codex_exec_args": {"model": "gpt", "effort": "medium", "batch_size": 30, "timeout": 300},\n'
        '    "make_dataset_args": {\n'
        '        "platform": "chat",\n'
        '        "base_url": "https://example.com/v1", // also used to clean SFT data\n'
        f'        "llm_api_key": "{KEY}",\n'
        '        "model_name": "m"\n'
        "    }\n"
        "}\n"
    )
    source = tmp_path / "run" / "profile_hierarchy" / "profile_hierarchy.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"domains":[],"dimensions":[],"facts":[],"sources":{}}')
    client = TestClient(create_app(None, source, tmp_path / "web"))
    body = {"password": "old", "confirmation": "old"}
    assert client.post("/api/auth/setup", json=body, headers=HEADERS).status_code == 200
    client.settings = settings
    return client


def form(saved, **api):
    """The page's form: what was read, with the fields given changed."""
    return {key: saved[key] for key in ("version", "provider", "codex")} | {"api": saved["api"] | api}


def test_settings_read_as_saved(web):
    saved = web.get("/api/settings/model").json()
    assert saved["provider"] == "codex_exec"
    assert saved["codex"] == {"model": "gpt", "effort": "medium", "batch_size": 30, "timeout": 300}
    assert saved["api"] == {
        "base_url": "https://example.com/v1",
        "key": KEY,
        "model": "m",
        "context_window": None,
        "reasoning_effort": None,
        "extra_body": None,
        "structured_output": None,
        "batch_size": 10,
        "timeout": 120,
    }


def test_saving_changes_only_what_changed(web):
    saved = web.get("/api/settings/model").json()
    before = web.settings.read_text()
    body = form(
        saved,
        context_window=64000,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "disabled"}},
        structured_output="json_object",
    )
    body["provider"] = "api"
    response = web.put("/api/settings/model", json=body, headers=HEADERS)
    assert response.status_code == 200
    text = web.settings.read_text()
    for comment in (
        "// which backend builds profiles",
        "// codex_exec or api",
        "// also used to clean SFT data",
    ):
        assert comment in text
    data = pyjson5.loads(text)
    assert data["agent_distill_args"] == {
        "llm_provider": "api",
        "max_samples_per_window": 8,
        "context_window": 64000,
        "reasoning_effort": "high",
        "extra_body": {"thinking": {"type": "disabled"}},
        "structured_output": "json_object",
    }
    # The key as it was and the default batch size and timeout are not written out.
    assert data["make_dataset_args"]["llm_api_key"] == KEY
    # Four new keys, extra_body over five lines; nothing else moved.
    assert len(text.splitlines()) == len(before.splitlines()) + 8
    # An emptied field is taken out again.
    again = form(response.json(), reasoning_effort="", extra_body=None, key="sk-new")
    assert web.put("/api/settings/model", json=again, headers=HEADERS).status_code == 200
    data = pyjson5.loads(web.settings.read_text())
    assert (
        "reasoning_effort" not in data["agent_distill_args"]
        and "extra_body" not in data["agent_distill_args"]
    )
    assert data["make_dataset_args"]["llm_api_key"] == "sk-new"


def test_a_file_changed_elsewhere_is_not_overwritten(web):
    saved = web.get("/api/settings/model").json()
    web.settings.write_text(web.settings.read_text().replace('"m"', '"other"'))
    response = web.put("/api/settings/model", json=form(saved, model="x"), headers=HEADERS)
    assert response.status_code == 409 and response.json()["code"] == "settings_changed"
    assert pyjson5.loads(web.settings.read_text())["make_dataset_args"]["model_name"] == "other"


def test_the_backend_in_use_needs_its_fields(web):
    saved = web.get("/api/settings/model").json()
    body = form(saved, model="")
    assert web.put("/api/settings/model", json=body, headers=HEADERS).status_code == 200
    body = form(web.get("/api/settings/model").json(), model="")
    body["provider"] = "api"
    response = web.put("/api/settings/model", json=body, headers=HEADERS)
    assert response.json() | {"detail": None} == {
        "code": "model_settings_required",
        "field": "model",
        "detail": None,
    }
    no_key = form(web.get("/api/settings/model").json(), key="")
    no_key["provider"] = "api"
    assert web.put("/api/settings/model", json=no_key, headers=HEADERS).json()["field"] == "key"
    reserved = form(saved, extra_body={"model": "other"})
    assert web.put("/api/settings/model", json=reserved, headers=HEADERS).status_code == 422


def test_testing_the_api_redacts_the_key_and_finds_the_format(web, monkeypatch):
    sent = {}
    formats = []
    answers = []
    listing = None

    class Client:
        def __init__(self, **options):
            sent.update(options)
            self.client = self

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        @property
        def models(self):
            return self

        def list(self, timeout):
            if listing is None:
                raise RuntimeError("401 no model list")
            return [SimpleNamespace(to_dict=lambda entry=entry: entry) for entry in listing]

        def chat(self, prompt, **kwargs):
            formats.append(self.structured_output)
            return answers.pop(0)

    monkeypatch.setattr(inference, "OpenAICompatibleClient", Client)
    saved = web.get("/api/settings/model").json()
    body = form(saved, reasoning_effort="low")
    del body["version"]
    body["provider"] = "api"
    refused = {"http_status": 401}
    answers.append(LLMResponse(ok=False, error=f"401 Incorrect API key provided: {KEY}", metadata=refused))
    result = web.post("/api/settings/model/test", json=body, headers=HEADERS).json()
    assert sent["api_key"] == KEY and sent["reasoning_effort"] == "low"
    assert result["ok"] is False and result["models"] is None and result["structured_output"] is None
    assert KEY not in result["error"] and "Incorrect API key" in result["error"]
    # Only a 400 moves on to the next format.
    assert formats == ["json_schema"]
    answers.extend([LLMResponse(ok=False, metadata={"http_status": 400}), LLMResponse(ok=True, text="{}")])
    result = web.post("/api/settings/model/test", json=body, headers=HEADERS).json()
    assert result["ok"] and result["structured_output"] == "json_object"
    assert formats == ["json_schema", "json_schema", "json_object"]
    # The model's window comes from the endpoint's list when it states one.
    listing = [{"id": "m", "object": "model", "context_window": 131072}, {"id": "other", "context_length": 8}]
    answers.append(LLMResponse(ok=True, text="{}"))
    result = web.post("/api/settings/model/test", json=body, headers=HEADERS).json()
    assert result["models"] == ["m", "other"] and result["context_window"] == 131072


def test_testing_without_a_model_lists_the_models(web, monkeypatch):
    listing = [{"id": "b"}, {"id": "a"}]

    class Client:
        def __init__(self, **options):
            self.client = self

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        @property
        def models(self):
            return self

        def list(self, timeout):
            if not listing:
                raise RuntimeError(f"401 Incorrect API key provided: {KEY}")
            return [SimpleNamespace(to_dict=lambda entry=entry: entry) for entry in listing]

    monkeypatch.setattr(inference, "OpenAICompatibleClient", Client)
    body = form(web.get("/api/settings/model").json(), model="")
    del body["version"]
    body["provider"] = "api"
    result = web.post("/api/settings/model/test", json=body, headers=HEADERS).json()
    assert result["ok"] and result["models"] == ["a", "b"] and result["structured_output"] is None
    listing.clear()
    result = web.post("/api/settings/model/test", json=body, headers=HEADERS).json()
    assert not result["ok"] and result["models"] is None and KEY not in result["error"]


def test_testing_codex_checks_the_cli(web, monkeypatch):
    monkeypatch.setattr(
        profile_pipeline,
        "codex_problems",
        lambda command: [{"code": "codex_missing", "params": {"command": command}}],
    )
    saved = web.get("/api/settings/model").json()
    body = form(saved)
    del body["version"]
    result = web.post("/api/settings/model/test", json=body, headers=HEADERS).json()
    assert result == {"ok": False, "problem": {"code": "codex_missing", "params": {"command": "codex"}}}


def test_the_demo_shows_no_model_settings(web, tmp_path):
    demo = TestClient(
        create_app(
            None,
            tmp_path / "run" / "profile_hierarchy" / "profile_hierarchy.json",
            tmp_path / "web",
            demo_password="old",
        )
    )
    assert demo.post("/api/auth/login", json={"password": "old"}, headers=HEADERS).status_code == 200
    assert demo.get("/api/settings/model").json()["code"] == "demo_settings_fixed"
