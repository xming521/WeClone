from __future__ import annotations

import json

import httpx
import pytest
from weclone_cloud.config import FinetuneSettings
from weclone_cloud.finetune import FineTuneClient, JobState, load_state, save_state


def build_settings(**overrides) -> FinetuneSettings:
    base = dict(
        base_url="https://ft.test/v1",
        api_key="ft-key",
        base_model="base-model",
        suffix="persona",
        epochs=2,
        api_style="method",
        poll_interval=0.01,
        timeout=5.0,
    )
    base.update(overrides)
    return FinetuneSettings(**base)


def test_upload_and_create_method_style(tmp_path):
    seen: list[dict] = []
    upload_payload: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/files"):
            raw = request.content
            assert b'name="purpose"' in raw
            upload_payload["purpose"] = b"fine-tune" in raw
            return httpx.Response(200, json={"id": "file-1", "object": "file"})
        if request.url.path.endswith("/fine_tuning/jobs"):
            body = json.loads(request.content)
            seen.append(body)
            return httpx.Response(200, json={"id": "ftjob-1", "status": "queued"})
        raise AssertionError(f"unexpected {request.url}")

    train = tmp_path / "train.jsonl"
    train.write_text('{"messages":[]}\n', encoding="utf-8")

    with FineTuneClient(build_settings(), transport=httpx.MockTransport(handler)) as client:
        file_id = client.upload(train)
        job = client.create(file_id, epochs=2, suffix="persona")

    assert file_id == "file-1"
    assert job["id"] == "ftjob-1"
    assert upload_payload["purpose"] is True
    body = seen[0]
    assert body["model"] == "base-model"
    assert body["training_file"] == "file-1"
    assert body["method"]["type"] == "supervised"
    assert body["method"]["supervised"]["hyperparameters"]["n_epochs"] == 2
    assert "hyperparameters" not in body


def test_create_legacy_style(tmp_path):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/fine_tuning/jobs"):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "ftjob-2", "status": "queued"})
        return httpx.Response(200, json={"id": "file-1"})

    with FineTuneClient(build_settings(api_style="legacy"), transport=httpx.MockTransport(handler)) as client:
        client.create("file-1")

    assert seen[0]["hyperparameters"]["n_epochs"] == 2
    assert "method" not in seen[0]


def test_wait_returns_succeeded_job(tmp_path):
    states = iter(
        [
            {"id": "ftjob-3", "status": "running"},
            {"id": "ftjob-3", "status": "succeeded", "fine_tuned_model": "ft:gpt-x"},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(states))

    with FineTuneClient(build_settings(), transport=httpx.MockTransport(handler)) as client:
        job = client.wait("ftjob-3", interval=0)

    assert job["status"] == "succeeded"
    assert job["fine_tuned_model"] == "ft:gpt-x"


def test_wait_raises_on_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "ftjob-4", "status": "failed", "error": {"message": "boom"}})

    with FineTuneClient(build_settings(), transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(Exception) as excinfo:
            client.wait("ftjob-4", interval=0)

    assert "failed" in str(excinfo.value)


def test_state_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    state = JobState(job_id="ftjob-1", status="running", model="")
    save_state(path, state)
    loaded = load_state(path)
    assert loaded.job_id == "ftjob-1"
    assert loaded.status == "running"
