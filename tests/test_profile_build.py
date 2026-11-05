import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from weclone.data.agent import profile_pipeline as pipeline
from weclone.server import profile_review
from weclone.server.profile_review import create_app
from weclone.utils import secure_storage

FAKE = Path(__file__).with_name("fake_build.py")
PASSWORD = {"password": "build-test", "confirmation": "build-test"}


def configure(tmp_path, monkeypatch, mode="plaintext", extra=None):
    # One file per mode: settings are cached by path and modification time.
    settings = tmp_path / f"settings.{mode}.jsonc"
    settings.write_text(
        json.dumps(
            {"security_args": {"storage_mode": mode, "state_dir": str(tmp_path / "state")}, **(extra or {})}
        )
    )
    monkeypatch.setenv("WECLONE_CONFIG_PATH", str(settings))
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path / "runs")
    secure_storage.configure()
    return settings


def write_profile(run, value):
    secure_storage.write_json(
        run / "profile_hierarchy" / "profile_hierarchy.json",
        {
            "domains": [],
            "dimensions": [
                {
                    "dim": 8,
                    "name": "偏好",
                    "groups": [{"name": "阅读", "items": [{"attr": "偏好", "fact_indices": [0]}]}],
                }
            ],
            "facts": [{"dim": 8, "attr": "偏好", "value": value, "source_ids": []}],
            "sources": {},
        },
    )


@pytest.fixture
def web(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "check", lambda *args: {"backend": {}, "chats": {}, "problems": []})
    monkeypatch.setenv("FAKE_BUILD", "succeed")

    def start(**options):
        client = TestClient(create_app(static_dir=tmp_path / "no-static", build_script=FAKE, **options))
        client.headers["X-WeClone-Request"] = "1"
        if not secure_storage.is_initialized():
            assert client.post("/api/auth/setup", json=PASSWORD).status_code == 200
        else:
            assert client.post("/api/auth/login", json={"password": PASSWORD["password"]}).status_code == 200
        return client

    return start


def build(client, purpose="assistant"):
    response = client.post("/api/build", json={"purpose": purpose, "consent": True})
    assert response.status_code == 202, response.text
    return response.json()


def wait(client, *states):
    deadline = time.time() + 30
    while time.time() < deadline:
        status = client.get("/api/build").json()
        if status["state"] in states:
            return status
        time.sleep(0.1)
    raise AssertionError(f"build stayed {status['state']}")


def values(client):
    return [fact["value"] for fact in client.get("/api/profile").json()["facts"]]


def test_a_web_build_opens_its_profile_when_nothing_was_shown(web):
    client = web()
    assert client.get("/api/profile").json()["code"] == "profile_missing"
    assert client.get("/api/run").json()["name"] is None
    assert client.get("/api/build").json() is None
    started = build(client)
    assert started["state"] == "running" and started["purpose"] == "assistant"
    assert [stage["key"] for stage in started["stages"]][-1] == "summaries"
    done = wait(client, "succeeded")
    assert "pid" not in done and done["chats"] == 1
    assert done["stages"][1]["done"] == 2 and done["stages"][1]["total"] == 2
    assert values(client) == ["新画像"]
    assert client.get("/api/build").json()["current"] is True
    [listed] = client.get("/api/runs").json()
    assert listed["name"] == done["run"] and listed["current"] and listed["state"] == "succeeded"
    assert listed["started_at"] == done["started_at"]
    assert (pipeline.RUNS_DIR / done["run"] / pipeline.LOG_FILE).is_file()


def test_reviews_keep_the_shown_run_until_the_owner_switches(web, tmp_path, monkeypatch):
    old = pipeline.RUNS_DIR / "20260101_000000_000000"
    write_profile(old, "旧画像")
    client = web()
    fact = client.get("/api/profile").json()["facts"][0]
    review = {"items": [{"id": fact["id"], "version": fact["version"]}], "status": "approved"}
    assert client.post("/api/reviews/batch", json=review).status_code == 200
    build(client)
    new = wait(client, "succeeded")
    assert new["current"] is False and values(client) == ["旧画像"]
    listed = {item["name"]: item for item in client.get("/api/runs").json()}
    assert listed[old.name]["reviewed_at"] and listed[new["run"]]["reviewed_at"] is None
    assert client.put("/api/run", json={"name": new["run"]}).json() == {"name": new["run"]}
    assert values(client) == ["新画像"]
    # The choice outlives a restart, and a run reviewed before stays as it was.
    assert values(web()) == ["新画像"]
    restarted = web()
    assert restarted.put("/api/run", json={"name": old.name}).status_code == 200
    assert restarted.get("/api/profile").json()["facts"][0]["status"] == "approved"
    assert restarted.put("/api/run", json={"name": "../x"}).json()["code"] == "run_not_found"


def test_a_running_request_finishes_on_the_run_it_started_with(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch)
    first, second = pipeline.RUNS_DIR / "1", pipeline.RUNS_DIR / "2"
    write_profile(first, "一")
    write_profile(second, "二")
    from weclone.server.auth import AuthStore

    runs = profile_review.Runs(AuthStore(tmp_path / "state" / "web.sqlite3"))
    store, _ = runs.current()
    assert store.source.parent.parent == second
    runs.switch(first)
    assert store.profile()["facts"][0]["value"] == "二"
    assert runs.store().profile()["facts"][0]["value"] == "一"


def test_a_failed_build_keeps_the_error_and_the_shown_run(web, monkeypatch):
    write_profile(pipeline.RUNS_DIR / "20260101_000000_000000", "旧画像")
    client = web()
    monkeypatch.setenv("FAKE_BUILD", "fail")
    build(client)
    failed = wait(client, "failed")
    assert failed["error"] == {
        "step": 2,
        "key": "distill_profile",
        "type": "RuntimeError",
        "message": "fake failure with private text",
    }
    assert failed["stages"][1]["state"] == "failed" and failed["stages"][2]["state"] == "pending"
    assert values(client) == ["旧画像"]
    states = [item["state"] for item in client.get("/api/runs").json()]
    assert states == ["failed", "succeeded"]


def test_one_build_at_a_time_and_cancel(web, monkeypatch):
    client = web()
    monkeypatch.setenv("FAKE_BUILD", "wait")
    build(client)
    wait(client, "running")
    deadline = time.time() + 30
    while client.get("/api/build").json()["stages"][1]["state"] != "running":
        assert time.time() < deadline
        time.sleep(0.1)
    assert pipeline.build_running()
    second = client.post("/api/build", json={"purpose": "assistant", "consent": True})
    assert second.json()["code"] == "build_running"
    assert client.delete("/api/build").json() == {"ok": True}
    cancelled = wait(client, "cancelled")
    assert cancelled["stages"][1]["state"] == "cancelled"
    assert not pipeline.build_running()
    assert client.delete("/api/build").json()["code"] == "build_not_running"


def test_a_build_that_died_without_saying_shows_as_interrupted(web):
    client = web()
    run = pipeline.RUNS_DIR / "20260101_000000_000000"
    run.mkdir(parents=True)
    pipeline.write_status(run, pipeline.new_status("clone", 999999, 999999))
    pipeline.record_build(run)
    status = client.get("/api/build").json()
    assert status["state"] == "interrupted" and status["stages"][-1]["key"] == "style"


def test_a_build_needs_consent_and_a_passing_check(web, monkeypatch):
    client = web()
    response = client.post("/api/build", json={"purpose": "assistant", "consent": False})
    assert response.json()["code"] == "consent_required"
    problem = {"code": "chats_missing", "params": {}, "message": "没有聊天记录"}
    monkeypatch.setattr(pipeline, "check", lambda *args: {"backend": {}, "chats": {}, "problems": [problem]})
    response = client.post("/api/build", json={"purpose": "assistant", "consent": True})
    assert response.status_code == 422
    assert response.json()["code"] == "build_check_failed" and response.json()["problems"] == [problem]
    assert client.get("/api/build").json() is None


def test_an_encrypted_build_writes_with_the_session_key(web, tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch, "encrypted")
    client = web()
    build(client)
    done = wait(client, "succeeded")
    source = pipeline.RUNS_DIR / done["run"] / "profile_hierarchy" / "profile_hierarchy.json"
    assert not source.exists() and secure_storage.encrypted_path(source).read_bytes().startswith(
        secure_storage.MAGIC
    )
    assert values(client) == ["新画像"]
    monkeypatch.setenv("FAKE_BUILD", "fail")
    build(client)
    # Encrypted mode keeps only the exception class, as persistent logs do.
    assert wait(client, "failed")["error"]["message"] == ""


def test_the_demo_cannot_build_or_switch(tmp_path, monkeypatch):
    from weclone.server import app

    configure(tmp_path, monkeypatch)
    write_profile(pipeline.RUNS_DIR / "20260101_000000_000000", "真实画像")
    with app.demo_run("zh") as run:
        client = TestClient(
            app.create_app(run=run, static_dir=tmp_path / "no-static", demo_password=app.DEMO_PASSWORD)
        )
        client.headers["X-WeClone-Request"] = "1"
        assert client.post("/api/auth/login", json={"password": app.DEMO_PASSWORD}).status_code == 200
        assert client.get("/api/build/check").json()["code"] == "demo_build_disabled"
        response = client.post("/api/build", json={"purpose": "assistant", "consent": True})
        assert response.json()["code"] == "demo_build_disabled"
        assert [item["name"] for item in client.get("/api/runs").json()] == ["run"]
        assert (
            client.put("/api/run", json={"name": "20260101_000000_000000"}).json()["code"]
            == "demo_runs_fixed"
        )


def fake_codex(tmp_path, login_code, models):
    command = tmp_path / "codex"
    command.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = login ]; then exit {login_code}; fi\n'
        f"if [ \"$1\" = debug ]; then echo '{json.dumps(models)}'; exit 0; fi\n"
        'if [ "$1" = -c ] || [ "$1" = exec ] || [ "$1" = app-server ]; then exit 1; fi\n'
        "exit 0\n"
    )
    command.chmod(0o755)
    return str(command)


def codes(found):
    return [problem["code"] for problem in found["problems"]]


@pytest.fixture
def chats(tmp_path):
    folder = tmp_path / "csv"
    (folder / "friend").mkdir(parents=True)
    (folder / "friend" / "chat_0.csv").write_text("x")
    return folder


def test_check_finds_missing_chats_and_api_settings(tmp_path, monkeypatch):
    settings = configure(
        tmp_path,
        monkeypatch,
        extra={
            "agent_distill_args": {"llm_provider": "api"},
            "make_dataset_args": {"platform": "chat", "csv_folder": str(tmp_path / "none"), "base_url": "u"},
        },
    )
    found = pipeline.check(settings, "assistant")
    assert codes(found) == ["chats_missing", "api_config_missing", "max_context_required"]
    assert found["problems"][1]["params"] == {"keys": ["llm_api_key", "model_name"]}
    assert found["chats"] == {"platform": "chat", "count": 0}


def test_check_reports_a_config_it_cannot_read(tmp_path, monkeypatch):
    settings = configure(tmp_path, monkeypatch, extra={"make_dataset_args": {"platform": "nope"}})
    found = pipeline.check(settings, "assistant")
    assert codes(found)[0] == "config_invalid"
    assert found["problems"][0]["params"]["detail"]


def test_check_counts_chats_and_passes_a_complete_api_setup(tmp_path, monkeypatch, chats):
    endpoint = {"base_url": "u", "llm_api_key": "k", "model_name": "m"}
    settings = configure(
        tmp_path,
        monkeypatch,
        extra={
            "agent_distill_args": {"llm_provider": "api"},
            "make_dataset_args": {"platform": "chat", "csv_folder": str(chats), **endpoint},
        },
    )
    found = pipeline.check(settings, "assistant", 64000)
    assert found == {
        "backend": {"provider": "api", "model": "m", "max_context_tokens": 64000},
        "chats": {"platform": "chat", "count": 1},
        "problems": [],
    }
    lock = pipeline.build_lock()
    with lock:
        assert codes(pipeline.check(settings, "assistant", 64000)) == ["build_running"]


def test_check_finds_telegram_without_my_id(tmp_path, monkeypatch, chats):
    settings = configure(
        tmp_path,
        monkeypatch,
        extra={
            "agent_distill_args": {"llm_provider": "api"},
            "make_dataset_args": {
                "platform": "telegram",
                "base_url": "u",
                "llm_api_key": "k",
                "model_name": "m",
            },
        },
    )
    export = tmp_path / "telegram" / "ChatExport_1"
    export.mkdir(parents=True)
    (export / "result.json").write_text("{}")
    monkeypatch.setattr(pipeline, "TELEGRAM_DIR", tmp_path / "telegram")
    found = pipeline.check(settings, "assistant", 64000)
    assert codes(found) == ["telegram_id_missing"] and found["chats"]["count"] == 1


@pytest.mark.parametrize(
    "login,models,expected",
    [
        (None, None, ["codex_missing"]),
        (1, None, ["codex_logged_out"]),
        (0, {"models": []}, ["context_unknown"]),
        (0, {"models": [{"slug": "m", "context_window": 1000, "effective_context_window_percent": 90}]}, []),
    ],
)
def test_check_finds_what_codex_lacks(tmp_path, monkeypatch, chats, login, models, expected):
    command = fake_codex(tmp_path, login, models) if login is not None else str(tmp_path / "no-codex")
    settings = configure(
        tmp_path,
        monkeypatch,
        extra={
            "codex_exec_args": {"model": "m", "effort": "low", "command": command, "sandbox": "read-only"},
            "make_dataset_args": {"platform": "chat", "csv_folder": str(chats)},
        },
    )
    from weclone.data.agent import organize_memories

    async def no_config(command):
        return {}

    monkeypatch.setattr(organize_memories, "codex_config", no_config)
    found = pipeline.check(settings, "assistant")
    assert codes(found) == expected
    if not expected:
        assert found["backend"] == {"provider": "codex_exec", "model": "m", "max_context_tokens": 900}


def test_check_wants_the_style_parser_for_a_clone(tmp_path, monkeypatch, chats):
    endpoint = {"base_url": "u", "llm_api_key": "k", "model_name": "m"}
    settings = configure(
        tmp_path,
        monkeypatch,
        extra={
            "agent_distill_args": {"llm_provider": "api"},
            "make_dataset_args": {"platform": "chat", "csv_folder": str(chats), **endpoint},
        },
    )
    monkeypatch.setattr(pipeline, "find_spec", lambda name: None)
    assert codes(pipeline.check(settings, "clone", 64000)) == ["style_deps_missing"]
    assert codes(pipeline.check(settings, "assistant", 64000)) == []


def test_the_command_line_lists_every_problem_before_asking(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch)
    problems = [
        {"code": "chats_missing", "params": {}, "message": "dataset/csv 里没有聊天记录"},
        {"code": "codex_logged_out", "params": {}, "message": "Codex CLI 没有登录"},
    ]
    monkeypatch.setattr(pipeline, "check", lambda *args: {"backend": {}, "chats": {}, "problems": problems})
    with pytest.raises(ValueError) as error:
        pipeline.run(with_style=False, output_dir=None, config_path=Path("settings.jsonc"))
    assert str(error.value) == "运行前检查没有通过：\n- dataset/csv 里没有聊天记录\n- Codex CLI 没有登录"
    assert not pipeline.RUNS_DIR.exists() or not any(pipeline.RUNS_DIR.iterdir())
