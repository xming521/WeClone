import json
import time
from functools import partial
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from weclone.core.inference.usage import UsageLimitReached
from weclone.data.agent import build_cost
from weclone.data.agent import profile_pipeline as pipeline
from weclone.server import profile_review, profile_summary
from weclone.server.profile_review import create_app
from weclone.utils import secure_storage

FAKE = Path(__file__).with_name("fake_build.py")


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
def web(tmp_path, monkeypatch, storage, owner_password):
    storage()
    monkeypatch.setattr(
        pipeline, "check", lambda *args, **kwargs: {"backend": {}, "chats": {}, "problems": []}
    )
    monkeypatch.setenv("FAKE_BUILD", "succeed")

    def start(**options):
        client = TestClient(create_app(static_dir=tmp_path / "no-static", build_script=FAKE, **options))
        client.headers["X-WeClone-Request"] = "1"
        if not secure_storage.is_initialized():
            assert client.post("/api/auth/setup", json=owner_password).status_code == 200
        else:
            assert (
                client.post("/api/auth/login", json={"password": owner_password["password"]}).status_code
                == 200
            )
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


def test_the_owner_deletes_versions_but_not_the_shown_one(web, tmp_path):
    old, shown = pipeline.RUNS_DIR / "20260101_000000_000000", pipeline.RUNS_DIR / "20260102_000000_000000"
    write_profile(old, "旧画像")
    write_profile(shown, "新画像")
    # A run linked in from elsewhere loses only its link.
    elsewhere = tmp_path / "experiment"
    write_profile(elsewhere, "实验画像")
    linked = pipeline.RUNS_DIR / "20251231_000000_000000"
    linked.symlink_to(elsewhere)
    client = web()
    assert client.put("/api/run", json={"name": old.name}).status_code == 200
    assert client.put("/api/run", json={"name": shown.name}).status_code == 200
    with profile_summary.GENERATING:
        assert client.delete(f"/api/runs/{old.name}").json()["code"] == "summary_generating"
    assert client.delete(f"/api/runs/{old.name}").json() == {"ok": True}
    assert client.delete(f"/api/runs/{linked.name}").json() == {"ok": True}
    assert not old.exists() and not linked.exists() and (elsewhere / "profile_hierarchy").is_dir()
    assert client.delete(f"/api/runs/{shown.name}").json()["code"] == "run_current"
    assert client.delete(f"/api/runs/{old.name}").json()["code"] == "run_not_found"
    assert client.delete("/api/runs/%2E%2E").json()["code"] == "run_not_found"
    assert [item["name"] for item in client.get("/api/runs").json()] == [shown.name]
    assert values(client) == ["新画像"]


def test_a_running_request_finishes_on_the_run_it_started_with(tmp_path, storage):
    storage()
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
    started = build(client)
    wait(client, "running")
    deadline = time.time() + 30
    while client.get("/api/build").json()["stages"][1]["state"] != "running":
        assert time.time() < deadline
        time.sleep(0.1)
    assert pipeline.build_running()
    second = client.post("/api/build", json={"purpose": "assistant", "consent": True})
    assert second.json()["code"] == "build_running"
    assert client.delete(f"/api/runs/{started['run']}").json()["code"] == "run_building"
    assert client.delete("/api/build").json() == {"ok": True}
    cancelled = wait(client, "cancelled")
    assert cancelled["stages"][1]["state"] == "cancelled"
    assert not pipeline.build_running()
    assert client.delete("/api/build").json()["code"] == "build_not_running"


def test_a_build_that_died_without_saying_shows_as_interrupted(web, monkeypatch):
    client = web()
    run = pipeline.RUNS_DIR / "20260101_000000_000000"
    run.mkdir(parents=True)
    pipeline.write_status(run, pipeline.new_status("clone", 999999, 999999))
    pipeline.record_build(run)
    status = client.get("/api/build").json()
    assert status["state"] == "interrupted" and status["stages"][-1]["key"] == "style"
    # It stays so while a later build runs.
    monkeypatch.setenv("FAKE_BUILD", "wait")
    started = build(client)
    wait(client, "running")
    states = {item["name"]: item["state"] for item in client.get("/api/runs").json()}
    assert states == {started["run"]: "running", run.name: "interrupted"}
    assert client.delete("/api/build").json() == {"ok": True}
    wait(client, "cancelled")


def test_a_server_started_on_a_run_keeps_it_over_a_build_that_ended_before(web):
    older, newer = pipeline.RUNS_DIR / "20260101_000000_000000", pipeline.RUNS_DIR / "20260102_000000_000000"
    write_profile(older, "旧画像")
    write_profile(newer, "新画像")
    pipeline.write_status(newer, {**pipeline.new_status("assistant"), "state": "succeeded"})
    pipeline.record_build(newer)
    client = web(source=older / "profile_hierarchy" / "profile_hierarchy.json")
    assert values(client) == ["旧画像"]


def test_a_build_needs_consent_and_a_passing_check(web, monkeypatch):
    client = web()
    response = client.post("/api/build", json={"purpose": "assistant", "consent": False})
    assert response.json()["code"] == "consent_required"
    problem = {"code": "chats_missing", "params": {}, "message": "没有聊天记录"}
    monkeypatch.setattr(
        pipeline, "check", lambda *args, **kwargs: {"backend": {}, "chats": {}, "problems": [problem]}
    )
    response = client.post("/api/build", json={"purpose": "assistant", "consent": True})
    assert response.status_code == 422
    assert response.json()["code"] == "build_check_failed" and response.json()["problems"] == [problem]
    assert client.get("/api/build").json() is None


def test_an_encrypted_build_writes_with_the_session_key(web, monkeypatch, storage):
    storage("encrypted")
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
    # Encrypted mode keeps only the exception class and the counts logs declare, in the status and in the log.
    failed = wait(client, "failed")
    assert failed["error"]["message"] == ""
    log = (pipeline.RUNS_DIR / failed["run"] / pipeline.LOG_FILE).read_text(encoding="utf-8")
    assert "private" not in log and "987654321" not in log and "uncaught error_type=RuntimeError" in log
    assert "event windows=7 error_type=none" in log and "event step=2 steps=" in log


def test_the_web_estimates_and_probes(web, monkeypatch):
    client = web()
    monkeypatch.setattr(build_cost, "estimate", lambda *args: {"requests": 3})
    assert client.get("/api/build/estimate?purpose=clone").json() == {"requests": 3}
    monkeypatch.setattr(build_cost, "probe", lambda *args: {"requests": 1})
    assert client.post("/api/build/probe", json={"purpose": "assistant"}).json() == {"requests": 1}

    def failing(*args):
        raise RuntimeError("Unauthorized: bad key")

    monkeypatch.setattr(build_cost, "probe", failing)
    response = client.post("/api/build/probe", json={"purpose": "assistant"})
    assert response.status_code == 502 and response.json()["reason"] == "Unauthorized: bad key"

    def limited(*args):
        raise UsageLimitReached("You've hit your usage limit.")

    monkeypatch.setattr(build_cost, "probe", limited)
    response = client.post("/api/build/probe", json={"purpose": "assistant"})
    assert (
        response.json()["code"] == "usage_limit"
        and response.json()["reason"] == "You've hit your usage limit."
    )


def test_the_demo_cannot_build_or_switch(tmp_path, storage):
    from weclone.server import app

    storage()
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
        assert client.delete("/api/runs/20260101_000000_000000").json()["code"] == "demo_runs_fixed"


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


def test_check_finds_missing_chats_and_api_settings(tmp_path, storage):
    settings = storage(
        extra={
            "agent_distill_args": {"llm_provider": "api"},
            "make_dataset_args": {"platform": "chat", "csv_folder": str(tmp_path / "none"), "base_url": "u"},
        },
    )
    found = pipeline.check(settings, "assistant")
    assert codes(found) == ["chats_missing", "api_config_missing"]
    assert found["problems"][1]["params"] == {"keys": ["llm_api_key", "model_name"]}
    assert [problem.get("backend") for problem in found["problems"]] == [None, True]
    assert found["chats"] == {"platform": "chat", "folder": str(tmp_path / "none"), "count": 0}


def test_check_reports_a_config_it_cannot_read(storage):
    settings = storage(extra={"make_dataset_args": {"platform": "nope"}})
    found = pipeline.check(settings, "assistant")
    assert codes(found)[0] == "config_invalid"
    assert found["problems"][0]["params"]["detail"]


def test_check_counts_chats_and_passes_a_complete_api_setup(storage, chats):
    endpoint = {"base_url": "u", "llm_api_key": "k", "model_name": "m"}
    settings = storage(
        extra={
            "agent_distill_args": {"llm_provider": "api"},
            "make_dataset_args": {"platform": "chat", "csv_folder": str(chats), **endpoint},
        },
    )
    found = pipeline.check(settings, "assistant", 64000)
    assert found == {
        "backend": {"provider": "api", "model": "m", "max_context_tokens": 64000},
        "chats": {"platform": "chat", "folder": str(chats), "count": 1},
        "problems": [],
    }
    lock = pipeline.build_lock()
    with lock:
        assert codes(pipeline.check(settings, "assistant", 64000)) == ["build_running"]
    # Without one given, a share of the model's window in the settings, or of the default.
    assert pipeline.check(settings, "assistant")["backend"]["max_context_tokens"] == 121600
    settings.write_text(
        settings.read_text().replace(
            '"llm_provider": "api"', '"llm_provider": "api", "context_window": 32000'
        )
    )
    assert pipeline.check(settings, "assistant")["backend"]["max_context_tokens"] == 30400


def test_check_finds_telegram_without_my_id(tmp_path, monkeypatch, storage, chats):
    settings = storage(
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
    # A folder of generic CSV chats is read without the platform's own settings.
    found = pipeline.check(settings, "assistant", 64000, chats)
    assert codes(found) == [] and found["chats"] == {"platform": "chat", "folder": str(chats), "count": 1}


@pytest.mark.parametrize(
    "login,models,expected",
    [
        (None, None, ["codex_missing"]),
        (1, None, ["codex_logged_out"]),
        (0, {"models": []}, []),
        (0, {"models": [{"slug": "m", "context_window": 1000, "effective_context_window_percent": 90}]}, []),
    ],
)
def test_check_finds_what_codex_lacks(tmp_path, monkeypatch, storage, chats, login, models, expected):
    command = fake_codex(tmp_path, login, models) if login is not None else str(tmp_path / "no-codex")
    settings = storage(
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
        # A model the catalog does not list gets the default window.
        window = 900 if models["models"] else 121600
        assert found["backend"] == {"provider": "codex_exec", "model": "m", "max_context_tokens": window}


def test_check_wants_the_style_parser_for_a_clone(monkeypatch, storage, chats):
    endpoint = {"base_url": "u", "llm_api_key": "k", "model_name": "m"}
    settings = storage(
        extra={
            "agent_distill_args": {"llm_provider": "api"},
            "make_dataset_args": {"platform": "chat", "csv_folder": str(chats), **endpoint},
        },
    )
    monkeypatch.setattr(pipeline, "find_spec", lambda name: None)
    assert codes(pipeline.check(settings, "clone", 64000)) == ["style_deps_missing"]
    assert codes(pipeline.check(settings, "assistant", 64000)) == []


def test_the_command_line_lists_every_problem_before_asking(monkeypatch, storage):
    storage()
    problems = [
        {"code": "chats_missing", "params": {}, "message": "dataset/csv 里没有聊天记录"},
        {"code": "codex_logged_out", "params": {}, "message": "Codex CLI 没有登录"},
    ]
    monkeypatch.setattr(
        pipeline, "check", lambda *args, **kwargs: {"backend": {}, "chats": {}, "problems": problems}
    )
    with pytest.raises(ValueError) as error:
        pipeline.run(with_style=False, output_dir=None, config_path=Path("settings.jsonc"))
    assert str(error.value) == "运行前检查没有通过：\n- dataset/csv 里没有聊天记录\n- Codex CLI 没有登录"
    assert not pipeline.RUNS_DIR.exists() or not any(pipeline.RUNS_DIR.iterdir())


@pytest.mark.parametrize("fails", [False, True])
def test_a_build_from_the_upload_clears_it_once_it_succeeds(monkeypatch, storage, tmp_path, fails):
    from weclone.data.agent import (
        chat_snapshot,
        distill_event,
        distill_profile,
        llm_profile_hierarchy,
        organize_memories,
    )
    from weclone.server import profile_summary

    codex = {
        "model": "m",
        "effort": "low",
        "batch_size": 1,
        "timeout": 30,
        "command": "codex",
        "sandbox": "read-only",
    }
    settings = storage(extra={"codex_exec_args": codex})
    (pipeline.UPLOAD_DIR / "chat").mkdir(parents=True)
    (pipeline.UPLOAD_DIR / "chat" / "chat_1_1.csv").write_text("x")
    found = {"backend": {"max_context_tokens": 64000}, "chats": {}, "problems": []}
    monkeypatch.setattr(pipeline, "check", lambda *args, **kwargs: found)
    monkeypatch.setattr(pipeline, "load_chat_args", lambda *args: None)
    monkeypatch.setattr(chat_snapshot, "build_snapshot", lambda people, chats: None)
    monkeypatch.setattr(distill_profile, "iter_chat_files", lambda people: [people / "chat.json"])
    monkeypatch.setattr(build_cost, "count", lambda *args: ({}, {}))
    monkeypatch.setattr(build_cost, "chats_of_snapshot", lambda people: [])
    monkeypatch.setattr(build_cost, "describe", lambda *args: None)
    for module in (distill_profile, distill_event):
        monkeypatch.setattr(module, "main", lambda **kwargs: None)

    def organize(argv):
        if fails:
            raise RuntimeError("3 LLM tasks failed")

    def hierarchy(argv):
        output = Path(argv[argv.index("--output-dir") + 1])
        output.mkdir(parents=True)
        (output / "profile_hierarchy.json").write_text("{}")

    monkeypatch.setattr(organize_memories, "main", organize)
    monkeypatch.setattr(llm_profile_hierarchy, "main", hierarchy)
    monkeypatch.setattr(profile_summary, "build_summaries", lambda directory: "")
    build = partial(
        pipeline.run,
        with_style=False,
        output_dir=tmp_path / "run",
        config_path=settings,
        confirmed=True,
        chats_folder=pipeline.UPLOAD_DIR,
    )
    if fails:
        # A failed build is built again from the same upload.
        with pytest.raises(RuntimeError):
            build()
        assert pipeline.count_csv_chats(pipeline.UPLOAD_DIR) == 1
    else:
        build()
        assert not pipeline.UPLOAD_DIR.exists()
