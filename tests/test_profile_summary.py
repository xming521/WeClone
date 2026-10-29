import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from weclone.data.agent.llm_cache import LLMCache
from weclone.data.agent.llm_profile_hierarchy import dimension_view
from weclone.prompts.profile_summary import SUMMARY_RULES, SUMMARY_SCHEMA

SHORT, LONG = SUMMARY_RULES["short"], SUMMARY_RULES["long"]
from weclone.server import profile_summary
from weclone.server.profile_review import ReviewStore, create_app
from weclone.utils import secure_storage


def fact(dim, attr, value, sensitivity, sids, *, scope="current", support=1):
    return {
        "dim": dim,
        "attr": attr,
        "value": value,
        "sensitivity": sensitivity,
        "time_scope": scope,
        "support_count": support,
        "source_ids": sids,
    }


FACTS = [
    fact(1, "在读身份", "计算机博士生", "public", ["s1", "s2"], support=3),
    fact(3, "职业目标", "想去高校  任教", "normal", ["s3"]),
    fact(5, "答辩安排", "六月答辩", "normal", ["s3"], scope="dated", support=4),
    fact(1, "在读身份", "硕士生", "public", ["s1"], scope="superseded", support=5),
    fact(4, "科研补助", "每月三千元", "private", ["s2"], support=2),
    fact(8, "研究偏好", "喜欢做实验", "normal", ["s4"]),
    fact(8, "口味偏好", "爱吃辣", "public", ["s4"], support=2),
    fact(8, "约会偏好", "喜欢看电影", "private", ["s5"], support=2),
]
TIMES = {
    "s1": "2023-09-01T10:00:00+08:00",
    "s2": "2024-05-20T10:00:00+08:00",
    "s3": "2024-02-11 09:00:00",
    "s4": "2024-07-03T10:00:00+08:00",
    "s5": "2024-08-03T10:00:00+08:00",
}
STUDY = [
    "在读身份：计算机博士生（人口学与基本身份，3 次，2023-09~2024-05）",
    "研究偏好：喜欢做实验（兴趣与偏好，1 次，2024-07）",
    "职业目标：想去高校 任教（长期需求、动机与目标，1 次，2024-02）",
]
# The full summary also reads plans and private facts. Only the superseded identity is left out.
FULL_STUDY = [
    "答辩安排：六月答辩（重要计划与承诺，4 次，2024-02）",
    STUDY[0],
    "科研补助：每月三千元（资源禀赋与生活条件，2 次，2024-05）",
    *STUDY[1:],
]


def hierarchy():
    domains = [
        {
            "name": "学业",
            "description": "读博",
            "topics": [{"name": "读博", "fact_indices": [0, 1, 2, 3, 4, 5]}],
        },
        {"name": "饮食", "description": "口味", "topics": [{"name": "口味", "fact_indices": [6]}]},
        {"name": "感情", "description": "约会", "topics": [{"name": "约会", "fact_indices": [7]}]},
    ]
    return {
        "domains": domains,
        "dimensions": dimension_view(FACTS, domains),
        "facts": FACTS,
        "sources": {sid: {"id": sid, "content": f"来源{sid}", "sample_time": t} for sid, t in TIMES.items()},
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    settings = tmp_path / "settings.jsonc"
    settings.write_text(
        json.dumps({"security_args": {"storage_mode": "plaintext", "state_dir": str(tmp_path / "state")}})
    )
    monkeypatch.setenv("WECLONE_CONFIG_PATH", str(settings))
    secure_storage.configure()
    source = tmp_path / "profile_hierarchy.json"
    source.write_text(json.dumps(hierarchy(), ensure_ascii=False), encoding="utf-8")
    database = tmp_path / "profile_review.sqlite3"
    client = TestClient(create_app(database, source, tmp_path / "static"))
    client.headers["X-WeClone-Request"] = "1"
    password = {"password": "summary-test", "confirmation": "summary-test"}
    assert client.post("/api/auth/setup", json=password).status_code == 200
    prompts = []

    class Runner:
        def __init__(self, output_dir):
            # A checkpoint directory for this generation only.
            assert output_dir.is_dir() and not output_dir.is_relative_to(tmp_path)

        def run(self, tasks, validate):
            answers = []
            for task in tasks:
                assert task["stage"] == "domain_summary" and task["schema"] is SUMMARY_SCHEMA
                prompts.append(task["prompt"])
                answers.append({"title": f"标题{len(prompts)}", "summary": f" 总结\n{len(prompts)} "})
                validate(answers[-1], task)
            # Like LLMTasks, every task is validated again once all answers are back.
            return [validate(answer, task) for answer, task in zip(answers, tasks)]

        def close(self):
            pass

    monkeypatch.setattr(profile_summary, "summary_runner", Runner)
    return client, ReviewStore(database, source), prompts


def generate(client, level="normal"):
    response = client.post("/api/summaries/generate", json={"level": level})
    assert response.status_code == 200
    # The background thread holds the lock until its summaries are stored.
    with profile_summary.GENERATING:
        pass
    return response.json()


def view(client, level="normal", **params):
    response = client.get("/api/summaries", params={"level": level, **params})
    assert response.status_code == 200
    return response.json()


def summaries(client, level="normal"):
    return {item["domain"]: item["summary"] for item in view(client, level)["domains"]}


def review(client, domains, status, level="normal"):
    rows = summaries(client, level=level)
    items = [{"domain": domain, "version": rows[domain]["version"]} for domain in domains]
    return client.post("/api/summaries/review", json={"level": level, "items": items, "status": status})


def reject_fact(client, value):
    target = next(row for row in client.get("/api/profile").json()["facts"] if row["value"] == value)
    response = client.post(
        "/api/reviews/batch",
        json={
            "items": [{"id": target["id"], "version": target["version"]}],
            "status": "rejected",
        },
    )
    assert response.status_code == 200


def test_inputs_keep_identity_goals_conditions_and_preferences_up_to_the_level(setup):
    _, store, _ = setup
    normal = {item["domain"]: item for item in profile_summary.domain_inputs(store.profile(), "normal")}
    assert list(normal) == ["学业", "饮食", "感情"]
    assert normal["学业"]["lines"] == STUDY
    assert normal["学业"]["facts"] == 3 and normal["学业"]["as_of"] == "2024-07"
    assert normal["感情"]["facts"] == 0 and normal["感情"]["lines"] == [] and normal["感情"]["as_of"] == ""
    public = {item["domain"]: item for item in profile_summary.domain_inputs(store.profile(), "public")}
    assert public["学业"]["lines"] == STUDY[:1] and public["学业"]["as_of"] == "2024-05"
    assert public["学业"]["sha256"] != normal["学业"]["sha256"]
    full = {item["domain"]: item for item in profile_summary.domain_inputs(store.profile(), "full")}
    assert full["学业"]["lines"] == FULL_STUDY and full["学业"]["facts"] == 5
    assert full["感情"]["lines"] == ["约会偏好：喜欢看电影（兴趣与偏好，2 次，2024-08）"]


def test_generate_approve_export_and_go_stale_when_facts_change(setup):
    client, _, prompts = setup
    first = view(client)
    assert first["text"] == "" and all(item["summary"] is None for item in first["domains"])
    assert generate(client) == {"generating": 2}
    assert len(prompts) == 2 and prompts[0] == SHORT + "\n".join(STUDY)
    rows = summaries(client)
    assert rows["感情"] is None
    assert rows["学业"] == {
        "title": "标题1",
        "summary": "总结 1",
        "origin": "generated",
        "status": "pending",
        "version": 1,
        "stale": False,
    }
    assert view(client)["text"] == ""

    assert review(client, ["学业"], "approved").status_code == 200
    text = view(client)["text"]
    assert text == f"{profile_summary.EXPORT_HEADER}\n\n- 标题1（截至 2024-07）：总结 1"
    assert client.get("/api/summaries", params={"level": "private"}).status_code == 422

    reject_fact(client, "喜欢做实验")
    assert summaries(client)["学业"]["stale"] is True
    assert view(client)["text"] == ""
    assert review(client, ["学业"], "approved").status_code == 409
    stale = summaries(client)["学业"]
    edit = {
        "level": "normal",
        "domain": "学业",
        "title": "学业",
        "summary": "改过",
        "version": stale["version"],
    }
    assert client.patch("/api/summaries", json=edit).status_code == 409

    assert generate(client) == {"generating": 1}
    assert prompts[-1] == SHORT + "\n".join([STUDY[0], STUDY[2]])
    assert summaries(client)["学业"] == {
        "title": "标题3",
        "summary": "总结 3",
        "origin": "generated",
        "status": "pending",
        "version": 3,
        "stale": False,
    }
    assert generate(client) == {"generating": 0}
    assert len(prompts) == 3


def test_summaries_go_stale_when_their_rules_change(setup, monkeypatch):
    client, _, prompts = setup
    generate(client)
    assert review(client, ["学业"], "approved").status_code == 200
    rules = SHORT.replace("一到两句话", "一句话")
    monkeypatch.setitem(SUMMARY_RULES, "short", rules)
    assert summaries(client)["学业"]["stale"] is True
    assert view(client)["text"] == ""
    assert generate(client) == {"generating": 2}
    assert prompts[-2] == rules + "\n".join(STUDY)


def test_edit_approves_and_versions_guard_changes(setup):
    client, _, _ = setup
    generate(client, "public")
    row = summaries(client, level="public")["饮食"]
    edit = {
        "level": "public",
        "domain": "饮食",
        "title": row["title"],
        "summary": row["summary"],
        "version": row["version"],
    }
    assert client.patch("/api/summaries", json=edit).json() == row | {"stale": False}
    changed = client.patch(
        "/api/summaries", json=edit | {"summary": "  爱吃辣，\n常吃火锅 ", "approve": True}
    ).json()
    assert changed == {
        "title": "标题2",
        "summary": "爱吃辣， 常吃火锅",
        "origin": "edited",
        "status": "approved",
        "version": 2,
        "stale": False,
    }
    assert client.patch("/api/summaries", json=edit | {"summary": "旧页面"}).status_code == 409
    assert client.patch("/api/summaries", json=edit | {"version": 2, "title": "长" * 13}).status_code == 422
    assert (
        client.patch("/api/summaries", json=edit | {"version": 2, "summary": "长" * 151}).status_code == 422
    )
    assert client.patch("/api/summaries", json=edit | {"domain": "感情"}).status_code == 404
    assert "爱吃辣， 常吃火锅" in view(client, level="public")["text"]
    assert summaries(client)["饮食"] is None


def test_summaries_left_out_of_export_are_not_regenerated(setup):
    client, _, prompts = setup
    generate(client)
    assert review(client, ["学业", "饮食"], "rejected").status_code == 200
    assert view(client)["text"] == ""
    reject_fact(client, "喜欢做实验")
    assert generate(client) == {"generating": 0}
    assert len(prompts) == 2
    assert review(client, ["学业"], "pending").status_code == 200
    assert generate(client) == {"generating": 1}


def test_batch_review_is_atomic(setup):
    client, _, _ = setup
    generate(client)
    rows = summaries(client)
    items = [{"domain": "学业", "version": rows["学业"]["version"]}, {"domain": "饮食", "version": 9}]
    response = client.post(
        "/api/summaries/review", json={"level": "normal", "items": items, "status": "approved"}
    )
    assert response.status_code == 409
    assert summaries(client)["学业"]["status"] == "pending"
    twice = {"level": "normal", "items": items[:1] * 2, "status": "approved"}
    assert client.post("/api/summaries/review", json=twice).status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "学业"},
        {"title": "学业", "summary": "在读", "extra": 1},
        {"title": "学业", "summary": " "},
        {"title": "长" * 13, "summary": "在读"},
        {"title": "学业", "summary": "长" * (profile_summary.SUMMARY_MAX_CHARS["normal"] + 1)},
        {"title": "学业", "summary": "和 P3 一起读博"},
        {"title": "学业", "summary": 1},
    ],
)
def test_invalid_model_output_is_rejected(payload):
    with pytest.raises(ValueError):
        profile_summary.validate_summary(payload, "normal")


def test_the_full_summary_may_run_longer():
    payload = {"title": "学业", "summary": "长" * profile_summary.SUMMARY_MAX_CHARS["full"]}
    assert profile_summary.validate_summary(payload, "full")["summary"] == payload["summary"]
    with pytest.raises(ValueError):
        profile_summary.validate_summary(payload, "normal")


def test_one_generation_at_a_time(setup):
    client, _, _ = setup
    assert profile_summary.GENERATING.acquire(blocking=False)
    try:
        assert view(client)["generating"] is True
        assert client.post("/api/summaries/generate", json={"level": "normal"}).status_code == 409
    finally:
        profile_summary.GENERATING.release()
    assert view(client)["generating"] is False


def test_summaries_that_came_back_are_kept_when_others_fail(setup, monkeypatch):
    client, _, prompts = setup
    model = profile_summary.summary_runner

    class Partial:
        def __init__(self, output_dir):
            pass

        def run(self, tasks, validate):
            validate({"title": "学业", "summary": "在读博士"}, tasks[0])
            raise RuntimeError("1 LLM tasks failed")

        def close(self):
            pass

    monkeypatch.setattr(profile_summary, "summary_runner", Partial)
    assert generate(client) == {"generating": 2}
    after = view(client)
    assert after["failure"] == {"failed": 1, "reason": ""}
    rows = {item["domain"]: item["summary"] for item in after["domains"]}
    assert rows["学业"]["summary"] == "在读博士" and rows["饮食"] is None

    monkeypatch.setattr(profile_summary, "summary_runner", model)
    assert generate(client) == {"generating": 1}
    assert prompts == [SHORT + "口味偏好：爱吃辣（兴趣与偏好，2 次，2024-07）"]
    assert view(client)["failure"] is None and summaries(client)["饮食"]["title"] == "标题1"


class Failing:
    def __init__(self, output_dir):
        raise OSError("codex not found")


def test_failed_runner_stores_nothing_and_reports_why(setup, monkeypatch):
    client, _, _ = setup
    monkeypatch.setattr(profile_summary, "summary_runner", Failing)
    assert generate(client) == {"generating": 2}
    after = view(client)
    assert after["generating"] is False
    assert after["failure"] == {"failed": 2, "reason": "codex not found"}
    assert all(item["summary"] is None for item in after["domains"])
    # Once nothing waits for a summary, the failure is no longer shown.
    for value in ("计算机博士生", "想去高校  任教", "喜欢做实验", "爱吃辣"):
        reject_fact(client, value)
    assert view(client)["failure"] is None


def test_build_profile_ends_with_the_full_summary_of_every_domain(setup, tmp_path):
    client, _, prompts = setup
    assert profile_summary.build_summaries(tmp_path) == ""
    assert prompts[0] == LONG + "\n".join(FULL_STUDY) and len(prompts) == 3
    rows = summaries(client, "full")
    assert [rows[domain]["title"] for domain in ("学业", "饮食", "感情")] == ["标题1", "标题2", "标题3"]
    assert {row["status"] for row in rows.values()} == {"pending"}
    assert not profile_summary.GENERATING.locked()
    # Versions for an assistant are written only when asked for.
    assert all(row is None for row in summaries(client).values())
    assert profile_summary.build_summaries(tmp_path) == "" and len(prompts) == 3


def test_build_profile_notes_domains_left_without_a_summary(setup, tmp_path, monkeypatch):
    monkeypatch.setattr(profile_summary, "summary_runner", Failing)
    note = profile_summary.build_summaries(tmp_path)
    assert note == "3 个板块没有生成成功，下次生成只补这些板块。原因：codex not found"


def test_build_profile_notes_a_failure_outside_the_model_run_and_frees_the_lock(setup, tmp_path, monkeypatch):
    def locked(profile, level):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(profile_summary, "domain_inputs", locked)
    assert profile_summary.build_summaries(tmp_path) == "没有生成成功。原因：database is locked"
    assert not profile_summary.GENERATING.locked()


def test_background_generation_reads_and_writes_with_the_session_key(tmp_path, monkeypatch):
    settings = tmp_path / "settings.jsonc"
    settings.write_text(
        json.dumps({"security_args": {"storage_mode": "encrypted", "state_dir": str(tmp_path / "state")}})
    )
    monkeypatch.setenv("WECLONE_CONFIG_PATH", str(settings))
    secure_storage.configure()
    secure_storage.ensure_initialized("summary-test")
    source = tmp_path / "profile_hierarchy.json"
    secure_storage.write_json(source, hierarchy())
    # From here on only the logged-in session holds the key.
    secure_storage.lock()

    caches = []

    class Runner:
        def __init__(self, output_dir):
            self.cache = LLMCache(output_dir / "llm_cache.sqlite")

        def run(self, tasks, validate):
            # The generation's answer cache is in secure storage too.
            answer = {"title": "标题", "summary": "总结"}
            self.cache.put_many([(task["prompt"], task["stage"], answer, "") for task in tasks])
            stored = secure_storage.encrypted_path(self.cache.path)
            caches.append((stored, secure_storage.is_encrypted_file(stored)))
            return [validate(answer, task) for task in tasks]

        def close(self):
            pass

    monkeypatch.setattr(profile_summary, "summary_runner", Runner)
    client = TestClient(create_app(tmp_path / "profile_review.sqlite3", source, tmp_path / "static"))
    client.headers["X-WeClone-Request"] = "1"
    assert client.post("/api/auth/login", json={"password": "summary-test"}).status_code == 200
    assert generate(client) == {"generating": 2}
    assert view(client)["failure"] is None
    assert summaries(client)["学业"]["title"] == "标题"
    # Encrypted while it lasted, and gone with the generation.
    stored, encrypted = caches[0]
    assert encrypted and not stored.exists()


def test_each_generation_asks_again_and_leaves_the_shared_cache_alone(tmp_path, monkeypatch):
    from weclone.core.inference import llm_client
    from weclone.core.inference.llm_client import LLMResponse
    from weclone.data.agent import llm_cache

    settings = tmp_path / "settings.jsonc"
    settings.write_text(json.dumps({
        "codex_exec_args": {
            "model": "fixture", "effort": "low", "batch_size": 1,
            "timeout": 30, "command": "codex", "sandbox": "read-only",
        },
        "agent_distill_args": {"llm_provider": "codex_exec"},
    }))
    monkeypatch.setenv("WECLONE_CONFIG_PATH", str(settings))
    asked = []

    class Client:
        def generate_batch(self, requests):
            requests = list(requests)
            asked.extend(requests)
            return [
                LLMResponse(ok=True, parsed_json={"title": "标题", "summary": "总结"}, finish_reason="stop")
                for _ in requests
            ]

        def close(self):
            pass

    monkeypatch.setattr(llm_client, "build_llm_client", lambda *a, **kw: Client())
    task = {"stage": "domain_summary", "prompt": "总结这个领域", "schema": SUMMARY_SCHEMA}
    for generation in ("first", "second"):
        runner = profile_summary.summary_runner(tmp_path / generation)
        assert runner.run([task], lambda payload, _: payload) == [{"title": "标题", "summary": "总结"}]
        runner.close()
    assert len(asked) == 2
    assert (tmp_path / "first" / "llm_cache.sqlite").is_file()
    assert not llm_cache.DEFAULT_PATH.exists()
