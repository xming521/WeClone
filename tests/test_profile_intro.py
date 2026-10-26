import json

import pytest
from fastapi.testclient import TestClient

from weclone.server.profile_intro import profile_intro, spread
from weclone.server.profile_review import create_app


@pytest.fixture
def client_for(tmp_path, monkeypatch):
    # Plaintext storage with its own password state, so the test never touches the real one.
    settings = tmp_path / "settings.jsonc"
    settings.write_text(json.dumps({"security_args": {"storage_mode": "plaintext", "state_dir": str(tmp_path / "state")}}))
    monkeypatch.setenv("WECLONE_CONFIG_PATH", str(settings))

    def create(data):
        source = tmp_path / "source.json"
        source.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        client = TestClient(create_app(tmp_path / "review.sqlite3", source, tmp_path / "no-static"))
        client.headers["X-WeClone-Request"] = "1"
        password = {"password": "intro-test", "confirmation": "intro-test"}
        assert client.post("/api/auth/setup", json=password).status_code == 200
        return client

    return create


def hierarchy(*groups):
    return [{"dim": dim, "name": name, "groups": [
        {"name": group, "items": [{"attr": attr, "fact_indices": indices}]} for group, attr, indices in items
    ]} for dim, name, items in groups]


@pytest.fixture
def chats(tmp_path):
    chat = tmp_path / "chats.json"
    chat.write_text(json.dumps([
        {"id": "a", "time": "2021-03-02 10:00", "messages": [
            {"role": "user", "content": "周末去哪"},
            {"role": "assistant", "content": "我打算去西湖边跑步\n顺便拍几张照片回来"},
        ]},
        {"id": "b", "time": "2023-07-09 21:00", "messages": [
            {"role": "assistant", "content": "最近睡眠一直不太好呢"},
            {"role": "assistant", "content": "好"},
        ]},
        {"id": "c", "time": "2019-12-30 08:00", "messages": [
            {"role": "assistant", "content": "这句话不属于任何记忆来源"},
        ]},
        {"id": "d", "time": "2025-01-01 09:00", "messages": [
            {"role": "assistant", "content": "新年第一天去爬山了"},
        ]},
    ], ensure_ascii=False), encoding="utf-8")
    origin = lambda sample: [{"file": str(chat), "sample_id": sample}]  # noqa: E731
    return {
        "domains": [],
        "dimensions": hierarchy(
            (8, "兴趣", [("运动", "运动偏好", [0]), ("户外", "户外偏好", [2])]),
            (1, "基本身份", [("健康状况", "睡眠", [1])]),
        ),
        "facts": [
            {"dim": 8, "attr": "运动偏好", "value": "喜欢跑步", "source_ids": ["s1"]},
            {"dim": 1, "attr": "睡眠", "value": "睡得少", "source_ids": ["s2"]},
            {"dim": 8, "attr": "户外偏好", "value": "爬山", "source_ids": ["s3"]},
        ],
        "sources": {
            "s1": {"id": "s1", "content": "跑步", "origins": origin("a")},
            "s2": {"id": "s2", "content": "睡眠", "origins": origin("b")},
            # The same chat again: counted once.
            "s3": {"id": "s3", "content": "爬山", "origins": [*origin("d"), *origin("a")]},
        },
    }


def test_counts_every_owner_message_and_quotes_only_linked_safe_chats(client_for, chats):
    client = client_for(chats)
    intro = client.get("/api/profile/intro").json()
    # Every non-empty owner message of the referenced file, newline-merged parts counted apart.
    assert intro["message_total"] == 6
    assert (intro["first_time"], intro["last_time"]) == ("2019-12-30 08:00", "2025-01-01 09:00")
    assert intro["sentences"] == [
        {"text": "我打算去西湖边跑步", "time": "2021-03-02 10:00", "dim": 8},
        {"text": "顺便拍几张照片回来", "time": "2021-03-02 10:00", "dim": 8},
        {"text": "新年第一天去爬山了", "time": "2025-01-01 09:00", "dim": 8},
    ]


def test_rejected_fact_no_longer_lends_lines(client_for, chats):
    client = client_for(chats)
    climbing = next(fact for fact in client.get("/api/profile").json()["facts"] if fact["value"] == "爬山")
    response = client.post("/api/reviews/batch", json={
        "items": [{"id": climbing["id"], "version": climbing["version"]}], "status": "rejected",
    })
    assert response.status_code == 200
    intro = client.get("/api/profile/intro").json()
    assert intro["message_total"] == 6
    assert [line["text"] for line in intro["sentences"]] == ["我打算去西湖边跑步", "顺便拍几张照片回来"]


def test_inline_conversations_use_the_most_frequent_last_speaker(client_for):
    data = {
        "_demo": {"message_total": 41236, "messages_by_month": {"2019-05": 3, "2026-02": 7, "2018-01": 0}},
        "domains": [],
        "dimensions": hierarchy((8, "兴趣", [("运动", "运动偏好", [0, 1])]), (5, "计划", [("投资理财", "基金", [2])])),
        "facts": [
            {"dim": 8, "attr": "运动偏好", "value": "跑步", "source_ids": ["s1"]},
            {"dim": 8, "attr": "运动偏好", "value": "游泳", "source_ids": ["s2"]},
            {"dim": 5, "attr": "基金", "value": "定投", "source_ids": ["s3"]},
        ],
        "sources": {
            "s1": {"sample_time": "2024-01-01", "content": "朋友：最近还跑步吗最近还跑步吗\n小林：每周都会去跑两三次"},
            "s2": {"sample_time": "2022-01-01", "content": "朋友：游泳呢\n小林：夏天常去游泳馆游泳"},
            "s3": {"sample_time": "2023-01-01", "content": "朋友：基金呢\n小林：每个月都定投一点基金"},
        },
    }
    intro = client_for(data).get("/api/profile/intro").json()
    assert intro["message_total"] == 41236
    assert (intro["first_time"], intro["last_time"]) == ("2019-05", "2026-02")
    assert [line["text"] for line in intro["sentences"]] == ["夏天常去游泳馆游泳", "每周都会去跑两三次"]


def test_lines_are_filtered_deduplicated_and_spread():
    data = {
        "dimensions": hierarchy((8, "兴趣", [("运动", "运动偏好", [0])])),
        "facts": [{"source_ids": ["s1"]}],
        "sources": {"s1": {"content": "\n".join(
            ["我：太短", "我：这一句话实在是太长了已经超过了十八个字的上限", "我：[图片]这是表情包", "我：上个月工资涨了一万",
             "我：周末一起去看展吧", "我：周末一起去看展吧"]
        )}},
    }
    assert [line["text"] for line in profile_intro(data)["sentences"]] == ["周末一起去看展吧"]
    assert profile_intro(data)["message_total"] is None
    assert spread(list(range(10)), 3) == [1, 5, 8]
    assert spread([1, 2], 3) == [1, 2]
