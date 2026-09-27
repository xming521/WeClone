import json

import pytest
from fastapi.testclient import TestClient

from weclone.server.profile_review import create_app


@pytest.fixture
def setup(tmp_path):
    data = {
        "dimensions": [{"dim": 8, "name": "兴趣", "groups": [
            {"name": "运动", "items": [{"attr": "运动偏好", "fact_indices": [0, 1]}]},
            {"name": "阅读", "items": [{"attr": "阅读偏好", "fact_indices": [2]}]},
        ]}],
        "facts": [
            {"dim": 8, "attr": "运动偏好", "value": "喜欢跑步", "source_ids": ["s1"]},
            {"dim": 8, "attr": "运动偏好", "value": "喜欢游泳", "source_ids": ["s2"]},
            {"dim": 8, "attr": "阅读偏好", "value": "喜欢科幻小说", "source_ids": ["s3"]},
        ],
        "sources": {sid: {"id": sid, "content": content} for sid, content in [
            ("s1", "跑步来源"), ("s2", "游泳来源"), ("s3", "阅读来源")
        ]},
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    database = tmp_path / "review.sqlite3"
    client = TestClient(create_app(database, source, tmp_path / "no-static"))
    return client, database, source


def facts(client):
    return client.get("/api/profile").json()["facts"]


def review(client, items, status):
    return client.post("/api/reviews/batch", json={
        "items": [{"id": item["id"], "version": item["version"]} for item in items], "status": status,
    })


def edit(client, fact, **changes):
    payload = {key: fact[key] for key in ("value", "attr", "group_id", "version")}
    payload.update(changes)
    return client.patch(f"/api/facts/{fact['id']}", json=payload)


def test_individual_approval_export_and_snapshot_persistence(setup):
    client, database, source = setup
    original = source.read_bytes()
    first, second, _ = facts(client)
    assert first["node_id"] == second["node_id"]
    assert client.get("/api/avatar-profile").json()["facts"] == []
    assert review(client, [first], "approved").status_code == 200
    assert review(client, [second], "rejected").status_code == 200
    avatar = client.get("/api/avatar-profile?download=true")
    assert "attachment" in avatar.headers["content-disposition"]
    assert avatar.headers["cache-control"] == "no-store"
    exported = avatar.json()
    assert [f["value"] for f in exported["facts"]] == ["喜欢跑步"]
    assert set(exported["sources"]) == {"s1"}
    assert exported["dimensions"][0]["groups"][0]["items"][0]["fact_indices"] == [0]
    assert len(exported["dimensions"][0]["groups"]) == 1
    assert "original" not in exported["facts"][0]
    assert source.read_bytes() == original
    source.unlink()  # Subsequent startup must use the persisted snapshot.
    restarted = TestClient(create_app(database, source))
    assert facts(restarted) == facts(client)


def test_edit_requires_review_and_keeps_original(setup):
    client, _, _ = setup
    first = facts(client)[0]
    review(client, [first], "approved")
    first = facts(client)[0]
    same = edit(client, first).json()
    assert same["status"] == "approved" and same["version"] == first["version"]
    changed = edit(client, first, value="现在偏好徒步").json()
    assert changed["status"] == "pending"
    assert changed["original"]["value"] == "喜欢跑步"
    assert changed["source_ids"] == ["s1"]
    assert client.get("/api/avatar-profile").json()["facts"] == []
    approved = edit(client, changed, value="喜欢短途徒步", approve=True).json()
    assert approved["status"] == "approved"
    assert client.get("/api/avatar-profile").json()["facts"][0]["value"] == "喜欢短途徒步"
    history = client.get(f"/api/facts/{first['id']}/history").json()
    assert len(history) == 3
    assert history[0]["before"]["value"] == "现在偏好徒步"
    assert history[0]["after"]["value"] == "喜欢短途徒步"


def test_move_and_rename_only_one_fact(setup):
    client, _, _ = setup
    first, second, third = facts(client)
    moved = edit(client, first, attr="新属性", group_id=third["group_id"]).json()
    assert moved["group_id"] == third["group_id"]
    assert moved["node_id"] != second["node_id"]
    assert facts(client)[1] == second
    # Moving back to an existing attribute reuses that node.
    moved_back = edit(client, moved, attr=second["attr"], group_id=second["group_id"]).json()
    assert moved_back["node_id"] == second["node_id"]


def test_batch_conflict_is_atomic_and_stale_edit_rejected(setup):
    client, _, _ = setup
    first, second, _ = facts(client)
    review(client, [second], "rejected")
    assert review(client, [first, second], "approved").status_code == 409
    assert [f["status"] for f in facts(client)][:2] == ["pending", "rejected"]
    assert client.get(f"/api/facts/{first['id']}/history").json() == []
    assert edit(client, second, value="过期页面修改").status_code == 409
    assert facts(client)[1]["value"] == second["value"]
    assert review(client, [first, first], "approved").status_code == 422


def test_manual_fact_and_direct_dimension_export(setup):
    client, _, _ = setup
    payload = {"value": "偏好手工制作", "attr": "手作", "group_id": "dimension:8"}
    created = client.post("/api/facts", json=payload)
    assert created.status_code == 201
    manual = created.json()
    assert manual["origin"] == "manual" and manual["source_ids"] == []
    assert manual["status"] == "pending"
    review(client, [manual], "approved")
    avatar = client.get("/api/avatar-profile").json()
    assert avatar["sources"] == {}
    assert avatar["dimensions"][0]["groups"][0]["items"][0]["fact_indices"] == [0]
    assert avatar["facts"][0]["value"] == payload["value"]


def test_validation_and_reversing_decision(setup):
    client, _, _ = setup
    first = facts(client)[0]
    assert edit(client, first, value="  ").status_code == 422
    assert edit(client, first, group_id=first["node_id"]).status_code == 422
    assert edit(client, first, source_ids=[]).status_code == 422
    for status in ("approved", "rejected", "pending", "approved"):
        assert review(client, [facts(client)[0]], status).status_code == 200
        assert facts(client)[0]["status"] == status
    assert len(facts(client)) == 3


def test_invalid_snapshot_import_rolls_back(tmp_path):
    source = tmp_path / "source.json"
    database = tmp_path / "review.sqlite3"
    data = {"dimensions": [{"dim": 8, "name": "兴趣", "groups": []}], "facts": [{"value": "未挂载"}], "sources": {}}
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="every fact"):
        create_app(database, source)
    data["facts"] = []
    source.write_text(json.dumps(data))
    client = TestClient(create_app(database, source))
    assert facts(client) == []
