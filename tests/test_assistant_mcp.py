import json
import sqlite3
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from weclone.data.agent.llm_profile_hierarchy import dimension_view
from weclone.server import assistant, auth, profile_summary
from weclone.server.profile_review import create_app
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
    fact(3, "职业目标", "想去高校任教", "normal", ["s3"]),
    fact(5, "答辩安排", "六月答辩", "normal", ["s3"], scope="dated", support=4),
    fact(1, "在读身份", "硕士生", "public", ["s1"], scope="superseded", support=5),
    fact(4, "科研补助", "每月三千元", "private", ["s2"], support=2),
    fact(8, "口味偏好", "爱吃辣", "public", ["s4"], support=2),
    fact(8, "运动偏好", "喜欢和 P3 一起打球", "normal", ["s4"]),
]
TIMES = {
    "s1": "2023-09-01T10:00:00+08:00",
    "s2": "2024-05-20T10:00:00+08:00",
    "s3": "2024-02-11 09:00:00",
    "s4": "2024-07-03T10:00:00+08:00",
}


def hierarchy():
    domains = [
        {
            "name": "学业",
            "description": "读博",
            "topics": [{"name": "读博", "fact_indices": [0, 1, 2, 3, 4]}],
        },
        {"name": "生活", "description": "口味", "topics": [{"name": "口味", "fact_indices": [5, 6]}]},
    ]
    return {
        "domains": domains,
        "dimensions": dimension_view(FACTS, domains),
        "facts": FACTS,
        "sources": {sid: {"id": sid, "content": f"来源{sid}", "sample_time": t} for sid, t in TIMES.items()},
    }


class Runner:
    """The summary model: summary i for the i-th domain it is asked about."""

    def __init__(self, output_dir):
        pass

    def run(self, tasks, validate):
        return [
            validate({"title": f"标题{i}", "summary": f"总结{i}"}, task) for i, task in enumerate(tasks, 1)
        ]

    def close(self):
        pass


def use_storage(tmp_path, monkeypatch, mode):
    settings = tmp_path / "settings.jsonc"
    settings.write_text(
        json.dumps({"security_args": {"storage_mode": mode, "state_dir": str(tmp_path / "state")}})
    )
    monkeypatch.setenv("WECLONE_CONFIG_PATH", str(settings))
    secure_storage.configure()
    monkeypatch.setattr(profile_summary, "summary_runner", Runner)


@pytest.fixture
def owner(tmp_path, monkeypatch):
    """The owner's logged-in browser, on plaintext storage."""
    use_storage(tmp_path, monkeypatch, "plaintext")
    source = tmp_path / "profile_hierarchy.json"
    source.write_text(json.dumps(hierarchy(), ensure_ascii=False), encoding="utf-8")
    client = TestClient(create_app(tmp_path / "profile_review.sqlite3", source, tmp_path / "static"))
    client.headers["X-WeClone-Request"] = "1"
    password = {"password": "assistant-test", "confirmation": "assistant-test"}
    assert client.post("/api/auth/setup", json=password).status_code == 200
    return client


def new_token(owner):
    response = owner.post("/api/assistant/token")
    assert response.status_code == 201
    return response.json()["token"]


def choose(owner, level):
    assert owner.patch("/api/assistant", json={"level": level}).status_code == 200


class Assistant:
    """An MCP client holding the token, without the owner's cookie."""

    def __init__(self, app, token, user_agent="testclient"):
        self.client = TestClient(app)
        self.headers = {"Authorization": f"Bearer {token}", "User-Agent": user_agent}
        self.next_id = 0

    def post(self, message, **headers):
        return self.client.post("/mcp", json=message, headers={**self.headers, **headers})

    def request(self, method, params=None):
        self.next_id += 1
        message = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        response = self.post({**message, "params": params} if params is not None else message)
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        return response.json()

    def call(self, tool, **arguments):
        result = self.request("tools/call", {"name": tool, "arguments": arguments})["result"]
        return result["content"][0]["text"], result["isError"]


def facts_by_value(owner):
    return {item["value"]: item for item in owner.get("/api/profile").json()["facts"]}


def test_the_token_is_shown_once_kept_as_a_hash_and_outlives_rebuilds(owner, tmp_path):
    view = owner.get("/api/assistant").json()
    assert view["encrypted"] is False and view["allowed_until"] is None
    assert view["mcp_url"] == "http://testserver/mcp"
    assert view["level"] == "full" and view["token"] is None and view["domains"] is None
    assert view["summaries"] == 0
    assert view["catalog"] == [
        {"name": "学业", "facts": 5, "summary": False, "shared": True},
        {"name": "生活", "facts": 2, "summary": False, "shared": True},
    ]
    token = new_token(owner)
    assert token.startswith("wc_")
    shown = owner.get("/api/assistant").json()["token"]
    assert shown["created"] and shown["last_used"] is None and token not in json.dumps(shown)
    stored = tmp_path / "state" / "assistant.sqlite3"
    assert stored.stat().st_mode & 0o777 == 0o600 and token.encode() not in stored.read_bytes()
    assert owner.patch("/api/assistant", json={"level": "private"}).status_code == 422
    stranger = TestClient(owner.app, headers={"X-WeClone-Request": "1"})
    assert stranger.post("/api/assistant/token").status_code == 401
    assert stranger.patch("/api/assistant", json={"level": "public"}).status_code == 401
    # Plaintext data needs no permission beyond the token.
    assert owner.post("/api/assistant/access", json={"hours": 1}).status_code == 409

    choose(owner, "normal")
    rebuilt = create_app(
        tmp_path / "rebuilt.sqlite3", tmp_path / "profile_hierarchy.json", tmp_path / "static"
    )
    agent = Assistant(rebuilt, token)
    assert agent.call("weclone_search", keywords=["补助"]) == (assistant.NOT_FOUND, False)
    # A new token stops the old one.
    renewed = new_token(owner)
    assert agent.post({"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code == 401
    assert Assistant(rebuilt, renewed).request("ping")["result"] == {}
    assert owner.delete("/api/assistant/token").status_code == 200
    assert Assistant(rebuilt, renewed).post({"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code == 401
    assert owner.get("/api/assistant").json()["token"] is None


def test_mcp_turns_away_web_pages_and_unknown_tokens(owner):
    agent = Assistant(owner.app, new_token(owner))
    ping = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    unknown = Assistant(owner.app, "wc_unknown").post(ping)
    assert unknown.status_code == 401 and unknown.headers["www-authenticate"] == "Bearer"
    # The owner's session cookie is no token.
    assert owner.post("/mcp", json=ping).status_code == 401
    assert agent.post(ping, Origin="http://127.0.0.1:5175").status_code == 403
    assert agent.post(ping, **{"MCP-Protocol-Version": "2024-01-01"}).status_code == 400
    assert agent.post(ping, **{"MCP-Protocol-Version": "2025-06-18"}).status_code == 200
    broken = agent.client.post("/mcp", content=b"{", headers=agent.headers)
    assert broken.status_code == 400 and broken.json()["error"]["code"] == -32700
    assert agent.post([ping]).status_code == 400
    notified = agent.post({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert notified.status_code == 202 and notified.content == b""
    stream = agent.client.get("/mcp", headers=agent.headers)
    assert stream.status_code == 405 and stream.headers["allow"] == "POST"
    assert agent.client.delete("/mcp", headers=agent.headers).status_code == 405


def test_initialize_and_list_read_only_tools(owner):
    agent = Assistant(owner.app, new_token(owner))
    hello = {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"},
    }
    started = agent.request("initialize", hello)["result"]
    assert started["protocolVersion"] == "2025-06-18" and started["serverInfo"]["name"] == "weclone"
    assert started["capabilities"] == {"tools": {"listChanged": False}}
    assert "weclone_profile" in started["instructions"]
    # A client asking for a version this server lacks is offered the newest.
    newer = agent.request("initialize", {**hello, "protocolVersion": "2099-01-01"})["result"]
    assert newer["protocolVersion"] == "2025-11-25"
    tools = agent.request("tools/list")["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["weclone_profile", "weclone_search"]
    assert all(
        tool["annotations"]["readOnlyHint"] and tool["inputSchema"]["type"] == "object" for tool in tools
    )
    assert agent.request("resources/list")["error"]["code"] == -32601
    assert agent.request("tools/call", {"name": "shell"})["error"]["code"] == -32602


def test_profile_serves_the_approved_current_summaries_of_the_chosen_level(owner):
    agent = Assistant(owner.app, new_token(owner))
    choose(owner, "normal")
    assert agent.call("weclone_profile") == (assistant.NO_SUMMARIES, False)
    assert owner.post("/api/summaries/generate", json={"level": "normal"}).status_code == 200
    with profile_summary.GENERATING:
        pass
    # Written but not approved yet.
    assert agent.call("weclone_profile")[0] == assistant.NO_SUMMARIES
    rows = {
        item["domain"]: item["summary"] for item in owner.get("/api/summaries?level=normal").json()["domains"]
    }
    approve = {"level": "normal", "items": [{"domain": "生活", "version": rows["生活"]["version"]}]}
    assert owner.post("/api/summaries/review", json={**approve, "status": "approved"}).status_code == 200

    text, failed = agent.call("weclone_profile")
    assert (
        not failed and text.startswith(assistant.PROFILE_HEADER) and text.endswith(assistant.PROFILE_FOOTER)
    )
    # Domain ids count every domain, so they are the ones weclone_search takes.
    assert "\n- [d2] 标题2（截至 2024-07）：总结2\n" in text and "[d1]" not in text
    assert owner.get("/api/assistant").json()["summaries"] == 1
    # A new level holds from the next call, with no change on the assistant's side.
    choose(owner, "public")
    assert agent.call("weclone_profile")[0] == assistant.NO_SUMMARIES
    choose(owner, "normal")

    # A changed fact takes its domain's summary out until it is written and approved again.
    spicy = facts_by_value(owner)["爱吃辣"]
    edit = {
        "value": "爱吃甜",
        "attr": spicy["attr"],
        "group_id": spicy["group_id"],
        "version": spicy["version"],
    }
    assert owner.patch(f"/api/facts/{spicy['id']}", json=edit).status_code == 200
    assert agent.call("weclone_profile")[0] == assistant.NO_SUMMARIES


def test_search_keeps_to_the_chosen_level_and_marks_what_no_longer_holds(owner):
    agent = Assistant(owner.app, new_token(owner))
    choose(owner, "normal")
    text, failed = agent.call("weclone_search", keywords=["身份"])
    assert not failed
    # What still holds comes before what came up more often.
    assert text.splitlines() == [
        "找到 2 条：",
        "- [d1] 在读身份：计算机博士生（人口学与基本身份，当前，3 次，2023-09~2024-05）",
        "- [d1] 在读身份：硕士生（人口学与基本身份，已过时，5 次，2023-09）",
    ]
    assert agent.call("weclone_search", keywords=["补助"]) == (assistant.NOT_FOUND, False)
    played = agent.call("weclone_search", keywords=["打球"])[0]
    assert "运动偏好：喜欢和 一位联系人 一起打球" in played and "P3" not in played
    assert agent.call("weclone_search", domain="d2")[0].splitlines()[1:] == [
        "- [d2] 口味偏好：爱吃辣（兴趣与偏好，当前，2 次，2024-07）",
        "- [d2] 运动偏好：喜欢和 一位联系人 一起打球（兴趣与偏好，当前，1 次，2024-07）",
    ]

    choose(owner, "full")
    assert "科研补助：每月三千元" in agent.call("weclone_search", keywords=["补助"])[0]

    spicy = facts_by_value(owner)["爱吃辣"]
    rejected = {"items": [{"id": spicy["id"], "version": spicy["version"]}], "status": "rejected"}
    assert owner.post("/api/reviews/batch", json=rejected).status_code == 200
    assert agent.call("weclone_search", keywords="辣") == (assistant.NOT_FOUND, False)


def test_search_matches_the_text_the_assistant_sees_not_the_peer_id(owner):
    agent = Assistant(owner.app, new_token(owner))
    choose(owner, "normal")
    assert agent.call("weclone_search", keywords=["P3"]) == (assistant.NOT_FOUND, False)
    text, failed = agent.call("weclone_search", keywords=["联系人"])
    assert not failed and "运动偏好：喜欢和 一位联系人 一起打球" in text and "P3" not in text


def test_domains_left_out_are_neither_summarised_nor_searched(owner):
    agent = Assistant(owner.app, new_token(owner))
    assert owner.post("/api/summaries/generate", json={"level": "full"}).status_code == 200
    with profile_summary.GENERATING:
        pass
    rows = owner.get("/api/summaries?level=full").json()["domains"]
    approve = [{"domain": item["domain"], "version": item["summary"]["version"]} for item in rows]
    review = {"level": "full", "items": approve, "status": "approved"}
    assert owner.post("/api/summaries/review", json=review).status_code == 200
    assert "[d1]" in agent.call("weclone_profile")[0]

    def share(domains):
        response = owner.put("/api/assistant/domains", json={"domains": domains})
        assert response.status_code == 200
        return response.json()["domains"]

    assert share([" 生活", "生活", "旧领域"]) == ["生活", "旧领域"]
    view = owner.get("/api/assistant").json()
    assert view["domains"] == ["生活", "旧领域"] and view["summaries"] == 1
    assert [(item["name"], item["shared"], item["summary"]) for item in view["catalog"]] == [
        ("学业", False, True),
        ("生活", True, True),
    ]
    text = agent.call("weclone_profile")[0]
    assert "- [d2] " in text and "[d1]" not in text
    assert agent.call("weclone_search", keywords=["身份"]) == (assistant.NOT_FOUND, False)
    assert "[d2] 口味偏好：爱吃辣" in agent.call("weclone_search", keywords=["辣"])[0]
    text, failed = agent.call("weclone_search", domain="d1")
    assert failed and text.startswith("没有板块 d1")

    # Domains that come after a list was chosen stay out. Without a list every domain is in.
    assert share(["学业", "生活"]) == ["学业", "生活"]
    assert "计算机博士生" in agent.call("weclone_search", keywords=["身份"])[0]
    assert share(None) is None
    assert owner.get("/api/assistant").json()["domains"] is None
    assert share([]) == []
    assert agent.call("weclone_profile")[0] == assistant.NO_SUMMARIES
    assert agent.call("weclone_search", keywords=["辣"]) == (assistant.NOT_FOUND, False)
    assert owner.put("/api/assistant/domains", json={}).status_code == 422
    assert owner.put("/api/assistant/domains", json={"domains": ["长" * 101]}).status_code == 422


def test_search_limits_and_explains_unusable_arguments(owner):
    agent = Assistant(owner.app, new_token(owner))
    lines = agent.call("weclone_search", domain="D1", limit=2)[0].splitlines()
    assert lines[0] == "找到 5 条，先列出最相关的 2 条："
    assert "六月答辩" in lines[1] and "计算机博士生" in lines[2]
    assert lines[3] == "还有 3 条没有列出，可以加关键词或用 domain 缩小范围。"
    for arguments, message in [
        ({}, "请至少给出 keywords 或 domain"),
        ({"keywords": [1]}, "keywords 要是字符串列表"),
        ({"domain": "学业"}, "domain 要是"),
        ({"domain": "d9"}, "没有板块 d9"),
        ({"keywords": ["辣"], "limit": 0}, "limit 要是"),
    ]:
        text, failed = agent.call("weclone_search", **arguments)
        assert failed and text.startswith(message)


def test_each_call_is_logged_with_the_client_and_what_it_returned(owner, tmp_path):
    agent = Assistant(owner.app, new_token(owner), user_agent="codex_cli_rs/0.163.0")
    agent.call("weclone_search", keywords=["辣"])
    agent.call("weclone_search", keywords=[])
    db = sqlite3.connect(tmp_path / "profile_review.sqlite3")
    try:
        rows = db.execute("SELECT client, tool, arguments, returned FROM assistant_calls").fetchall()
    finally:
        db.close()
    # Calls the tool could not run return nothing and leave no row.
    assert rows == [
        (
            "codex_cli_rs/0.163.0",
            "weclone_search",
            '{"keywords": ["辣"]}',
            json.dumps([facts_by_value(owner)["爱吃辣"]["id"]]),
        )
    ]
    assert owner.get("/api/assistant").json()["token"]["last_used"]


def test_a_failing_audit_write_is_a_tool_error_not_a_server_error(owner, monkeypatch):
    def locked(self, *args):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(assistant.AssistantTools, "log", locked)
    agent = Assistant(owner.app, new_token(owner))
    text, failed = agent.call("weclone_search", keywords=["辣"])
    assert failed and "database is locked" in text


def test_encrypted_data_is_read_only_while_the_owner_allows_it(tmp_path, monkeypatch):
    use_storage(tmp_path, monkeypatch, "encrypted")
    secure_storage.ensure_initialized("assistant-test")
    source = tmp_path / "profile_hierarchy.json"
    secure_storage.write_json(source, hierarchy())
    # From here on only the server holds the key.
    secure_storage.lock()
    app = create_app(tmp_path / "profile_review.sqlite3", source, tmp_path / "static")
    owner = TestClient(app, headers={"X-WeClone-Request": "1"})

    def login():
        assert owner.post("/api/auth/login", json={"password": "assistant-test"}).status_code == 200

    def allow(hours=1):
        response = owner.post("/api/assistant/access", json={"hours": hours})
        assert response.status_code == 200
        return response.json()["allowed_until"]

    login()
    agent = Assistant(app, new_token(owner))
    text, failed = agent.call("weclone_search", keywords=["辣"])
    assert failed and "http://testserver/#/assistant" in text
    assert owner.post("/api/assistant/access", json={"hours": 2}).status_code == 422
    until = allow()
    assert abs(until - time.time() - 3600) < 60
    assert owner.get("/api/assistant").json()["allowed_until"] == until
    # The owner's session may end. The kept key reads until the time is up.
    assert owner.post("/api/auth/logout").status_code == 200
    assert agent.call("weclone_search", keywords=["辣"]) == (
        "找到 1 条：\n- [d2] 口味偏好：爱吃辣（兴趣与偏好，当前，2 次，2024-07）",
        False,
    )
    assert not (tmp_path / "profile_review.sqlite3").exists()
    with monkeypatch.context() as later:
        later.setattr(auth, "time", SimpleNamespace(time=lambda: until + 1))
        assert agent.call("weclone_search", keywords=["辣"])[1]
    assert agent.call("weclone_search", keywords=["辣"])[1]

    login()
    allow()
    assert owner.delete("/api/assistant/access").status_code == 200
    assert agent.call("weclone_search", keywords=["辣"])[1]
    allow()
    assert not agent.call("weclone_search", keywords=["辣"])[1]
    # A new password ends access as it ends sessions.
    secure_storage.change_password("assistant-test", "assistant-new")
    assert agent.call("weclone_search", keywords=["辣"])[1]
