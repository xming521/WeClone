import json

import pytest
from fastapi.testclient import TestClient

from weclone.data.agent import llm_profile_hierarchy as hierarchy
from weclone.prompts.memory_organization import dimension_name
from weclone.prompts.profile_hierarchy import DESCRIBE_SCHEMA, SPLIT_SCHEMA
from weclone.server.profile_review import create_app


def fact(dim, domain, topic, attr, value, sid, **extra):
    return {"dim": dim, "domain": domain, "topic": topic, "attr": attr, "value": value, "source_ids": [sid], **extra}


def organized():
    return {
        "domains": [
            {"name": "出行旅行", "description": "旧描述一"},
            {"name": "学业", "description": "旧描述二"},
            {"name": "工作", "description": "没有画像条目"},
        ],
        "topics": [
            {"domain": "出行旅行", "name": "周末郊游", "attrs": [[8, "活动偏好"]]},
            {"domain": "学业", "name": "读博申请", "attrs": [[1, "目标院校"], [3, "读博目标"], [5, "面试安排"]]},
            {"domain": "学业", "name": "课程", "attrs": [[1, "课程"]]},
        ],
        "facts": [
            fact(1, "学业", "读博申请", "目标院校", "B申请了甲大学。", "s1"),
            fact(3, "学业", "读博申请", "读博目标", "B想读博。", "s2"),
            fact(5, "学业", "读博申请", "面试安排", "B下周面试。", "s3"),
            fact(1, "学业", "课程", "课程", "B在上机器学习课。", "s4"),
            fact(6, None, None, "关系状态", "P1是B的同学。", "s5", peer="P1"),
            fact(7, "出行旅行", None, "天气", "周末下雨。", "s6"),
            fact(8, "出行旅行", "周末郊游", "活动偏好", "B喜欢徒步。", "s7"),
        ],
        "unused": [],
        "unassigned_ids": [],
        "sources": {f"s{i}": {"id": f"s{i}", "content": f"来源{i}"} for i in range(1, 8)},
    }


def test_layout_keeps_profile_facts_in_domain_topics_and_drops_empty_domains():
    facts, domains = hierarchy.topic_layout(organized())
    assert [row["dim"] for row in facts] == [1, 3, 5, 1, 8]
    assert domains == [
        {"name": "出行旅行", "description": "旧描述一", "topics": [{"name": "周末郊游", "fact_indices": [4]}]},
        {"name": "学业", "description": "旧描述二", "topics": [
            {"name": "读博申请", "fact_indices": [0, 1, 2]}, {"name": "课程", "fact_indices": [3]},
        ]},
    ]


@pytest.mark.parametrize("change", [{"topic": "不存在"}, {"domain": None}, {"domain": "工作"}])
def test_layout_rejects_profile_facts_outside_known_topics(change):
    data = organized()
    data["facts"][0].update(change)
    with pytest.raises(ValueError, match="Profile fact 0"):
        hierarchy.topic_layout(data)


def test_dimension_view_nests_domain_topic_and_attribute():
    facts, domains = hierarchy.topic_layout(organized())
    assert hierarchy.dimension_view(facts, domains) == [
        {"dim": 1, "name": dimension_name(1), "groups": [{"name": "学业", "items": [
            {"name": "读博申请", "items": [{"attr": "目标院校", "fact_indices": [0]}]},
            {"name": "课程", "items": [{"attr": "课程", "fact_indices": [3]}]},
        ]}]},
        {"dim": 3, "name": dimension_name(3), "groups": [{"name": "学业", "items": [
            {"name": "读博申请", "items": [{"attr": "读博目标", "fact_indices": [1]}]},
        ]}]},
        {"dim": 5, "name": dimension_name(5), "groups": [{"name": "学业", "items": [
            {"name": "读博申请", "items": [{"attr": "面试安排", "fact_indices": [2]}]},
        ]}]},
        {"dim": 8, "name": dimension_name(8), "groups": [{"name": "出行旅行", "items": [
            {"name": "周末郊游", "items": [{"attr": "活动偏好", "fact_indices": [4]}]},
        ]}]},
    ]


def test_split_replaces_large_topics_and_renames_names_two_splits_share():
    facts = [fact(1, "学业", "主题", "属性", f"事实{i}", f"s{i}") for i in range(7)]
    domains = [{"name": "学业", "description": "读博和课程", "topics": [
        {"name": "读博申请", "fact_indices": [0, 1, 2]},
        {"name": "课程", "fact_indices": [3, 4, 5]},
        {"name": "其他", "fact_indices": [6]},
    ]}]
    second = {"读博申请": "申请材料", "课程": "作业"}
    prompts = []

    class Runner:
        def run(self, tasks, validate):
            results = []
            for task in tasks:
                assert task["stage"] == "split_topic" and task["schema"] is SPLIT_SCHEMA
                topic = task["prompt"].split("## 原主题\n", 1)[1].split("\n", 1)[0]
                prompts.append(task["prompt"])
                payload = {"topics": [{"name": "日常", "items": [3]}, {"name": second[topic], "items": [2, 1]}]}
                results.append(validate(payload, task))
            return results

    refined, report = hierarchy.split_topics(domains, facts, Runner(), leaf_facts=2)
    assert refined[0]["topics"] == [
        {"name": "日常", "fact_indices": [2]},
        {"name": "申请材料", "fact_indices": [0, 1]},
        {"name": "课程·日常", "fact_indices": [5]},
        {"name": "作业", "fact_indices": [3, 4]},
        {"name": "其他", "fact_indices": [6]},
    ]
    assert report == [
        {"domain": "学业", "topic": "读博申请", "facts": 3, "into": {"日常": 1, "申请材料": 2}},
        {"domain": "学业", "topic": "课程", "facts": 3, "into": {"课程·日常": 1, "作业": 2}},
    ]
    assert "## 本领域其他主题\n课程、其他\n" in prompts[0]
    assert "3 属性：事实2（人口学与基本身份）" in prompts[0]


def test_split_skips_the_model_when_topics_are_small():
    class Runner:
        def run(self, tasks, validate):
            pytest.fail("No topic exceeds the limit")

    domains = [{"name": "学业", "description": "课程", "topics": [{"name": "课程", "fact_indices": [0]}]}]
    assert hierarchy.split_topics(domains, [fact(1, "学业", "课程", "课程", "事实", "s1")], Runner(), 30) == (
        domains,
        [],
    )


@pytest.mark.parametrize("topics", [
    [{"name": "申请材料", "items": [1, 2]}],
    [{"name": "申请材料", "items": [1]}, {"name": "课程", "items": [2]}],
    [{"name": "申请材料", "items": [1]}, {"name": "读博申请", "items": [2]}],
    [{"name": "申请材料", "items": [1]}, {"name": "面试", "items": [1]}],
    [{"name": "申请材料", "items": [1]}, {"name": "与P19的相处", "items": [2]}],
])
def test_split_needs_two_new_topics_covering_every_fact(topics):
    with pytest.raises(ValueError):
        hierarchy.validate_split({"topics": topics}, 2, {"读博申请", "课程"})


def test_unique_name_prefixes_the_original_topic():
    assert hierarchy.unique_name("日常", "课程", set()) == "日常"
    assert hierarchy.unique_name("日常", "课程", {"日常"}) == "课程·日常"
    assert hierarchy.unique_name("日常", "课程", {"日常", "课程·日常"}) == "课程·日常2"


@pytest.mark.parametrize("defect", [None, "missing", "duplicate", "out_of_range", "empty"])
def test_descriptions_cover_every_domain_once(defect):
    items = [{"id": 2, "description": " 说明二 "}, {"id": 1, "description": "说明一"}]
    if defect == "missing":
        items.pop()
    elif defect == "duplicate":
        items[1]["id"] = 2
    elif defect == "out_of_range":
        items[1]["id"] = 3
    elif defect == "empty":
        items[1]["description"] = " "
    if defect is None:
        assert hierarchy.validate_descriptions({"domains": items}, 2) == ["说明一", "说明二"]
    else:
        with pytest.raises(ValueError):
            hierarchy.validate_descriptions({"domains": items}, 2)


def test_main_writes_a_hierarchy_the_review_store_accepts(tmp_path, monkeypatch):
    settings = tmp_path / "settings.jsonc"
    settings.write_text(json.dumps({"security_args": {"storage_mode": "plaintext", "state_dir": str(tmp_path / "state")}}))
    monkeypatch.setenv("WECLONE_CONFIG_PATH", str(settings))
    source = tmp_path / "organized_memories.json"
    source.write_text(json.dumps(organized(), ensure_ascii=False), encoding="utf-8")
    seen = []

    class Runner:
        def __init__(self, args):
            assert args.leaf_facts == 2

        def run(self, tasks, validate):
            results = []
            for task in tasks:
                seen.append(task)
                if task["stage"] == "split_topic":
                    payload = {"topics": [{"name": "院校选择", "items": [1]}, {"name": "申请进度", "items": [2, 3]}]}
                else:
                    assert task["schema"] is DESCRIBE_SCHEMA
                    count = len(task["prompt"].split("、主题（条目数）\n", 1)[1].splitlines())
                    payload = {"domains": [{"id": i, "description": f"新描述{i}"} for i in range(1, count + 1)]}
                results.append(validate(payload, task))
            return results

        def close(self):
            pass

    monkeypatch.setattr(hierarchy, "LLMTasks", Runner)
    output = tmp_path / "hierarchy"
    hierarchy.main(["--input-path", str(source), "--output-dir", str(output), "--config-path", str(settings),
                    "--max-context-tokens", "8192", "--leaf-facts", "2"])
    data = json.loads((output / "profile_hierarchy.json").read_text())
    assert [task["stage"] for task in seen] == ["split_topic", "describe_domains"]
    assert "2 学业：院校选择（1）、申请进度（2）、课程（1）" in seen[1]["prompt"]
    assert data["domains"] == [
        {"name": "出行旅行", "description": "新描述1", "topics": [{"name": "周末郊游", "fact_indices": [4]}]},
        {"name": "学业", "description": "新描述2", "topics": [
            {"name": "院校选择", "fact_indices": [0]},
            {"name": "申请进度", "fact_indices": [1, 2]},
            {"name": "课程", "fact_indices": [3]},
        ]},
    ]
    assert [row["topic"] for row in data["facts"]] == ["院校选择", "申请进度", "申请进度", "课程", "周末郊游"]
    assert sorted(data["sources"]) == ["s1", "s2", "s3", "s4", "s7"]
    assert json.loads((output / "summary.json").read_text())["splits"][0]["topic"] == "读博申请"

    client = TestClient(create_app(tmp_path / "review.sqlite3", output / "profile_hierarchy.json", tmp_path / "static"))
    client.headers["X-WeClone-Request"] = "1"
    password = {"password": "hierarchy-test", "confirmation": "hierarchy-test"}
    assert client.post("/api/auth/setup", json=password).status_code == 200
    profile = client.get("/api/profile").json()
    assert sorted(row["value"] for row in profile["facts"]) == sorted(row["value"] for row in data["facts"])
    names = {node["name"] for node in profile["locations"]}
    assert {"学业", "院校选择", "申请进度", "出行旅行", "周末郊游"} <= names
