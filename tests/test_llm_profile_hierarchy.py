import json

import pytest
from fastapi.testclient import TestClient

from weclone.data.agent import llm_profile_hierarchy as hierarchy
from weclone.prompts.memory_organization import dimension_name
from weclone.prompts.profile_hierarchy import SPLIT_SCHEMA, TOPIC_SCHEMA
from weclone.server.profile_review import create_app


def fact(dim, domain, attr, value, sid, **extra):
    return {"dim": dim, "domain": domain, "attr": attr, "value": value, "source_ids": [sid], **extra}


def organized():
    return {
        "domains": [
            {"name": "出行旅行", "description": "旧描述一"},
            {"name": "学业", "description": "旧描述二"},
            {"name": "工作", "description": "没有画像条目"},
        ],
        "facts": [
            fact(1, "学业", "目标院校", "B申请了甲大学。", "s1"),
            fact(3, "学业", "读博目标", "B想读博。", "s2"),
            fact(5, "学业", "面试安排", "B下周面试。", "s3"),
            fact(1, "学业", "课程", "B在上机器学习课。", "s4"),
            fact(6, None, "关系状态", "P1是B的同学。", "s5", peer="P1"),
            fact(7, "出行旅行", "天气", "周末下雨。", "s6"),
            fact(8, "出行旅行", "活动偏好", "B喜欢徒步。", "s7"),
        ],
        "unused": [],
        "unassigned_ids": [],
        "sources": {f"s{i}": {"id": f"s{i}", "content": f"来源{i}"} for i in range(1, 8)},
    }


@pytest.mark.parametrize("domain", [None, "不存在"])
def test_domain_facts_reject_profile_facts_outside_known_domains(domain):
    data = organized()
    data["facts"][0]["domain"] = domain
    with pytest.raises(ValueError, match="Profile fact 0"):
        hierarchy.domain_facts(data)


def test_group_topics_numbers_each_domain_and_rewrites_its_description():
    data = organized()
    facts, layout = hierarchy.domain_facts(data)
    answers = {
        "出行旅行": [{"name": "周末郊游", "items": [1]}],
        "学业": [{"name": "课程", "items": [4]}, {"name": "读博申请", "items": [3, 1, 2]}],
    }
    prompts = []

    class Runner:
        def run(self, tasks, validate):
            results = []
            for task in tasks:
                assert task["stage"] == "domain_topics" and task["schema"] is TOPIC_SCHEMA
                prompts.append(task["prompt"])
                name = task["prompt"].split("## 当前领域\n", 1)[1].split("：", 1)[0]
                results.append(validate({"description": f" {name}说明 ", "topics": answers[name]}, task))
            return results

    domains = hierarchy.group_topics(data["domains"], layout, facts, Runner(), min_facts=8, max_facts=30)
    assert domains == [
        {"name": "出行旅行", "description": "出行旅行说明", "topics": [{"name": "周末郊游", "fact_indices": [4]}]},
        {"name": "学业", "description": "学业说明", "topics": [
            {"name": "课程", "fact_indices": [3]}, {"name": "读博申请", "fact_indices": [0, 1, 2]},
        ]},
    ]
    assert "不超过 30 条" in prompts[1] and "不少于 8 条" in prompts[1]
    assert "## 当前领域\n学业：旧描述二\n" in prompts[1] and "3. 工作：没有画像条目" in prompts[1]
    assert f"4 课程：B在上机器学习课。（{dimension_name(1)}）" in prompts[1]
    assert prompts[0].split("## 当前领域")[0] == prompts[1].split("## 当前领域")[0]


def test_topics_strip_names_and_keep_items():
    payload = {"topics": [{"name": " 读博申请 ", "items": [2, 1]}, {"name": "日常出行", "items": [3]}]}
    assert hierarchy.validate_topics(payload, 3) == [
        {"name": "读博申请", "items": [2, 1]}, {"name": "日常出行", "items": [3]},
    ]


@pytest.mark.parametrize("defect", ["missing", "duplicate_id", "out_of_range", "duplicate_name", "english_name",
                                    "peer_id", "empty_items"])
def test_topics_cover_every_fact_once(defect):
    topics = [{"name": "读博申请", "items": [1, 2]}, {"name": "日常出行", "items": [3]}]
    if defect == "missing":
        topics.pop()
    elif defect == "duplicate_id":
        topics[1]["items"] = [3, 2]
    elif defect == "out_of_range":
        topics[1]["items"] = [3, 4]
    elif defect == "duplicate_name":
        topics[1]["name"] = "读博申请"
    elif defect == "english_name":
        topics[1]["name"] = "travel"
    elif defect == "peer_id":
        topics[1]["name"] = "和P3出行"
    else:
        topics.append({"name": "空主题", "items": []})
    with pytest.raises(ValueError):
        hierarchy.validate_topics({"topics": topics}, 3)


@pytest.mark.parametrize("defect", [None, "no_description", "blank_description", "extra_key", "missing_fact"])
def test_domain_topics_need_a_description(defect):
    payload = {"description": " 说明 ", "topics": [{"name": "读博申请", "items": [1, 2]}]}
    if defect == "no_description":
        del payload["description"]
    elif defect == "blank_description":
        payload["description"] = " "
    elif defect == "extra_key":
        payload["domains"] = []
    elif defect == "missing_fact":
        payload["topics"][0]["items"] = [1]
    if defect is None:
        assert hierarchy.validate_domain_topics(payload, 2) == {
            "description": "说明", "topics": [{"name": "读博申请", "items": [1, 2]}],
        }
    else:
        with pytest.raises(ValueError):
            hierarchy.validate_domain_topics(payload, 2)


def test_dimension_view_nests_domain_topic_and_attribute():
    facts, _ = hierarchy.domain_facts(organized())
    domains = [
        {"name": "出行旅行", "topics": [{"name": "周末郊游", "fact_indices": [4]}]},
        {"name": "学业", "topics": [
            {"name": "读博申请", "fact_indices": [0, 1, 2]}, {"name": "课程", "fact_indices": [3]},
        ]},
    ]
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
    facts = [fact(1, "学业", "属性", f"事实{i}", f"s{i}") for i in range(7)]
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
    assert hierarchy.split_topics(domains, [fact(1, "学业", "课程", "事实", "s1")], Runner(), 30) == (
        domains,
        [],
    )


@pytest.mark.parametrize("topics", [
    [{"name": "申请材料", "items": [1, 2]}],
    [{"name": "申请材料", "items": [1]}, {"name": "课程", "items": [2]}],
])
def test_split_needs_two_new_topics_covering_every_fact(topics):
    with pytest.raises(ValueError):
        hierarchy.validate_split({"topics": topics}, 2, {"读博申请", "课程"})


def test_unique_name_prefixes_the_original_topic():
    assert hierarchy.unique_name("日常", "课程", set()) == "日常"
    assert hierarchy.unique_name("日常", "课程", {"日常"}) == "课程·日常"
    assert hierarchy.unique_name("日常", "课程", {"日常", "课程·日常"}) == "课程·日常2"


def test_main_writes_a_hierarchy_the_review_store_accepts(tmp_path, monkeypatch):
    settings = tmp_path / "settings.jsonc"
    settings.write_text(json.dumps({"security_args": {"storage_mode": "plaintext", "state_dir": str(tmp_path / "state")}}))
    monkeypatch.setenv("WECLONE_CONFIG_PATH", str(settings))
    source = tmp_path / "organized_memories.json"
    source.write_text(json.dumps(organized(), ensure_ascii=False), encoding="utf-8")
    answers = {
        "出行旅行": [{"name": "周末郊游", "items": [1]}],
        "学业": [{"name": "读博申请", "items": [1, 2, 3]}, {"name": "课程", "items": [4]}],
    }
    seen = []

    class Runner:
        def __init__(self, args):
            assert (args.leaf_facts, args.topic_min_facts) == (2, 1)

        def run(self, tasks, validate):
            results = []
            for task in tasks:
                seen.append(task)
                if task["stage"] == "split_topic":
                    payload = {"topics": [{"name": "院校选择", "items": [1]}, {"name": "申请进度", "items": [2, 3]}]}
                else:
                    name = task["prompt"].split("## 当前领域\n", 1)[1].split("：", 1)[0]
                    payload = {"description": f"{name}的新说明", "topics": answers[name]}
                results.append(validate(payload, task))
            return results

        def close(self):
            pass

    monkeypatch.setattr(hierarchy, "LLMTasks", Runner)
    output = tmp_path / "hierarchy"
    hierarchy.main(["--input-path", str(source), "--output-dir", str(output), "--config-path", str(settings),
                    "--max-context-tokens", "8192", "--leaf-facts", "2", "--topic-min-facts", "1"])
    data = json.loads((output / "profile_hierarchy.json").read_text())
    assert [task["stage"] for task in seen] == ["domain_topics", "domain_topics", "split_topic"]
    assert "## 领域\n学业：学业的新说明\n" in seen[2]["prompt"]
    assert data["domains"] == [
        {"name": "出行旅行", "description": "出行旅行的新说明", "topics": [{"name": "周末郊游", "fact_indices": [4]}]},
        {"name": "学业", "description": "学业的新说明", "topics": [
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
