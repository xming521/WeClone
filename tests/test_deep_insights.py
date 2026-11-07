import json

import pytest

from weclone.data.agent.llm_profile_hierarchy import dimension_view
from weclone.prompts.deep_insights import INSIGHTS_SCHEMA, RULES, insight_limit
from weclone.server import deep_insights
from weclone.server.profile_review import ReviewBatch, ReviewItem, ReviewStore
from weclone.utils import secure_storage

FACTS = [
    {"dim": 1, "attr": "在读身份", "value": "计算机博士生", "sensitivity": "public", "time_scope": "current"},
    {"dim": 1, "attr": "在读身份", "value": "硕士生", "sensitivity": "public", "time_scope": "superseded"},
    {"dim": 4, "attr": "科研补助", "value": "每月三千元", "sensitivity": "private", "time_scope": "current"},
    {"dim": 8, "attr": "口味\n偏好", "value": "爱吃\n辣", "sensitivity": "normal", "time_scope": "stable"},
]


@pytest.fixture
def run(tmp_path, monkeypatch):
    directory = tmp_path / "profile_hierarchy"
    directory.mkdir()
    domains = [
        {"name": "学业", "description": "读博", "topics": [{"name": "读博", "fact_indices": [0, 1, 2]}]},
        {"name": "饮食", "description": "口味", "topics": [{"name": "口味", "fact_indices": [3]}]},
    ]
    facts = [{**fact, "source_ids": []} for fact in FACTS]
    profile = {
        "domains": domains,
        "dimensions": dimension_view(facts, domains),
        "facts": facts,
        "sources": {},
    }
    (directory / "profile_hierarchy.json").write_text(
        json.dumps(profile, ensure_ascii=False), encoding="utf-8"
    )
    asked, answers = [], []

    class Runner:
        def run(self, tasks, validate):
            asked.extend(tasks)
            return [validate(answers.pop(0), task) for task in tasks]

        def close(self):
            pass

    monkeypatch.setattr(deep_insights, "insights_runner", lambda max_context_tokens: Runner())
    return directory, asked, answers


def test_insight_limit_grows_with_the_square_root_of_the_facts():
    assert [insight_limit(n) for n in (1, 100, 600, 1042, 2000)] == [1, 4, 9, 12, 17]


def test_prompt_numbers_the_facts_left_by_domain_and_ends_with_the_cap(run):
    directory, asked, answers = run
    store = ReviewStore(directory / "profile_review.sqlite3", directory / "profile_hierarchy.json")
    rejected = next(fact for fact in store.profile()["facts"] if fact["value"] == "硕士生")
    store.review(
        ReviewBatch(items=[ReviewItem(id=rejected["id"], version=rejected["version"])], status="rejected")
    )
    answers.append({"implicit_facts": [], "insights": []})
    deep_insights.build_insights(directory)
    [task] = asked
    assert task["stage"] == "deep_insights" and task["schema"] is INSIGHTS_SCHEMA
    # The rules lead every prompt unchanged, so model services can cache them.
    assert task["prompt"] == RULES + (
        "\n\n## 学业\n0｜在读身份：计算机博士生（current）\n1｜科研补助：每月三千元（current）"
        "\n\n## 饮食\n2｜口味 偏好：爱吃 辣（stable）"
        "\n\n深度洞察最多写 1 条，优先写能解释最多事实的。"
    )


def test_facts_in_no_domain_come_last(run):
    directory, asked, answers = run
    path = directory / "profile_hierarchy.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["facts"].append({"dim": 8, "attr": "运动", "value": "爱打球", "sensitivity": "normal", "source_ids": []})
    # An attribute straight on its dimension, as one the owner adds there.
    next(item for item in profile["dimensions"] if item["dim"] == 8)["groups"].append(
        {"attr": "运动", "fact_indices": [4]}
    )
    path.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
    answers.append({"implicit_facts": [], "insights": []})
    deep_insights.build_insights(directory)
    assert asked[0]["prompt"].endswith(
        "\n\n## 饮食\n3｜口味 偏好：爱吃 辣（stable）\n\n## 不属于任何板块\n4｜运动：爱打球\n\n深度洞察最多写 1 条，"
        "优先写能解释最多事实的。"
    )


def test_conclusions_keep_their_facts_and_the_strictest_sensitivity(run):
    directory, _, answers = run
    answers.append(
        {
            "implicit_facts": [{"conclusion": " 在读博士 ", "basis": [0, 0]}],
            "insights": [{"conclusion": "钱和口味都将就", "basis": [3, 2]}],
        }
    )
    result = deep_insights.build_insights(directory)
    assert secure_storage.read_json(directory / deep_insights.OUTPUT_NAME) == result
    store = ReviewStore(directory / "profile_review.sqlite3", directory / "profile_hierarchy.json")
    ids = {fact["value"]: fact["id"] for fact in store.profile()["facts"]}
    assert result["implicit_facts"] == [
        {"text": "在读博士", "fact_ids": [ids["计算机博士生"]], "sensitivity": "public"}
    ]
    assert result["insights"] == [
        {"text": "钱和口味都将就", "fact_ids": [ids["每月三千元"], ids["爱吃\n辣"]], "sensitivity": "private"}
    ]


def test_a_fact_without_sensitivity_leaves_the_conclusion_unknown():
    assert deep_insights.strictest([{"sensitivity": "normal"}, {"sensitivity": "sensitive"}]) == "sensitive"
    assert deep_insights.strictest([{"sensitivity": "public"}, {}]) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"insights": []},
        {"implicit_facts": [], "insights": [{"conclusion": "", "basis": [0]}]},
        {"implicit_facts": [], "insights": [{"conclusion": "规律", "basis": []}]},
        {"implicit_facts": [], "insights": [{"conclusion": "规律", "basis": [4]}]},
        {"implicit_facts": [], "insights": [{"conclusion": "规律", "basis": [True]}]},
        {"implicit_facts": [{"conclusion": "身份", "basis": [0], "why": "理由"}], "insights": []},
    ],
)
def test_invalid_answers_are_rejected(payload):
    with pytest.raises(ValueError):
        deep_insights.validate_insights(payload, count=4, limit=2)


def test_insights_past_the_cap_keep_those_resting_on_the_most_facts():
    insights = [
        {"conclusion": "一", "basis": [0]},
        {"conclusion": "二", "basis": [0, 1, 2]},
        {"conclusion": "三", "basis": [1]},
        {"conclusion": "四", "basis": [2, 3]},
    ]
    result = deep_insights.validate_insights({"implicit_facts": [], "insights": insights}, count=4, limit=2)
    assert [row["text"] for row in result["insights"]] == ["二", "四"]


def test_no_facts_means_no_request(run):
    directory, asked, _ = run
    profile = json.loads((directory / "profile_hierarchy.json").read_text(encoding="utf-8"))
    profile.update(facts=[], dimensions=[])
    (directory / "profile_hierarchy.json").write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="没有可用的事实"):
        deep_insights.build_insights(directory)
    assert asked == []
