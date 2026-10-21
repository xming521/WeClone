import json
from datetime import datetime
from pathlib import Path

import pyjson5
import pytest

from weclone.core.inference.llm_client import LLMResponse
from weclone.harness.profile import (
    ChatMessage,
    JevRouter,
    LLMFilter,
    LLMRouter,
    ProfileContextSelector,
    ProfileTree,
    build_profile_context_selector,
    render_state,
)
from weclone.prompts.harness import PROFILE_FILTER_RULES, PROFILE_ROUTE_RULES
from weclone.server.profile_review import FactEdit, ReviewBatch, ReviewItem, ReviewStore
from weclone.utils.config_models import HarnessArgs

NOW = datetime(2024, 6, 1, 12, 0)
TEMPLATE = Path(__file__).resolve().parents[1] / "settings.template.jsonc"


def fact(attr, value, *sources):
    return {"attr": attr, "value": value, "source_ids": list(sources)}


def profile():
    return {
        "domains": [
            {
                "name": "学业",
                "description": "课程和读博",
                "topics": [
                    {"name": "读博申请", "fact_indices": [0, 1]},
                    {"name": "课程", "fact_indices": [2]},
                ],
            },
            {"name": "运动", "description": "锻炼习惯", "topics": [{"name": "跑步", "fact_indices": [3, 4]}]},
        ],
        "facts": [
            fact("目标院校", "B申请了甲大学。", "s1", "s3"),
            fact("面试安排", "B下周面试。", "s4"),
            fact("课程", "B在上机器学习课。", "s2"),
            fact("跑步习惯", "B每周跑三次。"),
            fact("比赛", "B报名了马拉松。", "s5"),
        ],
        "sources": {
            "s1": {"sample_time": "2024-03-02T10:00:00"},
            "s2": {"sample_time": "2024-04-01 09:00:00"},
            "s3": {"sample_time": "2024-05-20T08:00:00"},
            "s4": {"sample_time": "2024-07-01T08:00:00"},
            "s5": {"sample_time": "2024-05-03T08:00:00"},
        },
    }


class FakeClient:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.prompts = []
        self.closed = False

    def generate(self, request):
        self.prompts.append(request.messages[0]["content"])
        payload = self.payloads.pop(0)
        if isinstance(payload, str):
            return LLMResponse(ok=False, error=payload)
        return LLMResponse(ok=True, text=json.dumps(payload), parsed_json=payload)

    def close(self, wait=True, cancel_futures=False):
        self.closed = True


def chat():
    return [
        ChatMessage("user", "最近忙啥", datetime(2024, 6, 1, 9, 0)),
        ChatMessage("assistant", "准备面试", datetime(2024, 6, 1, 9, 5)),
        ChatMessage("user", "周末一起跑步？" + "啊" * 500, datetime(2024, 6, 1, 11, 50)),
    ]


def test_tree_hides_future_facts_and_dates_by_month():
    tree = ProfileTree(profile())
    assert tree.topics["0|0"].text == "读博申请：目标院校、面试安排"
    assert [k for k in range(5) if tree.visible(k, NOW)] == [0, 2, 3, 4]
    assert tree.fact_text(0, NOW) == "目标院校：B申请了甲大学。（2024-03~2024-05）"
    assert tree.fact_text(0, datetime(2024, 4, 1)) == "目标院校：B申请了甲大学。（2024-03）"
    assert tree.fact_text(3, NOW) == "跑步习惯：B每周跑三次。"
    assert tree.tree_text() == (
        "## 学业：课程和读博\n0|0 读博申请：目标院校、面试安排\n0|1 课程：课程\n## 运动：锻炼习惯\n1|0 跑步：跑步习惯、比赛"
    )


def test_tree_requires_domains():
    with pytest.raises(ValueError, match="domains"):
        ProfileTree({"dimensions": [], "facts": []})


def test_tree_applies_review_decisions_when_review_db_exists(tmp_path):
    data = profile()
    for item in data["facts"]:
        item["dim"] = 1
    # The per-dimension view the review store imports: dimension → domain → topic → attribute.
    data["dimensions"] = [
        {
            "dim": 1,
            "name": "人口学与基本身份",
            "groups": [
                {
                    "name": domain["name"],
                    "items": [
                        {
                            "name": topic["name"],
                            "items": [
                                {"attr": data["facts"][k]["attr"], "fact_indices": [k]}
                                for k in topic["fact_indices"]
                            ],
                        }
                        for topic in domain["topics"]
                    ],
                }
                for domain in data["domains"]
            ],
        }
    ]
    path = tmp_path / "profile_hierarchy.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    database = tmp_path / "profile_review.sqlite3"

    plain = ProfileTree.load(path)
    assert len(plain.facts) == 5
    assert not database.exists()  # loading never creates the review DB

    store = ReviewStore(database, path)
    facts = {item["attr"]: item for item in store.profile()["facts"]}
    store.review(
        ReviewBatch(
            items=[ReviewItem(id=facts["比赛"]["id"], version=facts["比赛"]["version"])], status="rejected"
        )
    )
    edited = facts["课程"]
    store.edit(
        edited["id"],
        FactEdit(
            value="B在上深度学习课。",
            attr="课程",
            group_id=edited["group_id"],
            version=edited["version"],
        ),
    )

    tree = ProfileTree.load(path)
    assert [fact["attr"] for fact in tree.facts] == ["目标院校", "面试安排", "课程", "跑步习惯"]
    assert "B在上深度学习课。" in [fact["value"] for fact in tree.facts]
    assert [name for name, _ in tree.domains] == ["学业", "运动"]
    assert [topic.name for topic in tree.topics.values()] == ["读博申请", "课程", "跑步"]
    assert sum(len(topic.facts) for topic in tree.topics.values()) == 4


def test_fill_skips_a_topic_that_does_not_fit():
    tree = ProfileTree(profile())
    # Visible facts: 0|0 has 1, 1|0 has 2, 0|1 has 1. With budget 2, 1|0 no longer fits after 0|0, but 0|1 does.
    assert tree.fill(["0|0", "1|0", "0|1"], NOW, 2) == [0, 2]
    assert tree.fill(["1|0", "0|0", "0|1"], NOW, 2) == [3, 4]


def test_render_state_marks_sessions_and_truncates():
    state = render_state(chat(), NOW, "老王")
    lines = state.splitlines()
    assert lines[:3] == [
        "聊天时间：2024-06-01 12:00",
        "B 给 A 的备注名：老王",
        "上文（最后一条是 A 发的，接下来轮到 B 回复）：",
    ]
    assert lines[3:7] == ["（2024-06-01 09:00）", "A: 最近忙啥", "B: 准备面试", "（2024-06-01 11:50）"]
    assert lines[7].endswith("…") and len(lines[7]) == len("A: ") + 401
    assert "备注名" not in render_state(chat(), NOW)
    with pytest.raises(ValueError):
        render_state(chat()[:2], NOW)


def test_jev_router_ranks_topics_by_domain_times_topic():
    tree = ProfileTree(profile())
    calls = []
    router = JevRouter(tree, "http://jev", top_domains=1)

    def ask(state, questions):
        calls.append(questions)
        if "domain" in questions:
            return {"domain": {"probabilities": {"0": 0.7, "1": 0.3}}}
        return {"d0": {"probabilities": {"0": 0.2, "1": 0.8}}}

    router.ask = ask
    assert router.route("state") == ["0|1", "0|0"]
    assert calls[0]["domain"]["criteria"] == {"0": "学业：课程和读博", "1": "运动：锻炼习惯"}
    assert set(calls[1]) == {"d0"}

    router.top_domains = 2
    calls.clear()
    ranking = router.route("state")
    # The one-topic domain is not asked about and keeps its domain's score.
    assert set(calls[1]) == {"d0"}
    assert ranking == ["0|1", "1|0", "0|0"]


def test_llm_router_validates_and_retries():
    tree = ProfileTree(profile())
    client = FakeClient({"topics": ["9|9"]}, {"topics": ["1|0", "0|0", "1|0"]})
    router = LLMRouter(tree, client, max_topics=8)
    assert router.route("STATE") == ["1|0", "0|0"]
    assert client.prompts[0].startswith(PROFILE_ROUTE_RULES.format(max_topics=8))
    assert client.prompts[0].endswith("STATE")
    assert "未知主题编号" in client.prompts[1]


def test_llm_router_drops_unknown_keys_but_keeps_valid_ones():
    tree = ProfileTree(profile())
    client = FakeClient({"topics": ["1|0", "9|9", "0|0"]})
    assert LLMRouter(tree, client, max_topics=2).route("STATE") == ["1|0", "0|0"]
    assert len(client.prompts) == 1

    client = FakeClient({"topics": ["9|9", "1|0", "0|0", "0|1"]})
    assert LLMRouter(tree, client, max_topics=2).route("STATE") == ["1|0", "0|0"]

    assert LLMRouter(tree, FakeClient({"topics": []})).route("STATE") == []


def test_llm_router_retries_when_every_key_is_unknown():
    tree = ProfileTree(profile())
    client = FakeClient({"topics": ["9|9", "8|8"]}, {"topics": ["1|0"]})
    assert LLMRouter(tree, client).route("STATE") == ["1|0"]
    assert "未知主题编号" in client.prompts[1]

    client = FakeClient({"topics": ["9|9"]}, {"topics": ["9|9"]})
    with pytest.raises(RuntimeError, match="harness_profile_route"):
        LLMRouter(tree, client).route("STATE")


def test_llm_prompts_keep_a_fixed_prefix():
    tree = ProfileTree(profile())
    client = FakeClient({"topics": ["0|0"]}, {"topics": ["0|0"]})
    router = LLMRouter(tree, client)
    router.route(render_state(chat(), NOW))
    router.route(render_state(chat()[:1], datetime(2025, 1, 1)))
    a, b = client.prompts
    prefix = a[: a.index("聊天时间")]
    assert b.startswith(prefix) and tree.tree_text() in prefix

    client = FakeClient({"must": [], "helpful": []})
    LLMFilter(tree, client).pick("STATE", [0], NOW)
    assert client.prompts[0].startswith(PROFILE_FILTER_RULES + "\n## 聊天\nSTATE")


def test_filter_validates_ids_and_tiers():
    tree = ProfileTree(profile())
    client = FakeClient({"must": ["F2"], "helpful": ["F2", "F0"]})
    assert LLMFilter(tree, client).pick("state", [0, 2], NOW) == ([2], [0])
    assert "F0 目标院校：B申请了甲大学。（2024-03~2024-05）" in client.prompts[0]

    with pytest.raises(RuntimeError, match="harness_profile_filter"):
        LLMFilter(tree, FakeClient("timeout", "timeout")).pick("state", [0], NOW)


def test_filter_drops_unknown_ids_but_keeps_valid_ones():
    tree = ProfileTree(profile())
    client = FakeClient({"must": ["F7", "F2"], "helpful": ["F9", "F0"]})
    assert LLMFilter(tree, client).pick("state", [0, 2], NOW) == ([2], [0])
    assert len(client.prompts) == 1

    client = FakeClient({"must": [], "helpful": []})
    assert LLMFilter(tree, client).pick("state", [0, 2], NOW) == ([], [])


def test_filter_retries_when_every_id_is_unknown():
    tree = ProfileTree(profile())
    client = FakeClient({"must": ["F7"], "helpful": ["F9"]}, {"must": ["F2"], "helpful": []})
    assert LLMFilter(tree, client).pick("state", [0, 2], NOW) == ([2], [])
    assert "F7" in client.prompts[1]

    client = FakeClient({"must": ["F7"], "helpful": []}, {"must": [], "helpful": ["F7"]})
    with pytest.raises(RuntimeError, match="harness_profile_filter"):
        LLMFilter(tree, client).pick("state", [0, 2], NOW)


class StubRouter:
    def __init__(self, ranking=None, error=None):
        self.ranking, self.error = ranking, error

    def route(self, state):
        if self.error:
            raise self.error
        return self.ranking


def test_selector_end_to_end():
    tree = ProfileTree(profile())
    client = FakeClient({"must": ["F3"], "helpful": ["F4"]})
    selector = ProfileContextSelector(tree, StubRouter(["1|0", "0|0"]), LLMFilter(tree, client), 3, client)
    context = selector.select(chat(), NOW, "老王")
    assert context.error is None
    assert context.candidates == [0, 3, 4]
    assert [(f.index, f.tier) for f in context.facts] == [(3, "must"), (4, "helpful")]
    assert context.text() == "跑步习惯：B每周跑三次。\n比赛：B报名了马拉松。（2024-05）"
    assert set(context.timings_ms) == {"route", "filter"}
    selector.close()
    assert client.closed


def test_selector_degrades_without_facts():
    tree = ProfileTree(profile())
    client = FakeClient()
    selector = ProfileContextSelector(
        tree, StubRouter(error=TimeoutError("jev down")), LLMFilter(tree, client)
    )
    context = selector.select(chat(), NOW)
    assert context.facts == [] and context.error == "TimeoutError: jev down"

    # Nothing visible to hand over: no filter call.
    context = ProfileContextSelector(tree, StubRouter(["0|0"]), LLMFilter(tree, client)).select(
        chat(), datetime(2024, 1, 1)
    )
    assert context.candidates == [] and context.facts == [] and client.prompts == []


def test_config_defaults_and_builder(tmp_path, monkeypatch):
    args = HarnessArgs.model_validate({}).profile_context
    assert (args.router, args.fact_budget, args.llm_provider) == ("jev", 30, "codex_exec")
    with pytest.raises(ValueError):
        HarnessArgs.model_validate({"profile_context": {"router": "embedding"}})

    template = pyjson5.loads(TEMPLATE.read_text(encoding="utf-8"))
    assert HarnessArgs.model_validate(template["harness_args"]).profile_context == args

    path = tmp_path / "profile_hierarchy.json"
    path.write_text(json.dumps(profile(), ensure_ascii=False), encoding="utf-8")
    settings = tmp_path / "settings.jsonc"
    settings.write_text(
        json.dumps(
            {
                "codex_exec_args": {
                    "model": "m",
                    "effort": "medium",
                    "command": "codex",
                    "sandbox": "read-only",
                },
                "harness_args": {
                    "profile_context": {"profile_path": str(path), "router": "llm", "fact_budget": 5}
                },
            }
        ),
        encoding="utf-8",
    )
    built = {}

    def fake_build(provider, **kwargs):
        built.update(kwargs, provider=provider)
        return FakeClient()

    monkeypatch.setattr("weclone.core.inference.llm_client.build_llm_client", fake_build)
    # configure() points WECLONE_CONFIG_PATH at the test settings; setenv restores it afterwards.
    monkeypatch.setenv("WECLONE_CONFIG_PATH", str(settings))
    selector = build_profile_context_selector(settings)
    assert isinstance(selector.router, LLMRouter) and selector.fact_budget == 5
    assert (built["provider"], built["model"], built["effort"], built["isolated_cwd"]) == (
        "codex_exec",
        "m",
        "medium",
        True,
    )
    assert selector.router.client is selector.filter.client is selector.client
