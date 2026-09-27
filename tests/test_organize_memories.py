import json
from types import SimpleNamespace

import numpy as np
import pytest

from weclone.core.inference.llm_client import LLMResponse
from weclone.data.agent import organize_memories as organize
from weclone.prompts import memory_organization as prompts


def test_source_statistics_weights_chats_equally_and_separates_peers():
    sources = {
        "a": {"peer": "p1", "sample": "one", "confidence": 3, "importance": 2},
        "b": {"peer": "p1", "sample": "one", "confidence": 3, "importance": 4},
        "c": {"peer": "p2", "sample": "one", "confidence": 4, "importance": 4},
    }
    assert organize.source_statistics(["a", "b", "c"], sources) == {
        "support_count": 2, "source_confidence_mean": 3.5, "source_importance_mean": 3.5,
    }


@pytest.mark.parametrize("dim", list(prompts.DIMENSIONS))
@pytest.mark.parametrize("reduced", [False, True])
def test_summary_injects_only_current_dimension_time_rule(dim, reduced):
    rows = ([{"attr": "属性", "value": "事实", "source_ids": ["a"]}] if reduced
            else [{"id": "a", "content": "事实", "sample_time": "2025-01-01"}])
    tasks = [{"dim": dim, "attrs": [attr], "source_ids": ["a"], "rows": rows,
              "stage": "reduce" if reduced else "summarize"} for attr in ("属性", "同义属性")]
    for prompt in (prompts.summarize_prompt(dim, ["属性"], rows, reduced=reduced),
                   prompts.batch_summary_prompt(tasks)):
        assert f"当前维度{dim}的时间规则：" in prompt
        assert prompt.count(prompts.DIMENSION_TIME_RULES[dim]) == 1
        assert all(rule not in prompt for other, rule in prompts.DIMENSION_TIME_RULES.items() if other != dim)


def record(rid, kind="S", memory_type="stable_fact", sample="one", tags=None):
    return {
        "id": rid, "sample": sample, "peer": "peer1", "kind": kind,
        "sample_time": "2025-01-01", "type": memory_type,
        "content": f"B的记忆 {rid}", "tags": tags or [], "origins": [], "importance": 2, "confidence": 3,
    }


@pytest.mark.parametrize("content,expected_content", [
    ("B从事算法研发。", "B从事算法研发。"),
    ("B研究AI和API。", "B研究AI和API。"),
    ("A是B的同事。", "peer1是B的同事。"),
    ("B与A约定见面。", "B与peer1约定见面。"),
    ("Ａ与B讨论ＡＩ。", "peer1与B讨论ＡＩ。"),
    ("A与B讨论A股和Ａ股。", "peer1与B讨论A股和Ａ股。"),
])
def test_prompt_record_limits_metadata_without_changing_source(content, expected_content):
    source = {**record("x"), "content": content, "status": "ongoing",
              "event_time": {"time_text": "明天"}, "people": ["B"], "event_types": ["daily_event"]}
    original = json.dumps(source)
    expected = {"id": "x", "content": expected_content, "status": "ongoing", "event_time": {"time_text": "明天"}}
    assert organize.prompt_record(source) == expected
    assert organize.prompt_record(source, include_sample_time=True) == {**expected, "sample_time": "2025-01-01"}
    assert json.dumps(source) == original
    assert organize.prompt_record(record("basic")) == {"id": "basic", "content": "B的记忆 basic"}


def write_input(root):
    sample = {
        "id": "sample1", "local_id": "0", "chat_with_id": "peer1", "time": "2025-01-01",
        "messages": [{"role": "assistant", "content": "RAW_CHAT_MUST_NOT_ENTER_PROMPTS"}],
    }
    profile = {**sample, "state_memories": [
        {"type": "goal", "content": "B计划周末去郊游。", "tags": ["出行旅行"], "importance": 2, "confidence": 3},
    ]}
    event = {**sample, "event_memories": {"surface_events": [
        {"surface_event": "B与朋友约定周末郊游。", "event_types": ["commitment_event"], "tags": ["出行旅行"],
         "importance": 2, "confidence": 3},
    ]}}
    for folder, value in (("state_people", profile), ("event_people", event)):
        directory = root / folder
        directory.mkdir(parents=True)
        (directory / "peer.json").write_text(json.dumps([value], ensure_ascii=False))
    return sample, profile


@pytest.mark.parametrize("kind", ["S", "ES", "EI"])
def test_reader_filters_scores_without_modifying_sources(tmp_path, kind):
    sample, _ = write_input(tmp_path)
    content_key = {"S": "content", "ES": "surface_event", "EI": "inferred_event"}[kind]
    memories = [{content_key: f"记忆{i}", "type": "stable_fact", "importance": importance, "confidence": confidence}
                for i, (importance, confidence) in enumerate([(1, 4), (4, 2), (2, 3), (4, 4)])]
    folder = "state_people" if kind == "S" else "event_people"
    payload = {"state_memories": memories} if kind == "S" else {
        "event_memories": {"surface_events" if kind == "ES" else "inferred_events": memories}}
    path = tmp_path / folder / "peer.json"
    path.write_text(json.dumps([{**sample, **payload}], ensure_ascii=False))
    before = path.read_bytes()
    rows = [row for row in organize.read_memories(tmp_path) if row["kind"] == kind]
    assert [row["content"] for row in rows] == ["记忆2", "记忆3"]
    assert [(row["importance"], row["confidence"]) for row in rows] == [(2, 3), (4, 4)]
    assert [row["origins"][0]["memory_index"] for row in rows] == [2, 3]
    assert path.read_bytes() == before


def test_reader_excludes_raw_chat_deduplicates_auxiliary_and_retains_origins(tmp_path):
    _, profile = write_input(tmp_path)
    (tmp_path / "state_people" / "duplicate.json").write_text(json.dumps([profile]))
    rows = organize.read_memories(tmp_path)
    assert len(rows) == 2
    assert "RAW_CHAT" not in json.dumps(rows)
    assert len(next(row for row in rows if row["kind"] == "S")["origins"]) == 2
    other = {**profile, "chat_with_id": "peer2"}
    (tmp_path / "state_people" / "other.json").write_text(json.dumps([other]))
    assert len(organize.read_memories(tmp_path)) == 3


@pytest.mark.parametrize("kind,type_,dims", [
    ("S", "stable_fact", (1, 3, 4, 6)),
    ("S", "goal", (1, 3, 4, 5, 6)),
    ("S", "current_state", (1, 3, 4, 6)),
    ("S", "preference", (8,)),
    ("ES", "ES", (5, 7)),
    ("EI", "EI", (5, 7)),
])
def test_dimension_source_contract(kind, type_, dims):
    assert organize.allowed_dimensions(record("x", kind, type_)) == dims


def test_preference_reader_preserves_ids_scores_and_origins_without_reading_events(tmp_path):
    _, profile = write_input(tmp_path)
    profile["state_memories"].extend([
        {"type": "preference", "content": text, "importance": importance, "confidence": confidence}
        for text, importance, confidence in [("B喜欢徒步。", 2, 3), ("B不喜欢登山。", 3, 4),
                                             ("B偏好独处。", 1, 4), ("B喜欢热闹。", 3, 2)]
    ])
    organize.save(tmp_path / "state_people" / "peer.json", [profile])
    expected = [row for row in organize.read_memories(tmp_path) if row["type"] == "preference"]
    (tmp_path / "event_people" / "peer.json").write_text("event input must not be read")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    actual = organize.read_memories(tmp_path, preferences_only=True)
    assert actual == expected
    assert [row["content"] for row in actual] == ["B喜欢徒步。", "B不喜欢登山。"]
    assert [row["origins"][0]["memory_index"] for row in actual] == [1, 2]
    assert all(p.read_bytes() == content for p, content in before.items())


def test_preference_reader_reports_no_matching_input(tmp_path):
    write_input(tmp_path)
    with pytest.raises(ValueError, match="No extracted memories match"):
        organize.read_memories(tmp_path, preferences_only=True)


@pytest.mark.parametrize("kind,type_", [("S", "stable_fact"), ("S", "goal"), ("ES", "ES")])
def test_preference_dimension_rejects_other_memory_types(kind, type_):
    with pytest.raises(ValueError, match="Invalid dimension/source"):
        organize.validate_classification({"items": [{"id": "a", "fields": [[8, "活动偏好"]]}]},
                                         [record("a", kind, type_)])


def test_preference_pipeline_only_sends_preferences_and_retains_sources(tmp_path, monkeypatch):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    _, profile = write_input(input_dir)
    profile["state_memories"].extend([
        {"type": "preference", "content": text, "importance": 2, "confidence": 3}
        for text in ["B喜欢与聊天对象一起徒步。", "B偏好与聊天对象结伴徒步。"]
    ])
    organize.save(input_dir / "state_people" / "peer.json", [profile])
    seen_prompts = []

    class Embeddings:
        def __init__(self, args):
            pass

        def get(self, texts):
            assert all("郊游" not in text for text in texts)
            return [[1, 0] for _ in texts]

        def close(self):
            pass

    class Runner:
        def __init__(self, args):
            pass

        def run(self, tasks, validate):
            results = []
            for task in tasks:
                seen_prompts.append(task["prompt"])
                if task["stage"] == "attributes":
                    payload = {"items": [{"id": rid, "attrs": [attr]}
                                         for rid, attr in zip(task["ids"], ["活动偏好", "休闲活动偏好"])]}
                else:
                    assert task["stage"] in {"summarize", "reduce"}
                    payload = {"facts": [{"attr": "活动偏好", "value": "B喜欢与P1结伴徒步。",
                                          "source_ids": task["source_ids"]}]}
                results.append(validate(payload, task))
            return results

        def close(self):
            pass

    monkeypatch.setattr(organize, "Embeddings", Embeddings)
    monkeypatch.setattr(organize, "LLMTasks", Runner)
    organize.main(["--preferences-only", "--input-dir", str(input_dir), "--output-dir", str(output_dir),
                   "--max-context-tokens", "8192"])
    output = organize.load(output_dir / "organized_memories.json")
    assert len(output["facts"]) == 1
    fact = output["facts"][0]
    assert fact["dim"] == 8
    assert set(fact["source_ids"]) == set(output["sources"])
    assert len(output["sources"]) == 2
    assert {row["type"] for row in output["sources"].values()} == {"preference"}
    assert fact["support_count"] == 1
    assert {task["stage"] for task in organize.load(output_dir / "classification_tasks.json")} == {"attributes"}
    groups = organize.load(output_dir / "attribute_groups.json")
    assert len(groups) == 1
    assert set(groups[0]["attrs"]) == {"活动偏好", "休闲活动偏好"}
    assert all("郊游" not in prompt and "RAW_CHAT" not in prompt for prompt in seen_prompts)
    assert all('"peer":"P1"' in prompt for prompt in seen_prompts)


@pytest.mark.parametrize("preferences_only", [False, True])
def test_preference_cli_uses_separate_default_output(monkeypatch, preferences_only):
    captured = []
    monkeypatch.setattr(organize, "run", lambda args: captured.append(args))
    organize.main(["--preferences-only"] if preferences_only else [])
    assert captured[0].output_dir.name == (
        "memory_organization_preferences" if preferences_only else "memory_organization"
    )


def test_mixed_prepare_separates_preference_naming_from_dimension_classification(tmp_path, monkeypatch):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    _, profile = write_input(input_dir)
    profile["state_memories"].append(
        {"type": "preference", "content": "B喜欢徒步。", "importance": 2, "confidence": 3}
    )
    organize.save(input_dir / "state_people" / "peer.json", [profile])

    class Embeddings:
        def __init__(self, args):
            pass

        def get(self, texts):
            return [[1, 0] for _ in texts]

        def close(self):
            pass

    monkeypatch.setattr(organize, "Embeddings", Embeddings)
    organize.main(["--stage", "prepare", "--input-dir", str(input_dir), "--output-dir", str(output_dir),
                   "--max-context-tokens", "8192"])
    records = organize.load(output_dir / "records.json")
    tasks = organize.load(output_dir / "classification_tasks.json")
    assert len(records) == 3
    assert {rid for task in tasks if task["stage"] == "classify" for rid in task["ids"]} == {
        row["id"] for row in records if row["type"] != "preference"
    }
    assert {rid for task in tasks if task["stage"] == "attributes" for rid in task["ids"]} == {
        row["id"] for row in records if row["type"] == "preference"
    }


def test_preference_attributes_fill_fixed_dimension_and_allow_multiple_attributes():
    rows = [record("a", memory_type="preference"), record("b", memory_type="preference")]
    payload = {"items": [{"id": "a", "attrs": ["活动偏好", "相处方式偏好"]}, {"id": "b", "attrs": []}]}
    assert organize.validate_preference_attributes(payload, rows) == [
        {"id": "a", "fields": [[8, "活动偏好"], [8, "相处方式偏好"]]}, {"id": "b", "fields": []},
    ]


@pytest.mark.parametrize("items", [
    [{"id": "unknown", "attrs": ["活动偏好"]}],
    [{"id": "a", "attrs": ["活动偏好", "活动偏好"]}],
    [{"id": "a", "attrs": [""]}],
    [{"id": "a", "attrs": "活动偏好"}],
    [{"id": "a", "attrs": ["活动偏好"], "dim": 1}],
    [],
])
def test_preference_attributes_reject_invalid_fields_and_references(items):
    with pytest.raises(ValueError):
        organize.validate_preference_attributes({"items": items}, [record("a", memory_type="preference")])


@pytest.mark.parametrize("configured, expected", [(None, 95000), (160000, 152000)])
def test_context_window_uses_codex_effective_window_not_model_maximum(tmp_path, monkeypatch, configured, expected):
    config = tmp_path / "settings.jsonc"
    config.write_text(json.dumps({"agent_distill_args": {
        "llm_provider": "codex_exec", "model": "test-model", "command": "codex"
    }}))
    catalog = {"models": [
        {"slug": "other-model", "max_context_window": 1000000},
        {"slug": "test-model", "context_window": 100000, "max_context_window": 200000,
         "effective_context_window_percent": 95},
    ]}

    async def effective_config(command):
        return {"model_context_window": configured}

    monkeypatch.setattr(organize, "codex_config", effective_config)
    monkeypatch.setattr(organize.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps(catalog)))
    args = SimpleNamespace(max_context_tokens=None, llm_provider=None, config_path=config,
                           token_encoding="o200k_base", tokenizer_file=None)
    budget = organize.input_budget(args)
    assert args.max_context_tokens == expected
    assert budget.max_tokens == expected
    assert budget.fits("这是一段超过旧字符上限的文本。" * 2000)
    args.max_context_tokens = 10000
    assert organize.input_budget(args).max_tokens == 10000


def test_unknown_model_requires_explicit_context_window(tmp_path, monkeypatch):
    config = tmp_path / "settings.jsonc"
    config.write_text(json.dumps({"agent_distill_args": {
        "llm_provider": "codex_exec", "model": "unknown-model", "command": "codex"
    }}))
    async def effective_config(command):
        return {}

    monkeypatch.setattr(organize, "codex_config", effective_config)
    monkeypatch.setattr(organize.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout='{"models":[]}'))
    args = SimpleNamespace(max_context_tokens=None, llm_provider=None, config_path=config)
    with pytest.raises(ValueError, match="--max-context-tokens"):
        organize.resolve_context_window(args)


def test_local_budget_does_not_override_codex_context_configuration(tmp_path, monkeypatch):
    from weclone.core.inference import llm_client

    config = tmp_path / "settings.jsonc"
    config.write_text("{}")
    options = SimpleNamespace(config_path=config, llm_provider="codex_exec", model="test-model", batch_size=30,
                              timeout=120, effort="low", codex_command="codex", codex_sandbox="read-only")
    monkeypatch.setattr(organize.distill_profile, "resolve_llm_args", lambda args: options)
    captured = {}
    monkeypatch.setattr(llm_client, "build_llm_client", lambda provider, **kwargs: captured.update(kwargs))
    args = SimpleNamespace(max_context_tokens=200000, llm_provider="codex_exec", config_path=config,
                           token_encoding="o200k_base", tokenizer_file=None, output_dir=tmp_path)
    runner = organize.LLMTasks(args)
    assert not captured.get("extra_args")
    assert runner.budget.max_tokens == 200000
    assert captured["max_workers"] == 30


def test_token_budget_splits_classification_even_when_characters_fit():
    rows = [record("a"), record("b", sample="two")]
    counter = organize.InputBudget(8192, "o200k_base")
    limit = max(counter.count(organize.classify_prompt(
        [organize.prompt_record(row)], organize.allowed_dimensions(row),
    )) for row in rows)
    budget = organize.InputBudget(limit, "o200k_base")
    batches = organize.classification_batches(
        rows, [[1, 0], [1, 0]], max_records=20, neighbors=8, budget=budget,
    )
    assert [len(batch) for batch in batches] == [1, 1]
    budget.max_tokens = limit - 1
    with pytest.raises(ValueError, match="budget"):
        organize.classification_batches(
            rows, [[1, 0], [1, 0]], max_records=20, neighbors=8, budget=budget,
        )


@pytest.mark.parametrize("reduced", [False, True])
def test_token_budget_splits_summary_and_cross_batch_merge(reduced):
    rows = ([{"attr": "职业", "value": "B从事算法工作。", "source_ids": [rid]} for rid in ("a", "b")]
            if reduced else [organize.prompt_record(record(rid)) for rid in ("a", "b")])
    counter = organize.InputBudget(8192, "o200k_base")
    limit = max(counter.count(organize.summarize_prompt(1, ["职业"], [row], reduced=reduced)) for row in rows)
    budget = organize.InputBudget(limit, "o200k_base")
    tasks = organize.pack_summary(1, ["职业"], rows, reduced=reduced, budget=budget)
    assert [task["source_ids"] for task in tasks] == [["a"], ["b"]]
    assert all(budget.fits(task["prompt"]) for task in tasks)


def test_llm_checks_input_budget_before_sending_or_loading_cache():
    runner = organize.LLMTasks.__new__(organize.LLMTasks)
    runner.budget = organize.InputBudget(1, "o200k_base")
    # Neither cache nor client is needed: oversized prompts must fail first.
    with pytest.raises(ValueError, match="exceeds input budget"):
        runner.run([{"prompt": "超过一个 token 的完整提示词", "stage": "classify"}], lambda *args: None)


def test_classification_batches_use_tag_and_vector_union_without_losing_singletons():
    rows = [record("a", tags=["读博"]), record("b", sample="two", tags=["读博"]),
            record("c", sample="three", tags=["博士申请"]),
            record("event", "ES", "ES", "four"), record("d", sample="five")]
    vectors = [[1, 0], [0, 1], [0.99, 0.01], [1, 0], [-1, 0]]
    batches = organize.classification_batches(rows, vectors, max_records=3, neighbors=1)
    assert {row["id"] for row in batches[0]} == {"a", "b", "c"}
    assert sorted(row["id"] for batch in batches for row in batch) == sorted(row["id"] for row in rows)
    for batch in batches:
        assert len({organize.allowed_dimensions(row) for row in batch}) == 1


@pytest.mark.parametrize("defect", ["missing", "unknown", "duplicate", "wrong_source", "removed_dim", "english_attr", "bool_dim"])
def test_classification_rejects_lost_ids_and_wrong_source_dimensions(defect):
    rows = [record("s"), record("e", "ES", "ES")]
    payload = {"items": [{"id": "s", "fields": [[1, "职业"]]}, {"id": "e", "fields": [[5, "约定"]]}]}
    if defect == "missing":
        payload["items"].pop()
    elif defect == "unknown":
        payload["items"][1]["id"] = "unknown"
    elif defect == "duplicate":
        payload["items"][1]["id"] = "s"
    elif defect == "wrong_source":
        payload["items"][1]["fields"] = [[1, "职业"]]
    elif defect == "removed_dim":
        payload["items"][1]["fields"] = [[2, "人生经历"]]
    elif defect == "english_attr":
        payload["items"][0]["fields"] = [[1, "occupation"]]
    else:
        payload["items"][0]["fields"] = [[True, "职业"]]
    with pytest.raises(ValueError):
        organize.validate_classification(payload, rows)


def test_open_attributes_multi_assignment_and_empty_result_are_supported():
    rows = [record("a"), record("b")]
    payload = {"items": [
        {"id": "a", "fields": [[1, "职业"], [4, "照护可用时间"]]},
        {"id": "b", "fields": []},
    ]}
    assert organize.validate_classification(payload, rows) == payload["items"]


@pytest.mark.parametrize("reverse", [False, True])
def test_attribute_clustering_connects_chains_within_dimension(reverse):
    assignments = [
        {"id": "a", "fields": [[1, "甲属性"]]},
        {"id": "b", "fields": [[1, "乙属性"]]},
        {"id": "c", "fields": [[1, "丙属性"]]},
        {"id": "d", "fields": [[6, "甲属性"]]},
        {"id": "e", "fields": [[1, "甲属性"]]},
        {"id": "f", "fields": [[1, "丁属性"]]},
        {"id": "a", "fields": [[1, "乙属性"]]},
    ]
    angles = {"甲属性": 0, "乙属性": 30, "丙属性": 60, "丁属性": 180}

    def embed(names):
        return [[np.cos(np.deg2rad(angles[name])), np.sin(np.deg2rad(angles[name]))] for name in names]

    groups = organize.attribute_groups(assignments[::-1] if reverse else assignments, embed, threshold=0.8,
                                       by_id={item["id"]: record(item["id"]) for item in assignments})
    assert len([group for group in groups if group["dim"] == 1]) == 2
    assert next(group for group in groups if group["dim"] == 6)["source_ids"] == ["d"]
    connected = next(group for group in groups if "b" in group["source_ids"])
    assert connected["attrs"] == sorted(["甲属性", "乙属性", "丙属性"])
    assert connected["source_ids"] == ["a", "b", "c", "e"]
    assert next(group for group in groups if "f" in group["source_ids"])["source_ids"] == ["f"]


def test_relation_groups_collect_each_peer_without_embedding_attributes():
    rows = {rid: {**record(rid), "peer": peer} for rid, peer in [("a", "P1"), ("b", "P1"), ("c", "P2")]}
    assignments = [
        {"id": "a", "fields": [[6, "关系状态"], [6, "互动方式"]]},
        {"id": "b", "fields": [[6, "导师关系"]]},
        {"id": "c", "fields": [[6, "关系状态"]]},
    ]

    def embed(names):
        pytest.fail("Relation attributes must not use embeddings")

    groups = organize.attribute_groups(assignments, embed, threshold=0.83, by_id=rows)
    assert groups == [
        {"dim": 6, "peer": "P1", "attrs": sorted(["关系状态", "互动方式", "导师关系"]), "source_ids": ["a", "b"]},
        {"dim": 6, "peer": "P2", "attrs": ["关系状态"], "source_ids": ["c"]},
    ]


@pytest.mark.parametrize("split", [False, True])
def test_relation_summary_isolates_peers_through_split_and_reduce(split):
    rows = {f"{peer}-{i}": {**record(f"{peer}-{i}"), "peer": peer,
                          "content": "B与聊天对象是同事，王老师是B的导师。" * (150 if split else 1)}
            for peer in ("P1", "P2") for i in range(3)}
    groups = [{"dim": 6, "peer": peer, "attrs": ["关系状态"],
               "source_ids": [rid for rid, row in rows.items() if row["peer"] == peer]} for peer in ("P1", "P2")]
    budget = organize.InputBudget(8192, "o200k_base")
    if split:
        budget.max_tokens = max(budget.count(prompts.summarize_prompt(
            6, ["关系状态"], [organize.prompt_record(row, include_sample_time=True)], peer=row["peer"],
        )) for row in rows.values())
    stages = []

    class Runner:
        def run(self, tasks, validate):
            results = []
            for task in tasks:
                assert "members" not in task
                peers = {rows[sid]["peer"] for sid in task["source_ids"]}
                assert len(peers) == 1
                peer = peers.pop()
                assert f'peer："{peer}"' in task["prompt"]
                budget.check(task["prompt"])
                stages.append(task["stage"])
                results.append(validate({"facts": [
                    {"attr": "关系状态", "value": f"B与{peer}是同事。", "source_ids": task["source_ids"]},
                ]}, task))
            return results

    results = organize.summarize_groups(groups, rows, Runner(), budget=budget)
    assert ("reduce" in stages) == split
    if not split:
        assert stages == ["summarize", "summarize"]
    for group, result in zip(groups, results):
        assert result["facts"][0]["source_ids"] == group["source_ids"]


@pytest.mark.parametrize("defect", [None, "wrong_peer", "mixed_sources"])
def test_relation_group_validation_distinguishes_same_attribute_across_peers(tmp_path, monkeypatch, defect):
    rows = [{**record(rid), "peer": peer} for rid, peer in [("a", "P1"), ("b", "P2")]]
    organize.save(tmp_path / "records.json", rows)
    organize.save(tmp_path / "classifications.json", [{"id": row["id"], "fields": [[6, "关系状态"]]} for row in rows])
    groups = [{"dim": 6, "peer": row["peer"], "attrs": ["关系状态"], "source_ids": [row["id"]]} for row in rows]
    if defect == "wrong_peer":
        groups[0]["peer"] = "P2"
    elif defect == "mixed_sources":
        groups[0]["source_ids"].append("b")
    organize.save(tmp_path / "attribute_groups.json", groups)

    class Runner:
        def __init__(self, args):
            pass

        def run(self, tasks, validate):
            return [validate({"facts": [{"attr": "关系状态", "value": "关系事实", "source_ids": task["source_ids"]}]}, task)
                    for task in tasks]

        def close(self):
            pass

    monkeypatch.setattr(organize, "LLMTasks", Runner)
    args = SimpleNamespace(stage="summarize", dry_run=False, preferences_only=False,
                           output_dir=tmp_path, config_path=tmp_path / "config",
                           embedding_url=None, no_auto_start_embedding_service=True, embedding_batch_size=None,
                           max_context_tokens=8192, token_encoding="o200k_base", tokenizer_file=None,
                           max_summary_groups=30, max_summary_records=300)
    if defect:
        with pytest.raises(ValueError, match="does not match classifications"):
            organize.run(args)
    else:
        organize.run(args)
        output = organize.load(tmp_path / "organized_memories.json")
        assert [fact["source_ids"] for fact in output["facts"]] == [["a"], ["b"]]
        assert output["unused"] == []


@pytest.mark.parametrize("defect", ["unknown", "empty", "duplicate"])
def test_summary_rejects_invalid_references(defect):
    result = {"facts": [{"attr": "职业", "value": "B从事算法工作。", "source_ids": ["a"]}]}
    if defect == "unknown":
        result["facts"][0]["source_ids"] = ["fake"]
    elif defect == "empty":
        result["facts"][0]["source_ids"] = []
    else:
        result["facts"][0]["source_ids"] = ["a", "a"]
    with pytest.raises(ValueError):
        organize.validate_summary(result, {"a", "b"})


def test_summary_allows_unreferenced_inputs_without_extra_output_fields():
    result = {"facts": [{"attr": "职业", "value": "B从事算法工作。", "source_ids": ["a"]}]}
    assert organize.validate_summary(result, {"a", "b"}) == result
    assert organize.validate_summary({"facts": []}, {"a"}) == {"facts": []}
    assert "unused_ids" not in json.dumps(organize.SUMMARY_SCHEMA)
    assert "unused_ids" not in json.dumps(organize.BATCH_SUMMARY_SCHEMA)


def test_llm_retry_truncation_and_content_addressed_resume(tmp_path):
    runner = organize.LLMTasks.__new__(organize.LLMTasks)
    runner.path, runner.entries = tmp_path / "checkpoint.json", {}
    runner.budget = organize.InputBudget(8192, "o200k_base")
    runner.options = SimpleNamespace(
        batch_size=2, llm_provider="api", model=None, effort=None, max_tokens=200, timeout=10, overwrite=False,
    )
    payload = {"items": [{"id": "a", "fields": [[1, "职业"]]}]}

    class Client:
        calls = 0

        def generate_batch(self, requests):
            requests = list(requests)
            assert all(request.json_schema == organize.CLASSIFY_SCHEMA for request in requests)
            self.calls += 1
            return [LLMResponse(ok=True, parsed_json=payload, finish_reason="length" if self.calls == 1 else "stop") for _ in requests]

    runner.client = Client()
    tasks = [{"prompt": "classify a", "stage": "classify"}]
    def validator(result, task):
        return organize.validate_classification(result, [record("a")])

    runner.run(tasks, validator)
    assert runner.client.calls == 2
    runner.entries = organize.load(runner.path)
    runner.run(tasks, validator)
    assert runner.client.calls == 2
    runner.run([{**tasks[0], "prompt": "updated rules: classify a"}], validator)
    assert runner.client.calls == 3


@pytest.mark.parametrize("change", ["comment", "extraction", "execution", "model", "overwrite"])
def test_llm_resume_ignores_configuration_changes(tmp_path, monkeypatch, change):
    from weclone.core.inference import llm_client

    config = tmp_path / "settings.jsonc"
    settings = {
        "agent_distill_args": {
            "llm_provider": "codex_exec", "overwrite": False,
            "model": "model-a", "effort": "low", "batch_size": 2, "max_tokens": 200,
            "timeout": 10, "command": "codex", "sandbox": "read-only",
        },
    }
    config.write_text(json.dumps(settings))
    args = SimpleNamespace(config_path=config, llm_provider=None, output_dir=tmp_path,
                           max_context_tokens=8192, token_encoding="o200k_base", tokenizer_file=None)
    payload = {"items": [{"id": "a", "fields": [[1, "职业"]]}]}

    class Client:
        calls = 0

        def generate_batch(self, requests):
            requests = list(requests)
            self.calls += 1
            return [LLMResponse(ok=True, parsed_json=payload, finish_reason="stop") for _ in requests]

        def close(self):
            pass

    client = Client()
    monkeypatch.setattr(llm_client, "build_llm_client", lambda *a, **kw: client)
    tasks = [{"prompt": "classify a", "stage": "classify"}]

    def validator(result, task):
        return organize.validate_classification(result, [record("a")])

    first = organize.LLMTasks(args)
    assert first.run(tasks, validator) == [payload["items"]]
    first.close()
    assert client.calls == 1

    if change == "extraction":
        settings["agent_distill_args"].update(max_samples_per_window=16, max_content_chars=800)
    elif change == "execution":
        settings["agent_distill_args"].update(batch_size=7, timeout=30)
    elif change == "model":
        settings["agent_distill_args"].update(model="model-b", effort="high", max_tokens=400)
    elif change == "overwrite":
        settings["agent_distill_args"]["overwrite"] = True
    config.write_text("// edited configuration\n" + json.dumps(settings, indent=2))

    resumed = organize.LLMTasks(args)
    assert resumed.run(tasks, validator) == [payload["items"]]
    expected_calls = 2 if change == "overwrite" else 1
    assert client.calls == expected_calls
    if change == "model":
        assert resumed.options.model == "model-b"
        assert resumed.options.effort == "high"
    if change == "overwrite":
        resumed.close()
        settings["agent_distill_args"]["overwrite"] = False
        config.write_text(json.dumps(settings))
        resumed = organize.LLMTasks(args)
        assert resumed.run(tasks, validator) == [payload["items"]]
        assert client.calls == expected_calls

    monkeypatch.setattr(organize, "CLASSIFY_SCHEMA", {
        **organize.CLASSIFY_SCHEMA, "description": "Updated output protocol",
    })
    resumed.run(tasks, validator)
    assert client.calls == expected_calls + 1
    resumed.close()


@pytest.mark.parametrize("batched", [False, True])
def test_summary_overwrite_refreshes_current_tasks_and_preserves_other_entries(tmp_path, batched):
    runner = organize.LLMTasks.__new__(organize.LLMTasks)
    runner.path, runner.entries = tmp_path / "checkpoint.json", {"other-stage": {"status": "done"}}
    runner.budget = organize.InputBudget(8192, "o200k_base")
    runner.options = SimpleNamespace(
        batch_size=2, llm_provider="api", model=None, effort=None, max_tokens=200, timeout=10, overwrite=False,
    )
    task = {"stage": "summarize", "prompt": "summarize a"}
    if batched:
        task["members"] = [0]

    class Client:
        calls = 0

        def generate_batch(self, requests):
            requests = list(requests)
            self.calls += 1
            facts = [{"attr": "职业", "value": f"第{self.calls}次结果", "source_ids": ["a"]}]
            payload = {"groups": [{"id": 0, "facts": facts}]} if batched else {"facts": facts}
            return [LLMResponse(ok=True, parsed_json=payload, finish_reason="stop") for _ in requests]

    def validator(result, task):
        payload = {"facts": result["groups"][0]["facts"]} if batched else result
        return organize.validate_summary(payload, {"a"})

    runner.client = Client()
    first = runner.run([task], validator)
    runner.options.overwrite = True
    refreshed = runner.run([task], validator)
    assert runner.client.calls == 2
    assert first != refreshed
    runner.entries = organize.load(runner.path)
    assert runner.entries["other-stage"] == {"status": "done"}
    runner.options.overwrite = False
    assert runner.run([task], validator) == refreshed
    assert runner.client.calls == 2


@pytest.mark.parametrize("cross_group", [False, True])
@pytest.mark.parametrize("extra_sample", [None, "same_peer", "other_peer"])
def test_full_pipeline_keeps_inputs_and_joins_profile_goals_with_events(tmp_path, monkeypatch, cross_group, extra_sample):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    _, profile = write_input(input_dir)
    if extra_sample:
        extra = {**profile, "id": "sample2"} if extra_sample == "same_peer" else {**profile, "chat_with_id": "peer2"}
        organize.save(input_dir / "state_people" / "peer.json", [profile, extra])
    before = {p: p.read_bytes() for p in input_dir.rglob("*.json")}
    prompts = []

    class Embeddings:
        def __init__(self, args):
            pass

        def get(self, texts):
            return [[0, 1] if cross_group and i % 2 else [1, 0] for i, _ in enumerate(texts)]

        def close(self):
            pass

    class Runner:
        def __init__(self, args):
            pass

        def run(self, tasks, validate):
            results = []
            for task in tasks:
                prompts.append(task["prompt"])
                if task["stage"] == "classify":
                    payload = {"items": [{"id": rid, "fields": [[5, "行程安排" if cross_group and rid.startswith("ES:") else "出行安排"]]} for rid in task["ids"]]}
                elif "members" in task:
                    ids = sorted({sid for member in task["members"] for sid in member["source_ids"]})
                    payload = {"groups": [
                        {"id": 0, "facts": [{"attr": "出行安排", "value": "B计划与朋友周末郊游。", "source_ids": ids}]},
                        {"id": 1, "facts": []},
                    ]}
                else:
                    payload = {"facts": [{"attr": "出行安排", "value": "B计划与朋友周末郊游。", "source_ids": task["source_ids"]}]}
                results.append(validate(payload, task))
            return results

        def close(self):
            pass

    monkeypatch.setattr(organize, "Embeddings", Embeddings)
    monkeypatch.setattr(organize, "LLMTasks", Runner)
    args = SimpleNamespace(stage="all", dry_run=False, preferences_only=False, input_dir=input_dir, output_dir=output_dir,
                           max_context_tokens=272000, token_encoding="o200k_base",
                           tokenizer_file=None, max_records=20, max_summary_groups=30, max_summary_records=300,
                           neighbor_top_k=8, attribute_similarity=0.83)
    organize.run(args)
    output = organize.load(output_dir / "organized_memories.json")
    assert len(output["facts"]) == 1
    assert output["facts"][0]["dim"] == 5
    assert len(output["facts"][0]["source_ids"]) == 2 + bool(extra_sample)
    assert output["facts"][0]["support_count"] == 1 + bool(extra_sample)
    assert output["facts"][0]["source_confidence_mean"] == 3
    assert output["facts"][0]["source_importance_mean"] == 2
    assert output["unused"] == []
    if cross_group:
        assert len(organize.load(output_dir / "attribute_groups.json")) == 2
    assert {row["kind"] for row in output["sources"].values()} == {"S", "ES"}
    assert "RAW_CHAT" not in "\n".join(prompts)
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("dim", [5, 6, 8])
def test_summary_record_limit_splits_group_and_reduces_all_sources(dim):
    rows = [record(str(i)) for i in range(501)]
    calls = []

    class Runner:
        def run(self, tasks, validate):
            results = []
            for task in tasks:
                calls.append((task["stage"], len(task["source_ids"])))
                payload = {"facts": [{"attr": "属性", "value": "同一事实", "source_ids": task["source_ids"]}]}
                results.append(validate(payload, task))
            return results

    group = {"dim": dim, "peer": "P1", "attrs": ["属性"], "source_ids": [row["id"] for row in rows]}
    result = organize.summarize_groups([group], {row["id"]: row for row in rows}, Runner())
    assert calls == [("summarize", 300), ("summarize", 201), ("reduce", 501)]
    assert len(result[0]["facts"]) == 1
    assert set(result[0]["facts"][0]["source_ids"]) == set(group["source_ids"])


def test_summary_request_limit_counts_unique_sources_across_groups():
    tasks = [organize.pack_summary(5, ["属性"], [record(str(i)) for i in ids], reduced=False)[0]
             for ids in (range(200), range(100, 300), range(300, 301))]
    requests = []

    class Runner:
        def run(self, batch, validate):
            requests.extend(batch)
            return [validate({"groups": [{"id": i, "facts": []} for i in range(len(t["members"]))]}, t)
                    if "members" in t else validate({"facts": []}, t) for t in batch]

    organize.run_summary_tasks(tasks, Runner(), 30, None)
    assert len(requests) == 2
    assert requests[0]["members"] == tasks[:2]
    assert requests[1] == tasks[2]
    batches = organize.pack_summary(5, ["属性"], [record(str(i)) for i in range(4)],
                                    reduced=False, max_records=2)
    assert [task["source_ids"] for task in batches] == [["0", "1"], ["2", "3"]]


def test_oversized_attribute_group_reduces_across_batches_without_losing_sources():
    rows = [record(str(i), memory_type="goal", sample=str(i)) for i in range(5)]
    for row in rows:
        row["content"] = "B计划出行。" * 180
    stages = []

    class Runner:
        def run(self, tasks, validate):
            results = []
            for task in tasks:
                stages.append(task["stage"])
                payload = {"facts": [{"attr": "出行计划", "value": "B计划出行。", "source_ids": task["source_ids"]}]}
                results.append(validate(payload, task))
            return results

    group = {"dim": 5, "attrs": ["出行计划"], "source_ids": [row["id"] for row in rows]}
    budget = organize.InputBudget(8192, "o200k_base")
    budget.max_tokens = max(budget.count(organize.summarize_prompt(
        5, group["attrs"], [organize.prompt_record(row, include_sample_time=True)],
    )) for row in rows)
    result = organize.summarize_groups([group], {row["id"]: row for row in rows}, Runner(), budget=budget, max_groups=1)[0]
    assert stages.count("summarize") > 1
    assert "reduce" in stages
    assert len(result["facts"]) == 1
    assert set(result["facts"][0]["source_ids"]) == {row["id"] for row in rows}


def test_different_attribute_groups_share_llm_batch():
    rows = [record("a"), record("b")]
    groups = [{"dim": dim, "attrs": [attr], "source_ids": [rid]}
              for dim, attr, rid in ((1, "职业", "a"), (4, "住房", "b"))]
    calls = []

    class Runner:
        def run(self, tasks, validate):
            calls.append(len(tasks))
            return [validate({"facts": [{"attr": "属性", "value": "已有值", "source_ids": task["source_ids"]}]}, task) for task in tasks]

    results = organize.summarize_groups(groups, {row["id"]: row for row in rows}, Runner())
    assert calls == [2]
    assert [result["facts"][0]["source_ids"] for result in results] == [["a"], ["b"]]


def test_classification_fills_fifty_after_related_neighbors():
    rows = [record(str(i), sample=str(i)) for i in range(125)]
    batches = organize.classification_batches(rows, [[1, 0]] * 125, max_records=50, neighbors=8)
    assert [len(batch) for batch in batches] == [50, 50, 25]
    assert len({row["id"] for batch in batches for row in batch}) == 125


def test_summary_combines_thirty_groups_and_restores_reordered_results():
    rows = {str(i): record(str(i)) for i in range(31)}
    groups = [{"dim": 1, "attrs": [f"属性{i}"], "source_ids": [str(i)]} for i in range(31)]
    sizes = []

    class Runner:
        def run(self, tasks, validate):
            results = []
            for task in tasks:
                members = task.get("members", [task])
                sizes.append(len(members))
                payloads = [{"id": i, "facts": [{"attr": member["attrs"][0], "value": "已有事实",
                                                   "source_ids": member["source_ids"]}]}
                            for i, member in enumerate(members)]
                payload = {"groups": list(reversed(payloads))} if "members" in task else {
                    key: value for key, value in payloads[0].items() if key != "id"
                }
                results.append(validate(payload, task))
            return results

    result = organize.summarize_groups(groups, rows, Runner(), max_groups=30)
    assert sizes == [30, 1]
    assert [r["facts"][0]["source_ids"] for r in result] == [[str(i)] for i in range(31)]


@pytest.mark.parametrize("defect", ["unknown_source", "missing_group", "duplicate_group"])
def test_combined_summary_rejects_unknown_sources_and_lost_groups(defect):
    task = {"members": [{"source_ids": ["a"]}, {"source_ids": ["b"]}]}
    payload = {"groups": [{"id": i, "facts": [{"attr": "属性", "value": "事实", "source_ids": [rid]}]} for i, rid in enumerate(("a", "b"))]}
    if defect == "unknown_source":
        payload["groups"][0]["facts"][0]["source_ids"] = ["not_in_request"]
    elif defect == "missing_group":
        payload["groups"].pop()
    else:
        payload["groups"][1]["id"] = 0
    with pytest.raises(ValueError):
        organize.validate_summary_batch(payload, task)


def test_cross_group_references_warn_and_survive_final_summary(caplog):
    rows = {rid: record(rid) for rid in ("a", "b")}
    groups = [{"dim": 1, "attrs": [attr], "source_ids": [rid]}
              for attr, rid in (("职业", "a"), ("身份", "b"))]

    class Runner:
        calls = 0

        def run(self, tasks, validate):
            self.calls += len(tasks)
            return [validate({"groups": [
                {"id": 0, "facts": [{"attr": "职业", "value": "已有事实", "source_ids": ["b"]}]},
                {"id": 1, "facts": []},
            ]}, task) for task in tasks]

    runner = Runner()
    results = organize.summarize_groups(groups, rows, runner)
    assert runner.calls == 1
    assert results[0]["facts"][0]["source_ids"] == ["b"]
    assert results[1] == {"facts": []}
    assert "references sources from other groups" in caplog.text


def test_combined_summary_deduplicates_shared_memories_and_respects_token_budget():
    row = organize.prompt_record(record("shared"))
    tasks = [organize.pack_summary(1, [attr], [row], reduced=False)[0] for attr in ("职业", "身份")]
    prompt = organize.batch_summary_prompt(tasks)
    assert prompt.count('"id":"shared"') == 1
    budget = organize.InputBudget(8192, "o200k_base")
    budget.max_tokens = max(budget.count(task["prompt"]) for task in tasks)
    calls = []

    class Runner:
        def run(self, requests, validate):
            calls.extend(requests)
            assert all(budget.fits(task["prompt"]) for task in requests)
            return [validate({"facts": [{"attr": "属性", "value": "已有值", "source_ids": ["shared"]}]}, task) for task in requests]

    organize.run_summary_tasks(tasks, Runner(), 30, budget)
    assert len(calls) == 2
    assert all("members" not in task for task in calls)


def test_embedding_cache_reuses_vectors_and_checks_model_identity(tmp_path, monkeypatch):
    args = SimpleNamespace(config_path=tmp_path / "config", embedding_url="http://localhost:9999",
                           no_auto_start_embedding_service=True, embedding_batch_size=None, output_dir=tmp_path)
    health = {"model": "model1", "max_length": "4096", "device": "cpu"}
    calls = []
    monkeypatch.setattr(organize.group_state_memories, "ensure_embedding_service", lambda *a, **kw: (dict(health), None))

    def embed(texts, **kwargs):
        calls.append(list(texts))
        return [[1, 0] for _ in texts]

    monkeypatch.setattr(organize.group_state_memories, "embed_texts", embed)
    instance = organize.Embeddings(args)
    assert instance.get(["职业", "职业"]) == [[1, 0], [1, 0]]
    assert calls == [["职业"]]
    instance.close()
    assert organize.Embeddings(args).get(["职业"]) == [[1, 0]]
    assert len(calls) == 1
    health["model"] = "model2"
    with pytest.raises(ValueError, match="model changed"):
        organize.Embeddings(args).get(["职业"])


def test_embedding_batches_resume_without_rewriting_history(tmp_path, monkeypatch):
    args = SimpleNamespace(config_path=tmp_path / "config", embedding_url="http://localhost:9999",
                           no_auto_start_embedding_service=True, embedding_batch_size=2, output_dir=tmp_path)
    health = {"model": "model1", "max_length": "4096", "device": "cpu"}
    monkeypatch.setattr(organize.group_state_memories, "ensure_embedding_service", lambda *a, **kw: (health, None))
    texts = ["职业", "爱好", "目标", "身份"]
    calls = []

    def embed(batch, **kwargs):
        calls.append(list(batch))
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return [[1, 0] for _ in batch]

    monkeypatch.setattr(organize.group_state_memories, "embed_texts", embed)
    with pytest.raises(RuntimeError, match="interrupted"):
        organize.Embeddings(args).get(texts)
    shards = list((tmp_path / "embeddings").glob("*.json"))
    assert len(shards) == 1
    first_bytes = shards[0].read_bytes()
    expected = [[1, 0]] * 4
    assert organize.Embeddings(args).get(texts) == expected
    assert calls == [["职业", "爱好"], ["目标", "身份"], ["目标", "身份"]]
    assert shards[0].read_bytes() == first_bytes
    shards = list((tmp_path / "embeddings").glob("*.json"))
    assert len(shards) == 2
    assert all(len(organize.load(path)["vectors"]) == 2 for path in shards)
    assert organize.Embeddings(args).get(texts) == expected
    assert len(calls) == 3
    assert not (tmp_path / "embeddings.json").exists()


def test_default_embedding_service_starts_as_module_and_keeps_custom_script(tmp_path, monkeypatch):
    grouping = organize.group_state_memories
    args = grouping.default_args()
    args.embedding_service_log_path = tmp_path / "service.log"
    commands = []
    monkeypatch.setattr(grouping, "configure_embedding_service_env", lambda *a: {})
    monkeypatch.setattr(grouping.subprocess, "Popen", lambda command, **kw: commands.append(command))
    grouping.start_embedding_service(args, "http://localhost:8097")
    assert commands[0][1:] == ["-m", "weclone.core.inference.embedding_service"]
    custom = tmp_path / "custom.py"
    custom.write_text("pass\n")
    args.embedding_service_script = custom
    grouping.start_embedding_service(args, "http://localhost:8097")
    assert commands[1][1:] == [str(custom)]
