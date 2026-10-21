import json
from collections import Counter
from types import SimpleNamespace

import pytest

from weclone.core.inference.llm_client import LLMResponse
from weclone.data.agent import organize_memories as organize
from weclone.prompts import memory_organization as prompts

DOMAINS = [
    {"name": "出行旅行", "description": "周末郊游和旅行安排"},
    {"name": "学业", "description": "读博申请和课程"},
]


def field(dim, attr, domain=1):
    return {"dim": dim, "domain": domain, "attr": attr}


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


@pytest.mark.parametrize("preferences_only", [False, True])
def test_preference_cli_uses_separate_default_output(monkeypatch, preferences_only):
    captured = []
    monkeypatch.setattr(organize, "run", lambda args: captured.append(args))
    organize.main(["--preferences-only"] if preferences_only else [])
    assert captured[0].output_dir.name == (
        "memory_organization_preferences" if preferences_only else "memory_organization"
    )


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


def test_record_tags_strip_and_deduplicate():
    row = record("a", tags=[" 读博 ", "读博", "", 3, "导师"])
    assert organize.record_tags(row) == ["读博", "导师"]
    assert organize.tag_counts([row, record("b", tags=["读博"])]) == Counter({"读博": 2, "导师": 1})


def test_domain_tags_keep_frequent_tags_between_floor_and_limit(monkeypatch):
    monkeypatch.setattr(organize, "DOMAIN_TAG_FLOOR", 2)
    monkeypatch.setattr(organize, "DOMAIN_TAG_LIMIT", 4)
    assert organize.domain_tags(Counter({"a": 5, "b": 3, "c": 3, "d": 1, "e": 4, "f": 3})) == ["a", "e", "b", "c"]
    assert organize.domain_tags(Counter({"a": 3, "b": 1, "c": 1})) == ["a", "b"]
    assert organize.domain_tags(Counter({"a": 5, "b": 4, "c": 3, "d": 1})) == ["a", "b", "c"]


def test_domain_induction_strips_names_and_descriptions():
    payload = {"domains": [{"name": " 出行旅行 ", "description": " 周末郊游和旅行安排 "}]}
    assert organize.validate_domains(payload) == [{"name": "出行旅行", "description": "周末郊游和旅行安排"}]


@pytest.mark.parametrize("defect", ["empty", "too_many", "duplicate", "english", "no_description", "extra_key"])
def test_domain_induction_rejects_invalid_domains(defect):
    domains = [dict(domain) for domain in DOMAINS]
    if defect == "empty":
        domains = []
    elif defect == "too_many":
        domains = [{"name": f"领域{i}", "description": "描述"} for i in range(organize.MAX_DOMAINS + 1)]
    elif defect == "duplicate":
        domains[1]["name"] = domains[0]["name"]
    elif defect == "english":
        domains[0]["name"] = "travel"
    elif defect == "no_description":
        domains[0]["description"] = " "
    else:
        domains[0]["topics"] = []
    with pytest.raises(ValueError):
        organize.validate_domains({"domains": domains})


@pytest.mark.parametrize("defect", [None, "missing", "unknown", "duplicate", "domain_range", "bool_domain"])
def test_tag_mapping_covers_each_tag_once_with_known_domains(defect):
    items = [{"id": 1, "domain": 2}, {"id": 2, "domain": 1}]
    if defect == "missing":
        items.pop()
    elif defect == "unknown":
        items[1]["id"] = 3
    elif defect == "duplicate":
        items[1]["id"] = 1
    elif defect == "domain_range":
        items[1]["domain"] = 3
    elif defect == "bool_domain":
        items[1]["domain"] = True
    if defect is None:
        assert organize.validate_tag_domains({"items": items}, {1, 2}, 2) == {1: 2, 2: 1}
    else:
        with pytest.raises(ValueError):
            organize.validate_tag_domains({"items": items}, {1, 2}, 2)


def test_provisional_domain_takes_majority_then_rarest_tag():
    rows = [record("a", tags=["读博", "导师", "租房"]), record("b", tags=["读博", "租房"]),
            record("c", tags=["未映射"]), record("d")]
    mapping = {"读博": 2, "导师": 2, "租房": 1}
    counts = Counter({"读博": 10, "导师": 3, "租房": 5})
    assert organize.provisional_domains(rows, mapping, counts) == [2, 1, None, None]


def test_classification_batches_follow_domain_then_rarest_shared_tag_then_time():
    times = {"a": "2025-03-01", "b": "2025-01-01", "c": "2025-02-01", "d": "2025-01-01", "e": "2025-01-01",
             "g": "2025-04-01"}
    tags = {"a": ["读博"], "b": ["郊游"], "c": ["读博", "导师"], "d": ["读博"], "e": [], "g": ["导师", "开题"]}
    rows = [{**record(rid, tags=tags[rid]), "sample_time": times[rid]} for rid in "abcdeg"]
    rows += [record("ev", "ES", "ES", tags=["读博"]), record("p", memory_type="preference", tags=["郊游"])]
    domains = [2, 1, 2, 2, None, 2, 2, 1]
    batches = organize.classification_batches(rows, domains, organize.tag_counts(rows), max_records=3,
                                              prompt=lambda batch, dims: "")
    assert [[row["id"] for row in batch] for batch in batches] == [["b", "c", "g"], ["d", "a", "e"], ["ev"], ["p"]]


def test_token_budget_splits_classification_even_when_characters_fit():
    rows = [record("a"), record("b", sample="two")]

    def prompt(batch, dims):
        return organize.classify_prompt([organize.prompt_record(row) for row in batch], dims, DOMAINS)

    counter = organize.InputBudget(8192, "o200k_base")
    limit = max(counter.count(prompt([row], organize.allowed_dimensions(row))) for row in rows)
    budget = organize.InputBudget(limit, "o200k_base")
    counts = organize.tag_counts(rows)
    batches = organize.classification_batches(rows, [None, None], counts, max_records=20, prompt=prompt, budget=budget)
    assert [len(batch) for batch in batches] == [1, 1]
    budget.max_tokens = limit - 1
    with pytest.raises(ValueError, match="budget"):
        organize.classification_batches(rows, [None, None], counts, max_records=20, prompt=prompt, budget=budget)


def test_classification_fills_fifty_per_request():
    rows = [record(str(i), sample=str(i)) for i in range(125)]
    batches = organize.classification_batches(rows, [None] * 125, Counter(), max_records=50,
                                              prompt=lambda batch, dims: "")
    assert [len(batch) for batch in batches] == [50, 50, 25]
    assert len({row["id"] for batch in batches for row in batch}) == 125


def test_classify_prompts_share_rules_and_domains_before_memories():
    first = organize.classify_prompt([organize.prompt_record(record("a"))], (1, 3, 4, 6), DOMAINS)
    second = organize.classify_prompt([organize.prompt_record(record("b"))], (1, 3, 4, 6), DOMAINS)
    prefix = first.split("\n记忆：\n")[0]
    assert second.startswith(prefix + "\n记忆：\n")
    assert prefix.startswith(prompts.CLASSIFY_RULES) and "1. 出行旅行：周末郊游和旅行安排" in prefix
    preference = organize.classify_prompt(
        [organize.prompt_record(record("p", memory_type="preference"))], (8,), DOMAINS
    )
    assert preference.startswith(prompts.PREFERENCE_RULES + prompts.domain_block(DOMAINS))
    assert "本次维度" not in preference


@pytest.mark.parametrize("kind,type_", [("S", "stable_fact"), ("S", "goal"), ("ES", "ES")])
def test_preference_dimension_rejects_other_memory_types(kind, type_):
    with pytest.raises(ValueError, match="Invalid dimension/source"):
        organize.validate_classification({"items": [{"id": "a", "fields": [field(8, "活动偏好")]}]},
                                         [record("a", kind, type_)], 2)


@pytest.mark.parametrize("defect", ["missing", "unknown", "duplicate", "wrong_source", "removed_dim", "english_attr",
                                    "bool_dim", "unknown_domain", "zero_domain", "bool_domain", "pair_field",
                                    "duplicate_attr"])
def test_classification_rejects_lost_ids_wrong_dimensions_and_domains(defect):
    rows = [record("s"), record("e", "ES", "ES")]
    payload = {"items": [{"id": "s", "fields": [field(1, "职业")]}, {"id": "e", "fields": [field(5, "约定")]}]}
    first = payload["items"][0]["fields"]
    if defect == "missing":
        payload["items"].pop()
    elif defect == "unknown":
        payload["items"][1]["id"] = "unknown"
    elif defect == "duplicate":
        payload["items"][1]["id"] = "s"
    elif defect == "wrong_source":
        payload["items"][1]["fields"] = [field(1, "职业")]
    elif defect == "removed_dim":
        payload["items"][1]["fields"] = [field(2, "人生经历")]
    elif defect == "english_attr":
        first[0]["attr"] = "occupation"
    elif defect == "bool_dim":
        first[0]["dim"] = True
    elif defect == "unknown_domain":
        first[0]["domain"] = 3
    elif defect == "zero_domain":
        first[0]["domain"] = 0
    elif defect == "bool_domain":
        first[0]["domain"] = True
    elif defect == "pair_field":
        payload["items"][0]["fields"] = [[1, "职业"]]
    else:
        first.append(field(1, "职业", 2))
    with pytest.raises(ValueError):
        organize.validate_classification(payload, rows, 2)


def test_open_attributes_multi_assignment_and_empty_result_are_supported():
    rows = [record("a"), record("b")]
    payload = {"items": [
        {"id": "a", "fields": [field(1, " 职业 "), field(4, "照护可用时间", 2)]},
        {"id": "b", "fields": []},
    ]}
    assert organize.validate_classification(payload, rows, 2) == [
        {"id": "a", "fields": [field(1, "职业"), field(4, "照护可用时间", 2)]},
        {"id": "b", "fields": []},
    ]


def test_topic_inputs_number_profile_attributes_per_domain_with_counts_and_example():
    rows = {rid: record(rid, kind) for rid, kind in [("a", "S"), ("b", "S"), ("c", "S"), ("e", "ES")]}
    rows["a"].update(content="A是B的导师，\n" + "很长的内容" * 10, importance=4)
    classifications = [
        {"id": "a", "fields": [field(1, "导师", 2), field(6, "师生关系", 2)]},
        {"id": "b", "fields": [field(1, "导师", 2)]},
        {"id": "c", "fields": [field(3, "读博目标", 2), field(1, "出行习惯")]},
        {"id": "e", "fields": [field(7, "天气")]},
    ]
    assert organize.topic_inputs(classifications, rows) == {
        1: [{"id": 1, "dim": 1, "attr": "出行习惯", "count": 1, "example": "B的记忆 c"}],
        2: [
            {"id": 1, "dim": 1, "attr": "导师", "count": 2, "example": ("peer1是B的导师， " + "很长的内容" * 10)[:30]},
            {"id": 2, "dim": 3, "attr": "读博目标", "count": 1, "example": "B的记忆 c"},
        ],
    }
    prompt = organize.topic_prompt(DOMAINS, 2, organize.topic_inputs(classifications, rows)[2], 20, 100)
    assert f"1 导师（{prompts.dimension_name(1)}，2 条）例：peer1是B的导师" in prompt
    assert "## 当前领域\n学业：读博申请和课程" in prompt and "一般在 20 到 100 条之间" in prompt


def test_topics_strip_names_and_keep_items():
    payload = {"topics": [{"name": " 读博申请 ", "items": [2, 1]}, {"name": "日常出行", "items": [3]}]}
    assert organize.validate_topics(payload, 3) == [
        {"name": "读博申请", "items": [2, 1]}, {"name": "日常出行", "items": [3]},
    ]


@pytest.mark.parametrize("defect", ["missing", "duplicate_id", "out_of_range", "duplicate_name", "english_name",
                                    "empty_items", "bool_id"])
def test_topics_cover_every_attribute_once(defect):
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
    elif defect == "empty_items":
        topics.append({"name": "空主题", "items": []})
    else:
        topics[1]["items"] = [True, 3]
    with pytest.raises(ValueError):
        organize.validate_topics({"topics": topics}, 3)


def test_summary_units_follow_topics_peers_and_event_domains():
    rows = {rid: {**record(rid, kind), "peer": peer} for rid, kind, peer in
            [("a", "S", "P1"), ("b", "S", "P2"), ("e", "ES", "P1"), ("f", "ES", "P2"), ("g", "ES", "P1")]}
    classifications = [
        {"id": "a", "fields": [field(1, "导师", 2), field(3, "读博目标", 2), field(6, "师生关系", 2)]},
        {"id": "b", "fields": [field(1, "导师", 2), field(6, "同学关系")]},
        {"id": "e", "fields": [field(5, "面试安排", 2), field(7, "天气")]},
        {"id": "f", "fields": [field(7, "交通")]},
        {"id": "g", "fields": [field(7, "宿舍条件", 2)]},
    ]
    topics = [{"domain": 2, "name": "读博申请", "attrs": [[1, "导师"], [3, "读博目标"], [5, "面试安排"]]}]
    assert organize.summary_groups(classifications, rows, topics) == [
        {"dim": 1, "domain": 2, "topic": "读博申请", "attrs": ["导师"], "source_ids": ["a", "b"]},
        {"dim": 3, "domain": 2, "topic": "读博申请", "attrs": ["读博目标"], "source_ids": ["a"]},
        {"dim": 5, "domain": 2, "topic": "读博申请", "attrs": ["面试安排"], "source_ids": ["e"]},
        {"dim": 6, "peer": "P1", "attrs": ["师生关系"], "source_ids": ["a"]},
        {"dim": 6, "peer": "P2", "attrs": ["同学关系"], "source_ids": ["b"]},
        {"dim": 7, "domain": 1, "attrs": ["交通", "天气"], "source_ids": ["e", "f"]},
        {"dim": 7, "domain": 2, "attrs": ["宿舍条件"], "source_ids": ["g"]},
    ]


@pytest.mark.parametrize("defect", ["missing_attr", "extra_attr", "duplicate_attr", "wrong_domain"])
def test_summary_units_reject_topics_that_do_not_match_classifications(defect):
    rows = {"a": record("a")}
    classifications = [{"id": "a", "fields": [field(1, "导师", 2), field(3, "读博目标", 2)]}]
    topics = [{"domain": 2, "name": "读博申请", "attrs": [[1, "导师"], [3, "读博目标"]]}]
    if defect == "missing_attr":
        topics[0]["attrs"].pop()
    elif defect == "extra_attr":
        topics[0]["attrs"].append([1, "专业"])
    elif defect == "duplicate_attr":
        topics.append({"domain": 2, "name": "导师沟通", "attrs": [[1, "导师"]]})
    else:
        topics[0]["domain"] = 1
    with pytest.raises(ValueError, match="Topics do not match"):
        organize.summary_groups(classifications, rows, topics)


def test_summary_requests_combine_groups_only_within_one_domain():
    rows = {rid: record(rid) for rid in ("a", "b", "c")}
    groups = [{"dim": 1, "domain": domain, "topic": topic, "attrs": ["职业"], "source_ids": [rid]}
              for domain, topic, rid in [(1, "工作", "a"), (1, "副业", "b"), (2, "读博", "c")]]
    requests = []

    class Runner:
        def run(self, tasks, validate):
            requests.extend(tasks)
            return [validate({"groups": [{"id": i, "facts": []} for i in range(len(task["members"]))]}, task)
                    if "members" in task else validate({"facts": []}, task) for task in tasks]

    organize.summarize_groups(groups, rows, Runner())
    assert len(requests) == 2
    assert [member["topic"] for member in requests[0]["members"]] == ["工作", "副业"]
    assert '"topic":"工作"' in requests[0]["prompt"]
    assert requests[1]["topic"] == "读博" and '主题："读博"' in requests[1]["prompt"]


def test_task_schema_follows_stage_and_overrides():
    assert organize.task_schema({"stage": "domains"}) is organize.DOMAIN_SCHEMA
    assert organize.task_schema({"stage": "tag_domains"}) is organize.TAG_DOMAIN_SCHEMA
    assert organize.task_schema({"stage": "classify"}) is organize.CLASSIFY_SCHEMA
    assert organize.task_schema({"stage": "topics"}) is organize.TOPIC_SCHEMA
    assert organize.task_schema({"stage": "reduce"}) is organize.SUMMARY_SCHEMA
    assert organize.task_schema({"stage": "summarize", "members": []}) is organize.BATCH_SUMMARY_SCHEMA
    assert organize.task_schema({"stage": "split_topic", "schema": {"type": "object"}}) == {"type": "object"}


def fake_runner(fields_for, seen, *, value="B计划与朋友周末郊游。"):
    """Stand-in for LLMTasks that answers every organizer stage."""

    class Runner:
        def __init__(self, args):
            pass

        def run(self, tasks, validate):
            results = []
            for task in tasks:
                seen.append(task)
                if task["stage"] == "domains":
                    payload = {"domains": DOMAINS}
                elif task["stage"] == "tag_domains":
                    payload = {"items": [{"id": i, "domain": 1} for i in task["ids"]]}
                elif task["stage"] == "classify":
                    payload = {"items": [{"id": rid, "fields": fields_for(rid)} for rid in task["ids"]]}
                elif task["stage"] == "topics":
                    count = len(task["prompt"].split("记忆开头\n", 1)[1].splitlines())
                    payload = {"topics": [{"name": "周末郊游", "items": list(range(1, count + 1))}]}
                elif "members" in task:
                    payload = {"groups": [
                        {"id": i, "facts": [{"attr": member["attrs"][0], "value": value,
                                             "source_ids": member["source_ids"]}]}
                        for i, member in enumerate(task["members"])
                    ]}
                else:
                    payload = {"facts": [{"attr": task["attrs"][0], "value": value,
                                          "source_ids": task["source_ids"]}]}
                results.append(validate(payload, task))
            return results

        def close(self):
            pass

    return Runner


def run_args(input_dir, output_dir, stage="all", **overrides):
    return SimpleNamespace(stage=stage, dry_run=False, preferences_only=False, input_dir=input_dir,
                           output_dir=output_dir, max_context_tokens=272000, token_encoding="o200k_base",
                           tokenizer_file=None, max_records=20, max_summary_groups=30, max_summary_records=300,
                           topic_min_memories=20, topic_max_memories=100, **overrides)


@pytest.mark.parametrize("stagewise", [False, True])
@pytest.mark.parametrize("cross_attr", [False, True])
@pytest.mark.parametrize("extra_sample", [None, "same_peer", "other_peer"])
def test_full_pipeline_keeps_inputs_and_joins_profile_goals_with_events(tmp_path, monkeypatch, stagewise, cross_attr,
                                                                       extra_sample):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    _, profile = write_input(input_dir)
    if extra_sample:
        extra = {**profile, "id": "sample2"} if extra_sample == "same_peer" else {**profile, "chat_with_id": "peer2"}
        organize.save(input_dir / "state_people" / "peer.json", [profile, extra])
    before = {p: p.read_bytes() for p in input_dir.rglob("*.json")}
    seen = []

    def fields_for(rid):
        return [field(5, "行程安排" if cross_attr and rid.startswith("ES:") else "出行安排")]

    monkeypatch.setattr(organize, "LLMTasks", fake_runner(fields_for, seen))
    for stage in organize.STAGES if stagewise else ("all",):
        organize.run(run_args(input_dir, output_dir, stage))
    output = organize.load(output_dir / "organized_memories.json")
    attrs = ["出行安排", "行程安排"] if cross_attr else ["出行安排"]
    assert organize.load(output_dir / "domains.json") == {"domains": DOMAINS, "tags": {"出行旅行": 1}}
    assert organize.load(output_dir / "topics.json") == [
        {"domain": 1, "name": "周末郊游", "attrs": [[5, attr] for attr in attrs]}
    ]
    assert output["topics"] == [{"domain": "出行旅行", "name": "周末郊游", "attrs": [[5, attr] for attr in attrs]}]
    assert [task["attrs"] for task in seen if task["stage"] == "summarize"] == [attrs]
    assert len(output["facts"]) == 1
    fact = output["facts"][0]
    assert (fact["dim"], fact["domain"], fact["topic"]) == (5, "出行旅行", "周末郊游")
    assert len(fact["source_ids"]) == 2 + bool(extra_sample)
    assert fact["support_count"] == 1 + bool(extra_sample)
    assert fact["source_confidence_mean"] == 3
    assert fact["source_importance_mean"] == 2
    assert output["unused"] == [] and output["unassigned_ids"] == []
    assert output["domains"] == DOMAINS
    assert {row["kind"] for row in output["sources"].values()} == {"S", "ES"}
    classify = [task["prompt"] for task in seen if task["stage"] == "classify"]
    assert classify and all(prompts.domain_block(DOMAINS) in prompt for prompt in classify)
    assert "出行旅行 " in next(task["prompt"] for task in seen if task["stage"] == "domains")
    assert "RAW_CHAT" not in "\n".join(task["prompt"] for task in seen)
    assert all(p.read_bytes() == content for p, content in before.items())


def test_pipeline_keeps_relations_per_peer_and_events_per_domain(tmp_path, monkeypatch):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    sample, profile = write_input(input_dir)
    organize.save(input_dir / "state_people" / "peer.json", [profile, {**profile, "chat_with_id": "peer2"}])
    events = [{"surface_event": text, "event_types": ["state_event"], "tags": ["出行旅行"], "importance": 2,
               "confidence": 3} for text in ("周末下雨。", "地铁停运。")]
    organize.save(input_dir / "event_people" / "peer.json",
                  [{**sample, "event_memories": {"surface_events": events}}])
    event_attrs = iter(["天气", "交通"])

    def fields_for(rid):
        return [field(6, "关系状态")] if rid.startswith("S:") else [field(7, next(event_attrs), 2)]

    monkeypatch.setattr(organize, "LLMTasks", fake_runner(fields_for, []))
    organize.run(run_args(input_dir, output_dir))
    output = organize.load(output_dir / "organized_memories.json")
    assert organize.load(output_dir / "topics.json") == []
    assert [(fact["dim"], fact["domain"], fact["topic"], fact.get("peer")) for fact in output["facts"]] == [
        (6, None, None, "P1"), (6, None, None, "P2"), (7, "学业", None, None),
    ]
    assert len(output["facts"][2]["source_ids"]) == 2


def test_prepare_rejects_a_changed_input_snapshot(tmp_path):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    _, profile = write_input(input_dir)
    organize.run(run_args(input_dir, output_dir, "prepare"))
    organize.run(run_args(input_dir, output_dir, "prepare"))
    profile["state_memories"][0]["content"] = "B计划周末去爬山。"
    organize.save(input_dir / "state_people" / "peer.json", [profile])
    with pytest.raises(ValueError, match="snapshot changed"):
        organize.run(run_args(input_dir, output_dir, "prepare"))


def test_domains_require_tagged_memories(tmp_path, monkeypatch):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    _, profile = write_input(input_dir)
    profile["state_memories"][0]["tags"] = []
    organize.save(input_dir / "state_people" / "peer.json", [profile])
    (input_dir / "event_people" / "peer.json").unlink()
    monkeypatch.setattr(organize, "LLMTasks", fake_runner(lambda rid: [], []))
    with pytest.raises(ValueError, match="no tags"):
        organize.run(run_args(input_dir, output_dir))


def test_preference_pipeline_only_sends_preferences_and_retains_sources(tmp_path, monkeypatch):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    _, profile = write_input(input_dir)
    profile["state_memories"].extend([
        {"type": "preference", "content": text, "tags": ["户外运动"], "importance": 2, "confidence": 3}
        for text in ["B喜欢与聊天对象一起徒步。", "B偏好与聊天对象结伴徒步。"]
    ])
    organize.save(input_dir / "state_people" / "peer.json", [profile])
    seen = []
    attrs = iter(["活动偏好", "休闲活动偏好"])
    monkeypatch.setattr(organize, "LLMTasks", fake_runner(lambda rid: [field(8, next(attrs))], seen,
                                                          value="B喜欢与P1结伴徒步。"))
    organize.main(["--preferences-only", "--input-dir", str(input_dir), "--output-dir", str(output_dir),
                   "--max-context-tokens", "8192"])
    output = organize.load(output_dir / "organized_memories.json")
    assert len(output["facts"]) == 1
    fact = output["facts"][0]
    assert (fact["dim"], fact["topic"]) == (8, "周末郊游")
    assert set(fact["source_ids"]) == set(output["sources"])
    assert len(output["sources"]) == 2
    assert {row["type"] for row in output["sources"].values()} == {"preference"}
    assert fact["support_count"] == 1
    tasks = organize.load(output_dir / "classification_tasks.json")
    assert tasks and all(task["prompt"].startswith(prompts.PREFERENCE_RULES) for task in tasks)
    assert organize.load(output_dir / "topics.json")[0]["attrs"] == [[8, "休闲活动偏好"], [8, "活动偏好"]]
    assert all("B计划周末去郊游" not in task["prompt"] and "RAW_CHAT" not in task["prompt"] for task in seen)
    domain_prompt = next(task["prompt"] for task in seen if task["stage"] == "domains")
    assert domain_prompt.endswith("\n户外运动 2") and "出行旅行" not in domain_prompt
    assert all('"peer":"P1"' in task["prompt"] for task in seen if task["stage"] == "classify")


def test_classify_stage_sends_preferences_with_preference_rules(tmp_path, monkeypatch):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    _, profile = write_input(input_dir)
    profile["state_memories"].append(
        {"type": "preference", "content": "B喜欢徒步。", "tags": ["户外运动"], "importance": 2, "confidence": 3}
    )
    organize.save(input_dir / "state_people" / "peer.json", [profile])

    def fields_for(rid):
        return []

    monkeypatch.setattr(organize, "LLMTasks", fake_runner(fields_for, []))
    for stage in ("prepare", "domains", "classify"):
        organize.run(run_args(input_dir, output_dir, stage))
    records = organize.load(output_dir / "records.json")
    tasks = organize.load(output_dir / "classification_tasks.json")
    preference_ids = {row["id"] for row in records if row["type"] == "preference"}
    assert len(records) == 3 and len(preference_ids) == 1
    for task in tasks:
        is_preference = set(task["ids"]) <= preference_ids
        assert is_preference or not set(task["ids"]) & preference_ids
        assert task["prompt"].startswith(prompts.PREFERENCE_RULES if is_preference else prompts.CLASSIFY_RULES)
    assert {rid for task in tasks for rid in task["ids"]} == {row["id"] for row in records}
    assert organize.load(output_dir / "classifications.json") == [
        {"id": rid, "fields": []} for task in tasks for rid in task["ids"]
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
    payload = {"items": [{"id": "a", "fields": [field(1, "职业")]}]}

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
        return organize.validate_classification(result, [record("a")], 2)

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
    payload = {"items": [{"id": "a", "fields": [field(1, "职业")]}]}

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
        return organize.validate_classification(result, [record("a")], 2)

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
