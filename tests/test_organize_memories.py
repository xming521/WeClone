import json
from collections import Counter
from types import SimpleNamespace

import pytest

from weclone.core.inference.llm_client import LLMResponse
from weclone.data.agent import organize_memories as organize
from weclone.data.agent.llm_cache import LLMCache
from weclone.prompts import memory_organization as prompts

DOMAINS = [
    {"name": "出行旅行", "description": "周末郊游和旅行安排"},
    {"name": "学业", "description": "读博申请和课程"},
]


def field(dim, domain=1):
    return {"dim": dim, "domain": domain}


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
def test_summary_injects_only_current_dimension_time_rule(dim):
    rows = [{"id": "a", "content": "事实", "sample_time": "2025-01-01"}]
    tasks = [{"dim": dim, "domain": domain, "source_ids": ["a"], "rows": rows, "stage": "summarize"}
             for domain in ("学业", "工作")]
    for prompt in (prompts.summarize_prompt(dim, rows),
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
    ("B研究AI和API。", "B研究AI和API。"),
    ("B与A约定见面。", "B与peer1约定见面。"),
    ("Ａ与B讨论ＡＩ。", "peer1与B讨论ＡＩ。"),
    ("A与B讨论A股和Ａ股。", "peer1与B讨论A股和Ａ股。"),
])
def test_prompt_record_limits_metadata_without_changing_source(content, expected_content):
    source = {**record("x"), "content": content, "status": "ongoing", "people": ["B"], "event_types": ["daily_event"]}
    original = json.dumps(source)
    expected = {"id": "x", "content": expected_content, "status": "ongoing"}
    assert organize.prompt_record(source) == expected
    assert organize.prompt_record(source, include_sample_time=True) == {**expected, "sample_time": "2025-01-01"}
    assert json.dumps(source) == original
    assert organize.prompt_record(record("basic")) == {"id": "basic", "content": "B的记忆 basic"}


def write_input(root):
    sample = {
        "id": "sample1", "local_id": "0", "chat_with_id": "P1", "time": "2025-01-01",
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


@pytest.mark.parametrize("kind", ["S", "ES"])
def test_reader_filters_scores_without_modifying_sources(tmp_path, kind):
    sample, _ = write_input(tmp_path)
    content_key = {"S": "content", "ES": "surface_event"}[kind]
    memories = [{content_key: f"记忆{i}", "type": "stable_fact", "importance": importance, "confidence": confidence}
                for i, (importance, confidence) in enumerate([(1, 4), (4, 2), (2, 3), (4, 4)])]
    folder = "state_people" if kind == "S" else "event_people"
    payload = {"state_memories": memories} if kind == "S" else {
        "event_memories": {"surface_events": memories}}
    path = tmp_path / folder / "peer.json"
    path.write_text(json.dumps([{**sample, **payload}], ensure_ascii=False))
    before = path.read_bytes()
    rows = [row for row in organize.read_memories(tmp_path)[0] if row["kind"] == kind]
    assert [row["content"] for row in rows] == ["记忆2", "记忆3"]
    assert [(row["importance"], row["confidence"]) for row in rows] == [(2, 3), (4, 4)]
    assert [row["origins"][0]["memory_index"] for row in rows] == [2, 3]
    assert path.read_bytes() == before


def test_reader_excludes_raw_chat_deduplicates_auxiliary_and_retains_origins(tmp_path):
    _, profile = write_input(tmp_path)
    (tmp_path / "state_people" / "duplicate.json").write_text(json.dumps([profile]))
    rows, peers = organize.read_memories(tmp_path)
    assert len(rows) == 2
    assert "RAW_CHAT" not in json.dumps(rows)
    assert len(next(row for row in rows if row["kind"] == "S")["origins"]) == 2
    assert peers == {"P1": {"chat_with": None}}
    other = {**profile, "chat_with_id": "P3", "chat_with": "同学"}
    (tmp_path / "state_people" / "other.json").write_text(json.dumps([other], ensure_ascii=False))
    rows, peers = organize.read_memories(tmp_path)
    assert len(rows) == 3
    assert peers == {"P1": {"chat_with": None}, "P3": {"chat_with": "同学"}}
    assert {peers[row["peer"]]["chat_with"] for row in rows if row["origins"][0]["file"].endswith("other.json")} == {"同学"}


@pytest.mark.parametrize("chat_with_id", ["chat_with_0003", "P0", None])
def test_reader_rejects_samples_without_a_contact_id(tmp_path, chat_with_id):
    _, profile = write_input(tmp_path)
    (tmp_path / "state_people" / "peer.json").write_text(json.dumps([{**profile, "chat_with_id": chat_with_id}]))
    with pytest.raises(ValueError, match="contact id"):
        organize.read_memories(tmp_path)


@pytest.mark.parametrize("kind,type_,dims", [
    ("S", "stable_fact", (1, 2, 3, 4, 6, 8)),
    ("S", "goal", (1, 3, 4, 5, 6)),
    ("S", "preference", (8,)),
    ("ES", "ES", (5, 7)),
])
def test_dimension_source_contract(kind, type_, dims):
    assert organize.allowed_dimensions(record("x", kind, type_)) == dims


@pytest.mark.parametrize("configured, expected", [(None, 95000), (160000, 152000)])
def test_context_window_uses_codex_effective_window_not_model_maximum(tmp_path, monkeypatch, configured, expected):
    config = tmp_path / "settings.jsonc"
    config.write_text(json.dumps({
        "codex_exec_args": {"model": "test-model", "command": "codex"},
        "agent_distill_args": {"llm_provider": "codex_exec"},
    }))
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
    config.write_text(json.dumps({
        "codex_exec_args": {"model": "unknown-model", "command": "codex"},
        "agent_distill_args": {"llm_provider": "codex_exec"},
    }))
    async def effective_config(command):
        return {}

    monkeypatch.setattr(organize, "codex_config", effective_config)
    monkeypatch.setattr(organize.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout='{"models":[]}'))
    args = SimpleNamespace(max_context_tokens=None, llm_provider=None, config_path=config)
    with pytest.raises(ValueError, match="--max-context-tokens"):
        organize.resolve_context_window(args)


@pytest.mark.parametrize("reduced", [False, True])
def test_token_budget_splits_summary_and_cross_batch_merge(reduced):
    rows = ([{"attr": "职业", "value": "B从事算法工作。", "source_ids": [rid]} for rid in ("a", "b")]
            if reduced else [organize.prompt_record(record(rid)) for rid in ("a", "b")])
    counter = organize.InputBudget(8192, "o200k_base")
    limit = max(counter.count(organize.summarize_prompt(1, [row], reduced=reduced)) for row in rows)
    budget = organize.InputBudget(limit, "o200k_base")
    tasks = organize.pack_summary(1, rows, reduced=reduced, budget=budget)
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


@pytest.mark.parametrize("defect", ["missing", "unknown", "duplicate", "wrong_source", "extra_key",
                                    "bool_dim", "unknown_domain", "zero_domain", "bool_domain", "pair_field",
                                    "duplicate_domain", "no_sensitivity", "unknown_sensitivity"])
def test_classification_rejects_lost_ids_wrong_dimensions_and_domains(defect):
    rows = [record("s"), record("e", "ES", "ES")]
    payload = {"items": [{"id": "s", "fields": [field(1)], "sensitivity": "normal"}, {"id": "e", "fields": [field(5)], "sensitivity": "normal"}]}
    first = payload["items"][0]["fields"]
    if defect == "missing":
        payload["items"].pop()
    elif defect == "unknown":
        payload["items"][1]["id"] = "unknown"
    elif defect == "duplicate":
        payload["items"][1]["id"] = "s"
    elif defect == "wrong_source":
        payload["items"][1]["fields"] = [field(1)]
    elif defect == "extra_key":
        first[0]["attr"] = "职业"
    elif defect == "bool_dim":
        first[0]["dim"] = True
    elif defect == "unknown_domain":
        first[0]["domain"] = 3
    elif defect == "zero_domain":
        first[0]["domain"] = 0
    elif defect == "bool_domain":
        first[0]["domain"] = True
    elif defect == "pair_field":
        payload["items"][0]["fields"] = [[1, 1]]
    elif defect == "no_sensitivity":
        del payload["items"][1]["sensitivity"]
    elif defect == "unknown_sensitivity":
        payload["items"][1]["sensitivity"] = "secret"
    else:
        first.append(field(1))
    with pytest.raises(ValueError, match="Invalid dimension/source" if defect == "wrong_source" else None):
        organize.validate_classification(payload, rows, 2)


def test_multi_assignment_and_empty_result_are_supported():
    rows = [record("a"), record("b")]
    payload = {"items": [
        {"id": "a", "fields": [field(1), field(1, 2), field(4, 2)], "sensitivity": "normal"},
        {"id": "b", "fields": [], "sensitivity": "normal"},
    ]}
    assert organize.validate_classification(payload, rows, 2) == [
        {"id": "a", "fields": [field(1), field(1, 2), field(4, 2)], "sensitivity": "normal"},
        {"id": "b", "fields": [], "sensitivity": "normal"},
    ]


def test_summary_units_follow_domains_and_peers():
    rows = {rid: {**record(rid, kind), "peer": peer} for rid, kind, peer in
            [("a", "S", "P1"), ("b", "S", "P2"), ("e", "ES", "P1"), ("f", "ES", "P2"), ("g", "ES", "P1")]}
    classifications = [
        {"id": "a", "fields": [field(1, 2), field(3, 2), field(6, 2)]},
        {"id": "b", "fields": [field(1, 2), field(1), field(6)]},
        {"id": "e", "fields": [field(5, 2), field(7)]},
        {"id": "f", "fields": [field(7)]},
        {"id": "g", "fields": [field(7, 2)]},
    ]
    assert organize.summary_groups(classifications, rows, DOMAINS) == [
        {"dim": 1, "domain": "出行旅行", "source_ids": ["b"]},
        {"dim": 1, "domain": "学业", "source_ids": ["a", "b"]},
        {"dim": 3, "domain": "学业", "source_ids": ["a"]},
        {"dim": 5, "domain": "学业", "source_ids": ["e"]},
        {"dim": 6, "peer": "P1", "source_ids": ["a"]},
        {"dim": 6, "peer": "P2", "source_ids": ["b"]},
        {"dim": 7, "domain": "出行旅行", "source_ids": ["e", "f"]},
        {"dim": 7, "domain": "学业", "source_ids": ["g"]},
    ]


@pytest.mark.parametrize("dim", [1, 3])
def test_summary_requests_combine_identity_across_domains_and_keep_other_domains_apart(dim):
    rows = {rid: record(rid) for rid in ("a", "b", "c")}
    groups = [{"dim": dim, "domain": domain, "source_ids": [rid]}
              for domain, rid in [("工作", "a"), ("副业", "b"), ("学业", "c")]]
    requests = []

    class Runner:
        def run(self, tasks, validate):
            requests.extend(tasks)
            return [validate({"groups": [{"id": i, "facts": []} for i in range(len(task["members"]))]}, task)
                    if "members" in task else validate({"facts": []}, task) for task in tasks]

    organize.summarize_groups(groups, rows, Runner())
    if dim == 1:
        assert len(requests) == 1
        assert [member["domain"] for member in requests[0]["members"]] == ["工作", "副业", "学业"]
        assert '"domain":"工作"' in requests[0]["prompt"]
        return
    assert [request["domain"] for request in requests] == ["工作", "副业", "学业"]
    assert '领域："学业"' in requests[2]["prompt"]


@pytest.mark.parametrize("dim", [3, 7])
def test_summary_requests_join_groups_of_different_domains_that_share_a_memory(dim):
    rows = {rid: record(rid) for rid in ("a", "b", "c", "d")}
    units = [("工作", ["a"]), ("副业", ["b", "c"]), ("学业", ["c", "d"])] if dim == 3 else [
        ("工作", ["a", "b", "c"]), ("学业", ["c", "d"])]
    groups = [{"dim": dim, "domain": domain, "source_ids": ids} for domain, ids in units]
    requests = []

    class Runner:
        def run(self, tasks, validate):
            requests.extend(tasks)
            return [validate({"groups": [{"id": i, "facts": []} for i in range(len(task["members"]))]}, task)
                    if "members" in task else validate({"facts": []}, task) for task in tasks]

    organize.summarize_groups(groups, rows, Runner())
    if dim == 7:
        assert [request["domain"] for request in requests] == ["工作", "学业"]
        return
    assert len(requests) == 2
    assert requests[0]["domain"] == "工作"
    assert [member["domain"] for member in requests[1]["members"]] == ["副业", "学业"]


def test_task_schema_follows_stage_and_overrides():
    assert organize.task_schema({"stage": "domains"}) is organize.DOMAIN_SCHEMA
    assert organize.task_schema({"stage": "tag_domains"}) is organize.TAG_DOMAIN_SCHEMA
    assert organize.task_schema({"stage": "peers"}) is organize.PEER_SCHEMA
    assert organize.task_schema({"stage": "classify"}) is organize.CLASSIFY_SCHEMA
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
                elif task["stage"] == "peers":
                    payload = {"relation": " B的 朋友 "}
                elif task["stage"] == "classify":
                    payload = {"items": [{"id": rid, "fields": fields_for(rid), "sensitivity": "private"} for rid in task["ids"]]}
                elif "members" in task:
                    payload = {"groups": [
                        {"id": i, "facts": [{"attr": "安排", "value": value,
                                             "source_ids": member["source_ids"], "sensitivity": "normal", "time_scope": "current"}]}
                        for i, member in enumerate(task["members"])
                    ]}
                else:
                    payload = {"facts": [{"attr": "安排", "value": value,
                                          "source_ids": task["source_ids"], "sensitivity": "normal", "time_scope": "current"}]}
                results.append(validate(payload, task))
            return results

        def close(self):
            pass

    return Runner


def run_args(input_dir, output_dir, stage="all", **overrides):
    return SimpleNamespace(stage=stage, dry_run=False, input_dir=input_dir,
                           output_dir=output_dir, max_context_tokens=272000, token_encoding="o200k_base",
                           tokenizer_file=None, max_records=20, max_summary_groups=30, max_summary_records=300,
                           max_part_records=30, **overrides)


@pytest.mark.parametrize("stagewise", [False, True])
@pytest.mark.parametrize("extra_sample", [None, "same_peer", "other_peer"])
def test_full_pipeline_keeps_inputs_and_joins_profile_goals_with_events(tmp_path, monkeypatch, stagewise,
                                                                       extra_sample):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    _, profile = write_input(input_dir)
    if extra_sample:
        extra = {**profile, "id": "sample2"} if extra_sample == "same_peer" else {**profile, "chat_with_id": "P2"}
        organize.save(input_dir / "state_people" / "peer.json", [profile, extra])
    before = {p: p.read_bytes() for p in input_dir.rglob("*.json")}
    seen = []
    def fields_for(rid):
        return [field(5), field(6)] if rid.startswith("S:") else [field(5), field(7)]

    monkeypatch.setattr(organize, "LLMTasks", fake_runner(fields_for, seen))
    for stage in organize.STAGES if stagewise else ("all",):
        organize.run(run_args(input_dir, output_dir, stage))
    output = organize.load(output_dir / "organized_memories.json")
    assert organize.load(output_dir / "domains.json") == {"domains": DOMAINS, "tags": {"出行旅行": 1}}
    assert "topics" not in output
    assert [task["domain"] for task in seen if task["stage"] == "summarize" and task["dim"] == 5] == ["出行旅行"]
    by_dim = {dim: [row for row in output["facts"] if row["dim"] == dim] for dim in (5, 6, 7)}
    assert len(output["facts"]) == sum(map(len, by_dim.values()))
    [fact] = by_dim[5]
    # Relations stay per peer, with no domain; events stay per domain.
    peers = ["P1", "P2"] if extra_sample == "other_peer" else ["P1"]
    assert [(row["domain"], row.get("peer")) for row in by_dim[6]] == [(None, peer) for peer in peers]
    assert [(row["domain"], row.get("peer"), len(row["source_ids"])) for row in by_dim[7]] == [("出行旅行", None, 1)]
    assert (fact["dim"], fact["domain"]) == (5, "出行旅行") and "topic" not in fact
    assert len(fact["source_ids"]) == 2 + bool(extra_sample)
    assert fact["support_count"] == 1 + bool(extra_sample)
    assert fact["source_confidence_mean"] == 3
    assert fact["source_importance_mean"] == 2
    assert output["unused"] == [] and output["unassigned_ids"] == []
    assert output["domains"] == DOMAINS
    assert {row["kind"] for row in output["sources"].values()} == {"S", "ES"}
    assert sorted(output["peers"]) == (["P1", "P2"] if extra_sample == "other_peer" else ["P1"])
    assert {row["peer"] for row in output["sources"].values()} <= set(output["peers"])
    assert fact["sensitivity"] == "normal"
    assert fact["time_scope"] == "current"
    assert {row["sensitivity"] for row in output["sources"].values()} == {"private"}
    summaries = [task["prompt"] for task in seen if task["stage"] == "summarize"]
    assert summaries and all('"sensitivity":"private"' in prompt for prompt in summaries)
    assert all(prompts.TIME_SCOPE_RULES in prompt for prompt in summaries)
    assert all(prompts.SENSITIVITY_RULES in task["prompt"] for task in seen if task["stage"] in ("classify", "summarize"))
    classify = [task["prompt"] for task in seen if task["stage"] == "classify"]
    assert classify and all(prompts.domain_block(DOMAINS) in prompt for prompt in classify)
    assert "出行旅行 " in next(task["prompt"] for task in seen if task["stage"] == "domains")
    assert "RAW_CHAT" not in "\n".join(task["prompt"] for task in seen)
    assert all(p.read_bytes() == content for p, content in before.items())


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
    for stage in ("prepare", "domains", "peers", "classify"):
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
        {"id": rid, "fields": [], "sensitivity": "private"} for task in tasks for rid in task["ids"]
    ]


def test_peer_memories_keep_mentions_state_first_then_important_events_in_time_order():
    rows = [{**record(rid, kind=kind), "peer": peer, "content": content, "importance": importance,
             "sample_time": time}
            for rid, kind, peer, content, importance, time in [
                ("e1", "ES", "P10", "B与A吃饭。", 2, "2024-03"),
                ("e2", "ES", "P10", "B请A帮忙。", 4, "2024-02"),
                ("s1", "S", "P10", "B与A是同事。", 2, "2024-04"),
                ("e3", "ES", "P10", "B看了A股行情。", 5, "2024-01"),
                ("e4", "ES", "P2", "B与A约饭。", 2, "2024-01"),
                ("e5", "ES", "P3", "B在学习。", 5, "2024-01"),
            ]]
    selected = organize.peer_memories(rows, limit=2)
    assert list(selected) == ["P2", "P10"]
    assert [row["id"] for row in selected["P10"]] == ["e2", "s1"]
    assert [row["id"] for row in selected["P2"]] == ["e4"]


@pytest.mark.parametrize("payload,expected", [
    ({"relation": " B的 同事 "}, "B的 同事"),
    ({"relation": ""}, ""),
    ({}, ValueError),
    ({"relation": "同事", "note": "x"}, ValueError),
    ({"relation": 1}, ValueError),
    ({"relation": "关" * 101}, ValueError),
])
def test_peer_relation_is_one_short_line(payload, expected):
    if expected is ValueError:
        with pytest.raises(ValueError):
            organize.validate_peer(payload)
    else:
        assert organize.validate_peer(payload) == expected


def test_peer_relations_reach_classify_prompts_and_organized_peers(tmp_path, monkeypatch):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    sample, profile = write_input(input_dir)
    profile["state_memories"].append(
        {"type": "goal", "content": "B打算和大学同学A一起去郊游。", "tags": ["出行旅行"], "importance": 2, "confidence": 3}
    )
    organize.save(input_dir / "state_people" / "peer.json", [profile, {**profile, "chat_with_id": "P2",
                                                                       "state_memories": profile["state_memories"][:1]}])
    seen = []
    monkeypatch.setattr(organize, "LLMTasks", fake_runner(lambda rid: [field(5)], seen))
    organize.run(run_args(input_dir, output_dir, "prepare"))
    organize.run(run_args(input_dir, output_dir, "domains"))
    with pytest.raises(ValueError, match="peers stage"):
        organize.run(run_args(input_dir, output_dir, "classify"))
    for stage in ("peers", "classify", "summarize"):
        organize.run(run_args(input_dir, output_dir, stage))
    requests = [task for task in seen if task["stage"] == "peers"]
    assert [task["peer"] for task in requests] == ["P1"]
    assert requests[0]["prompt"].startswith(prompts.PEER_RULES)
    assert "B打算和大学同学P1一起去郊游。" in requests[0]["prompt"] and "B计划周末去郊游" not in requests[0]["prompt"]
    assert organize.load(output_dir / "peer_relations.json") == {"P1": "B的 朋友"}
    classify = [task["prompt"] for task in seen if task["stage"] == "classify"]
    block = prompts.domain_block(DOMAINS) + "\n## 聊天对象\nP1：B的 朋友\n"
    assert classify and all(block in prompt for prompt in classify)
    assert all(prompt.index(block) < prompt.index("记忆：") for prompt in classify)
    output = organize.load(output_dir / "organized_memories.json")
    assert {peer: info["relation"] for peer, info in output["peers"].items()} == {"P1": "B的 朋友", "P2": ""}


@pytest.mark.parametrize("split", [False, True])
def test_relation_summary_isolates_peers_through_split_and_reduce(split):
    rows = {f"{peer}-{i}": {**record(f"{peer}-{i}"), "peer": peer,
                          "content": "B与聊天对象是同事，王老师是B的导师。" * (150 if split else 1)}
            for peer in ("P1", "P2") for i in range(3)}
    groups = [{"dim": 6, "peer": peer, "source_ids": [rid for rid, row in rows.items() if row["peer"] == peer]}
              for peer in ("P1", "P2")]
    budget = organize.InputBudget(8192, "o200k_base")
    if split:
        budget.max_tokens = max(budget.count(prompts.summarize_prompt(
            6, [organize.prompt_record(row, include_sample_time=True)], peer=row["peer"],
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
                    {"attr": "关系状态", "value": f"B与{peer}是同事。", "source_ids": task["source_ids"], "sensitivity": "normal", "time_scope": "current"},
                ]}, task))
            return results

    results = organize.summarize_groups(groups, rows, Runner(), budget=budget)
    assert ("reduce" in stages) == split
    if not split:
        assert stages == ["summarize", "summarize"]
    for group, result in zip(groups, results):
        assert result["facts"][0]["source_ids"] == group["source_ids"]


@pytest.mark.parametrize("defect", ["unknown", "empty", "duplicate", "no_sensitivity", "unknown_sensitivity",
                                    "unknown_time_scope"])
def test_summary_rejects_invalid_references(defect):
    result = {"facts": [{"attr": "职业", "value": "B从事算法工作。", "source_ids": ["a"], "sensitivity": "normal", "time_scope": "current"}]}
    if defect == "unknown":
        result["facts"][0]["source_ids"] = ["fake"]
    elif defect == "empty":
        result["facts"][0]["source_ids"] = []
    elif defect == "duplicate":
        result["facts"][0]["source_ids"] = ["a", "a"]
    elif defect == "no_sensitivity":
        del result["facts"][0]["sensitivity"]
    elif defect == "unknown_sensitivity":
        result["facts"][0]["sensitivity"] = "secret"
    else:
        result["facts"][0]["time_scope"] = "permanent"
    with pytest.raises(ValueError):
        organize.validate_summary(result, {"a", "b"})


def bare_runner(cache_path, client):
    runner = organize.LLMTasks.__new__(organize.LLMTasks)
    runner.cache = LLMCache(cache_path)
    runner.budget = organize.InputBudget(8192, "o200k_base")
    runner.options = SimpleNamespace(
        batch_size=2, llm_provider="api", model=None, effort=None, max_tokens=200, timeout=10,
    )
    runner.client = client
    return runner


def test_llm_retry_truncation_and_content_addressed_resume(tmp_path):
    payload = {"items": [{"id": "a", "fields": [field(1)], "sensitivity": "normal"}]}

    class Client:
        calls = 0

        def generate_batch(self, requests):
            requests = list(requests)
            assert all(request.json_schema == organize.CLASSIFY_SCHEMA for request in requests)
            self.calls += 1
            return [LLMResponse(ok=True, parsed_json=payload, finish_reason="length" if self.calls == 1 else "stop") for _ in requests]

    client = Client()
    runner = bare_runner(tmp_path / "cache.sqlite", client)
    tasks = [{"prompt": "classify a", "stage": "classify"}]
    def validator(result, task):
        return organize.validate_classification(result, [record("a")], 2)

    runner.run(tasks, validator)
    assert client.calls == 2
    runner.run(tasks, validator)
    assert client.calls == 2
    # A later run with the same cache, say into a new output directory, asks nothing either.
    assert bare_runner(tmp_path / "cache.sqlite", client).run(tasks * 2, validator) == [payload["items"]] * 2
    assert client.calls == 2
    runner.run([{**tasks[0], "prompt": "updated rules: classify a"}], validator)
    assert client.calls == 3


def test_failed_tasks_are_not_cached_and_a_rerun_asks_only_them(tmp_path):
    asked = []
    broken = {"classify b"}

    class Client:
        def generate_batch(self, requests):
            rows = []
            for request in requests:
                prompt = request.messages[0]["content"]
                asked.append(prompt)
                rid = prompt[-1]
                payload = {"items": [{"id": rid, "fields": [field(1)], "sensitivity": "normal"}]}
                rows.append(LLMResponse(ok=False, error="quota") if prompt in broken else LLMResponse(ok=True, parsed_json=payload, finish_reason="stop"))
            return rows

    tasks = [{"prompt": f"classify {rid}", "stage": "classify", "id": rid} for rid in "ab"]

    def validator(result, task):
        return organize.validate_classification(result, [record(task["id"])], 2)

    runner = bare_runner(tmp_path / "cache.sqlite", Client())
    with pytest.raises(RuntimeError, match=r"1 LLM tasks failed \(quota ×1\)"):
        runner.run(tasks, validator)
    assert asked == ["classify a", "classify b", "classify b"]
    broken.clear()
    assert [result[0]["id"] for result in runner.run(tasks, validator)] == ["a", "b"]
    assert asked[3:] == ["classify b"]


def test_llm_resume_ignores_configuration_changes(tmp_path, monkeypatch):
    from weclone.core.inference import llm_client

    config = tmp_path / "settings.jsonc"
    settings = {
        "codex_exec_args": {
            "model": "model-a", "effort": "low", "batch_size": 2, "max_tokens": 200,
            "timeout": 10, "command": "codex", "sandbox": "read-only",
        },
        "agent_distill_args": {"llm_provider": "codex_exec"},
    }
    config.write_text(json.dumps(settings))
    args = SimpleNamespace(config_path=config, llm_provider=None, output_dir=tmp_path,
                           max_context_tokens=8192, token_encoding="o200k_base", tokenizer_file=None)
    payload = {"items": [{"id": "a", "fields": [field(1)], "sensitivity": "normal"}]}

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

    settings["agent_distill_args"].update(max_samples_per_window=16, max_content_chars=800)
    settings["codex_exec_args"].update(batch_size=7, timeout=30, model="model-b", effort="high", max_tokens=400)
    config.write_text("// edited configuration\n" + json.dumps(settings, indent=2))

    resumed = organize.LLMTasks(args)
    assert resumed.run(tasks, validator) == [payload["items"]]
    assert client.calls == 1
    assert resumed.options.model == "model-b"
    assert resumed.options.effort == "high"

    monkeypatch.setattr(organize, "CLASSIFY_SCHEMA", {
        **organize.CLASSIFY_SCHEMA, "description": "Updated output protocol",
    })
    resumed.run(tasks, validator)
    assert client.calls == 2
    resumed.close()


def summary_fact(source_ids):
    return {"attr": "属性", "value": "同一事实", "source_ids": source_ids, "sensitivity": "normal", "time_scope": "current"}


class FactPerGroupRunner:
    """Answers every summary group, or part of one, with a single fact citing all its sources."""

    def __init__(self):
        self.calls = []

    def run(self, tasks, validate):
        results = []
        for task in tasks:
            members = task.get("members", [task])
            self.calls.append((task["stage"], [len(member["source_ids"]) for member in members]))
            facts = [[summary_fact(member["source_ids"])] for member in members]
            payload = ({"groups": [{"id": i, "facts": rows} for i, rows in enumerate(facts)]} if "members" in task
                       else {"facts": facts[0]})
            results.append(validate(payload, task))
        return results


@pytest.mark.parametrize("dim", [1, 6])
def test_summary_record_limit_splits_group_and_reduces_all_sources(dim):
    rows = [record(str(i)) for i in range(501)]
    runner = FactPerGroupRunner()
    group = {"dim": dim, "peer": "P1", "source_ids": [row["id"] for row in rows]}
    result = organize.summarize_groups([group], {row["id"]: row for row in rows}, runner)
    assert runner.calls == [("summarize", [300]), ("summarize", [201]), ("reduce", [501])]
    assert len(result[0]["facts"]) == 1
    assert set(result[0]["facts"][0]["source_ids"]) == set(group["source_ids"])


def test_parts_share_requests_with_other_groups_of_their_partition():
    rows = {str(i): record(str(i)) for i in range(61)}
    groups = [{"dim": 3, "domain": "工作", "source_ids": list(rows)}, {"dim": 3, "domain": "学业", "source_ids": ["0"]}]
    runner = FactPerGroupRunner()
    results = organize.summarize_groups(groups, rows, runner)
    assert runner.calls == [("summarize", [21, 1]), ("summarize", [21]), ("summarize", [19]), ("reduce", [61])]
    assert len(results[0]["facts"]) == 1
    assert set(results[0]["facts"][0]["source_ids"]) == set(rows)


def test_summary_request_limit_counts_unique_sources_across_groups():
    tasks = [organize.pack_summary(5, [record(str(i)) for i in ids], reduced=False)[0]
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
    batches = organize.pack_summary(5, [record(str(i)) for i in range(4)], reduced=False, max_records=2)
    assert [task["source_ids"] for task in batches] == [["0", "1"], ["2", "3"]]


def test_oversized_group_reduces_across_batches_without_losing_sources():
    rows = [record(str(i), memory_type="goal", sample=str(i)) for i in range(5)]
    for row in rows:
        row["content"] = "B计划出行。" * 180
    stages = []

    class Runner:
        def run(self, tasks, validate):
            results = []
            for task in tasks:
                stages.append(task["stage"])
                payload = {"facts": [{"attr": "出行计划", "value": "B计划出行。", "source_ids": task["source_ids"], "sensitivity": "normal", "time_scope": "current"}]}
                results.append(validate(payload, task))
            return results

    group = {"dim": 5, "domain": "出行旅行", "source_ids": [row["id"] for row in rows]}
    budget = organize.InputBudget(8192, "o200k_base")
    budget.max_tokens = max(budget.count(organize.summarize_prompt(
        5, [organize.prompt_record(row, include_sample_time=True)], domain=group["domain"],
    )) for row in rows)
    result = organize.summarize_groups([group], {row["id"]: row for row in rows}, Runner(), budget=budget, max_groups=1)[0]
    assert stages.count("summarize") > 1
    assert "reduce" in stages
    assert len(result["facts"]) == 1
    assert set(result["facts"][0]["source_ids"]) == {row["id"] for row in rows}


def test_groups_of_different_dimensions_share_llm_batch():
    rows = [record("a"), record("b")]
    groups = [{"dim": dim, "domain": "工作", "source_ids": [rid]} for dim, rid in ((1, "a"), (4, "b"))]
    calls = []

    class Runner:
        def run(self, tasks, validate):
            calls.append(len(tasks))
            return [validate({"facts": [{"attr": "属性", "value": "已有值", "source_ids": task["source_ids"], "sensitivity": "normal", "time_scope": "current"}]}, task) for task in tasks]

    results = organize.summarize_groups(groups, {row["id"]: row for row in rows}, Runner())
    assert calls == [2]
    assert [result["facts"][0]["source_ids"] for result in results] == [["a"], ["b"]]


def test_summary_combines_thirty_groups_and_restores_reordered_results():
    rows = {str(i): record(str(i)) for i in range(31)}
    groups = [{"dim": 1, "domain": f"领域{i}", "source_ids": [str(i)]} for i in range(31)]
    sizes = []

    class Runner:
        def run(self, tasks, validate):
            results = []
            for task in tasks:
                members = task.get("members", [task])
                sizes.append(len(members))
                payloads = [{"id": i, "facts": [{"attr": "属性", "value": "已有事实",
                                                   "source_ids": member["source_ids"], "sensitivity": "normal", "time_scope": "current"}]}
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
    payload = {"groups": [{"id": i, "facts": [{"attr": "属性", "value": "事实", "source_ids": [rid], "sensitivity": "normal", "time_scope": "current"}]} for i, rid in enumerate(("a", "b"))]}
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
    groups = [{"dim": 1, "domain": domain, "source_ids": [rid]} for domain, rid in (("工作", "a"), ("学业", "b"))]

    class Runner:
        calls = 0

        def run(self, tasks, validate):
            self.calls += len(tasks)
            return [validate({"groups": [
                {"id": 0, "facts": [{"attr": "职业", "value": "已有事实", "source_ids": ["b"], "sensitivity": "normal", "time_scope": "current"}]},
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
    tasks = [organize.pack_summary(1, [row], reduced=False, domain=domain)[0] for domain in ("工作", "学业")]
    prompt = organize.batch_summary_prompt(tasks)
    assert prompt.count('"id":"shared"') == 1
    budget = organize.InputBudget(8192, "o200k_base")
    budget.max_tokens = max(budget.count(task["prompt"]) for task in tasks)
    calls = []

    class Runner:
        def run(self, requests, validate):
            calls.extend(requests)
            assert all(budget.fits(task["prompt"]) for task in requests)
            return [validate({"facts": [{"attr": "属性", "value": "已有值", "source_ids": ["shared"], "sensitivity": "normal", "time_scope": "current"}]}, task) for task in requests]

    organize.run_summary_tasks(tasks, Runner(), 30, budget)
    assert len(calls) == 2
    assert all("members" not in task for task in calls)
