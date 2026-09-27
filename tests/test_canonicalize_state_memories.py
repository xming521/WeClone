import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from weclone.core.inference import llm_client
from weclone.data.agent import canonicalize_state_memories as canonicalize


@pytest.fixture(autouse=True)
def isolated_default_config(tmp_path, monkeypatch):
    config = tmp_path / "default_settings.jsonc"
    config.write_text('{"agent_distill_args": {"merge_enabled": true}}', encoding="utf-8")
    monkeypatch.setattr(canonicalize, "DEFAULT_CONFIG_PATH", config)


def write_grouping(directory, name="person"):
    records = [
        {
            "memory_id": f"{i}:0",
            "type": "stable_fact",
            "content": f"B 的画像事实 {i}",
            "tags": ["工作"],
            "importance": 3 if i != 6 else 1,
            "confidence": 4,
            "sample_time": "2026-09-01",
            "prefilter_status": "candidate",
        }
        for i in range(1, 7)
    ]
    path = directory / f"{name}.prefilter_candidates.json"
    path.write_text(json.dumps({
        "records": records,
        "candidate_groups": [{
            "group_id": "g1",
            "memory_ids": [r["memory_id"] for r in records[:4]],
            "members": records[:4],
        }],
        "singletons": ["5:0", "6:0"],
    }), encoding="utf-8")
    return path


def merge_response():
    return llm_client.LLMResponse(ok=True, parsed_json={
        "canonical_memories": [{
            "content": "B 的两条记录描述同一工作事实",
            "source_ids": ["1:0", "2:0"],
            "status": "active",
            "time_scope": "long_term",
        }],
        "source_decisions": [
            {"source_id": "3:0", "decision": "archived", "reason": "仅留档"},
            {"source_id": "4:0", "decision": "discarded", "reason": "缺乏画像价值"},
        ],
    })


@pytest.fixture
def backend(tmp_path, monkeypatch):
    config = tmp_path / "config.jsonc"
    config.write_text(json.dumps({"codex_exec_args": {
        "model": "test-model", "effort": "high", "batch_size": 3,
        "timeout": 120, "command": "test-command", "sandbox": "read-only",
    }}))
    client = Mock()
    client.generate_batch.side_effect = lambda requests: [merge_response() for _ in requests]
    factory = Mock(return_value=client)
    monkeypatch.setattr(llm_client, "build_llm_client", factory)
    return config, client, factory


@pytest.mark.parametrize("directory_mode", [False, True])
def test_cli_pipeline_and_resume(tmp_path, backend, directory_mode):
    config, client, factory = backend
    grouping_dir = tmp_path / "grouping"
    grouping_dir.mkdir()
    names = ["alice", "bob"] if directory_mode else ["alice"]
    inputs = [write_grouping(grouping_dir, name) for name in names]
    originals = [path.read_bytes() for path in inputs]
    output = tmp_path / "output"
    common = ["--grouping-path", str(grouping_dir if directory_mode else inputs[0]),
              "--output-dir", str(output)]
    canonicalize.main(["prepare", *common])
    canonicalize.main(["run", *common, "--config-path", str(config)])
    assert client.generate_batch.call_count == len(names)
    assert factory.call_args.kwargs["max_workers"] == 3
    canonicalize.main(["run", *common, "--config-path", str(config)])
    assert client.generate_batch.call_count == len(names)
    canonicalize.main(["apply", *common])
    for name in names:
        tasks = canonicalize.read_jsonl(output / f"{name}.llm_merge_tasks.jsonl")
        assert len(tasks) == 1
        assert [r["id"] for r in tasks[0]["input"]["records"]] == ["1:0", "2:0", "3:0", "4:0"]
        bank = json.loads((output / f"{name}.canonical_memories.json").read_text())
        assert bank["unresolved_source_ids"] == []
        assert {item["source_id"] for item in bank["archive_items"]} == {"3:0", "4:0", "6:0"}
        assert [m["source_ids"] for m in bank["memory_provenance"]] == [["1:0", "2:0"], ["5:0"]]
    assert [path.read_bytes() for path in inputs] == originals


def test_failed_run_is_nonzero_and_retries_from_checkpoint(tmp_path, backend):
    config, client, _ = backend
    grouping = write_grouping(tmp_path)
    common = ["--grouping-path", str(grouping), "--output-dir", str(tmp_path)]
    canonicalize.main(["prepare", *common])
    client.generate_batch.side_effect = lambda requests: [
        llm_client.LLMResponse(ok=False, error="synthetic failure") for _ in requests
    ]
    with pytest.raises(SystemExit) as exc:
        canonicalize.main(["run", *common, "--config-path", str(config)])
    assert exc.value.code == 1
    checkpoint = json.loads((tmp_path / "person.llm_merge_state.json").read_text())
    assert checkpoint["entries"]["g1"]["status"] == "failed"
    assert canonicalize.read_jsonl(tmp_path / "person.llm_merge_results.jsonl") == []
    client.generate_batch.side_effect = lambda requests: [merge_response() for _ in requests]
    canonicalize.main(["run", *common, "--config-path", str(config)])
    assert len(canonicalize.read_jsonl(tmp_path / "person.llm_merge_results.jsonl")) == 1


def test_dry_run_preserves_checkpoint_and_token_configuration(tmp_path, backend):
    config, client, factory = backend
    grouping = write_grouping(tmp_path)
    common = ["--grouping-path", str(grouping), "--output-dir", str(tmp_path)]
    canonicalize.main(["prepare", *common])
    state = tmp_path / "person.llm_merge_state.json"
    state.write_text('{"version": 1, "entries": {}}')
    original = state.read_bytes()
    canonicalize.main(["run", *common, "--config-path", str(config), "--dry-run", "--overwrite"])
    assert state.read_bytes() == original
    assert not (tmp_path / "person.llm_merge_results.jsonl").exists()
    factory.assert_not_called()
    requests_seen = []

    def generate(requests):
        requests_seen.extend(requests)
        return [merge_response() for _ in requests_seen]

    client.generate_batch.side_effect = generate
    canonicalize.main(["run", *common, "--config-path", str(config)])
    assert requests_seen[0].max_tokens is None
    assert requests_seen[0].model == "test-model"


@pytest.mark.parametrize("command,option", [
    ("prepare", "--output-path"), ("run", "--tasks-path"),
    ("apply", "--results-path"),
])
def test_directory_rejects_shared_file_overrides(tmp_path, command, option):
    write_grouping(tmp_path)
    with pytest.raises(SystemExit) as exc:
        canonicalize.main([command, "--grouping-path", str(tmp_path), option, str(tmp_path / "shared")])
    assert exc.value.code == 2
    assert not (tmp_path / "shared").exists()


@pytest.mark.parametrize("module_mode", [False, True])
def test_executable_entrypoint(module_mode):
    target = ["-m", canonicalize.__name__] if module_mode else [canonicalize.__file__]
    process = subprocess.run([sys.executable, *target, "--help"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[1], timeout=30)
    assert process.returncode == 0
    assert "{prepare,run,apply}" in process.stdout


@pytest.mark.parametrize("memory_type", ["goal", "current_state", "stable_fact"])
@pytest.mark.parametrize("temporal_fields,expected_scope,expected_status", [
    ({}, "", "none"),
    ({"time_scope": "time_sensitive"}, "time_sensitive", "none"),
    ({"time_scope": "historical_only", "status": "expired"}, "historical_only", "expired"),
    ({"status": "historical"}, "", "historical"),
])
def test_optional_temporality_never_expires_from_record_age(
    tmp_path, memory_type, temporal_fields, expected_scope, expected_status,
):
    grouping = write_grouping(tmp_path)
    data = json.loads(grouping.read_text())
    for record in data["records"]:
        record["type"] = memory_type
        record["sample_time"] = "2020-01-01"
    grouping.write_text(json.dumps(data))
    response = merge_response().parsed_json
    memory = response["canonical_memories"][0]
    del memory["status"]
    del memory["time_scope"]
    memory.update(temporal_fields)
    canonicalize.write_jsonl(tmp_path / "person.llm_merge_results.jsonl", [
        {"group_id": "g1", "result": response},
    ])
    common = ["--grouping-path", str(grouping), "--output-dir", str(tmp_path)]
    canonicalize.main(["apply", *common])
    bank = json.loads((tmp_path / "person.canonical_memories.json").read_text())
    merged, singleton = bank["canonical_memories"]
    assert bank["memory_provenance"][0]["time_scope"] == expected_scope
    assert merged["status"] == expected_status
    assert "valid_from" not in merged and "valid_to" not in merged
    assert bank["memory_provenance"][1]["time_scope"] == ""
    assert singleton["status"] == "none"
    assert "valid_from" not in singleton and "valid_to" not in singleton


def test_split_parent_links_and_derived_fields(tmp_path):
    grouping = write_grouping(tmp_path)
    data = json.loads(grouping.read_text())
    data["records"][0].update(importance=4, confidence=2, sample_time="2026-08-01")
    data["records"][1].update(importance=2, confidence=4, sample_time="2026-09-01")
    grouping.write_text(json.dumps(data))
    result = merge_response().parsed_json
    result["canonical_memories"] = [
        {"content": "目标用户的工作事实", "source_ids": ["1:0", "2:0"]},
        {"content": "目标用户工作中的具体事实", "source_ids": ["2:0"], "parent_index": 0},
    ]
    canonicalize.write_jsonl(tmp_path / "person.llm_merge_results.jsonl", [
        {"group_id": "g1", "result": result},
    ])
    common = ["--grouping-path", str(grouping), "--output-dir", str(tmp_path)]
    canonicalize.main(["apply", *common])
    bank = json.loads((tmp_path / "person.canonical_memories.json").read_text())
    parent, child, singleton = bank["canonical_memories"]
    parent_source, child_source, _ = bank["memory_provenance"]
    assert parent_source["children_ids"] == [child_source["id"]]
    assert child_source["parent_id"] == parent_source["id"]
    assert [item["memory_index"] for item in bank["memory_provenance"]] == [0, 1, 2]
    assert parent["importance"] == 4
    assert parent["confidence"] == 2
    assert parent["source_time_range"] == {"first": "2026-08-01T00:00:00", "last": "2026-09-01T00:00:00"}
    assert child["source_time_range"] == {"first": "2026-09-01T00:00:00", "last": "2026-09-01T00:00:00"}
    assert all(set(memory) == {"content", "type", "tags", "importance", "confidence", "preference_type", "status", "source_time_range"} for memory in (parent, child, singleton))
    assert child["importance"] == 2
    assert child["confidence"] == 4
    removed = {"relation", "supersedes", "conflicts_with", "retrieval_policy", "subsumes", "expires_at"}
    assert not any(removed & memory.keys() for memory in (parent, child, singleton))
    assert {decision["source_id"] for decision in bank["source_decisions"]} == {"3:0", "4:0"}
    assert all("canonical_index" not in decision for decision in bank["source_decisions"])
    task = canonicalize.build_merge_task(data["candidate_groups"][0])
    assert set(task["input"]["records"][0]) == {"id", "type", "content", "time"}


@pytest.mark.parametrize("defect", [
    "missing_source", "retained_and_excluded", "duplicate_decision", "outside_group",
    "self_parent", "cyclic_parent", "valid_from", "valid_to", "obsolete_field",
])
def test_invalid_protocol_fails_run_and_apply(tmp_path, backend, defect):
    config, client, _ = backend
    grouping = write_grouping(tmp_path)
    common = ["--grouping-path", str(grouping), "--output-dir", str(tmp_path)]
    canonicalize.main(["prepare", *common])
    response = merge_response()
    result = response.parsed_json
    memory = result["canonical_memories"][0]
    if defect == "missing_source":
        result["source_decisions"].pop()
    elif defect == "retained_and_excluded":
        result["source_decisions"][0]["source_id"] = "1:0"
    elif defect == "duplicate_decision":
        result["source_decisions"].append(dict(result["source_decisions"][0]))
    elif defect == "outside_group":
        memory["source_ids"].append("5:0")
    elif defect == "self_parent":
        memory["parent_index"] = 0
    elif defect == "cyclic_parent":
        memory["parent_index"] = 1
        result["canonical_memories"].append({"content": "子事实", "source_ids": ["2:0"], "parent_index": 0})
    elif defect in {"valid_from", "valid_to"}:
        memory[defect] = "2026-09-15"
    else:
        memory["last_evidence_time"] = "2099-01-01"
    client.generate_batch.side_effect = lambda requests: [response for _ in requests]
    with pytest.raises(SystemExit) as exc:
        canonicalize.main(["run", *common, "--config-path", str(config)])
    assert exc.value.code == 1
    checkpoint = json.loads((tmp_path / "person.llm_merge_state.json").read_text())
    assert checkpoint["entries"]["g1"]["status"] == "failed"
    assert checkpoint["entries"]["g1"]["last_error"]
    assert canonicalize.read_jsonl(tmp_path / "person.llm_merge_results.jsonl") == []
    canonicalize.write_jsonl(tmp_path / "person.llm_merge_results.jsonl", [{"group_id": "g1", "result": result}])
    with pytest.raises(ValueError):
        canonicalize.main(["apply", *common])
    assert not (tmp_path / "person.canonical_memories.json").exists()


@pytest.mark.parametrize("directory_mode", [False, True])
def test_skip_merge_does_not_read_convert_or_write_memories(tmp_path, monkeypatch, directory_mode):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "person.json"
    source.write_text('[{"id": 12, "state_memories": []}]')
    original = source.read_bytes()
    output = tmp_path / "output"
    factory = Mock(side_effect=AssertionError("skip-merge must not call an LLM"))
    reader = Mock(side_effect=AssertionError("skip-merge must not read memories or results"))
    monkeypatch.setattr(llm_client, "build_llm_client", factory)
    monkeypatch.setattr(canonicalize, "load_json", reader)
    monkeypatch.setattr(canonicalize, "read_jsonl", reader)
    canonicalize.main([
        "apply", "--skip-merge", "--input-path", str(inputs if directory_mode else source),
        "--output-dir", str(output),
    ])
    assert source.read_bytes() == original
    assert not output.exists()
    factory.assert_not_called()
    reader.assert_not_called()


@pytest.mark.parametrize("command", ["prepare", "run", "apply"])
def test_config_disables_entire_merge_stage_before_input_or_model(tmp_path, monkeypatch, command):
    from weclone.data.agent import group_state_memories

    config = tmp_path / "settings.jsonc"
    config.write_text('{"agent_distill_args": {"merge_enabled": false}}')
    output = tmp_path / "output"
    fail = Mock(side_effect=AssertionError("Disabled merge stage must not process inputs or start models"))
    monkeypatch.setattr(canonicalize, "run_command", fail)
    monkeypatch.setattr(group_state_memories, "ensure_embedding_service", fail)
    monkeypatch.setattr(group_state_memories, "input_paths_for", fail)
    canonicalize.main([command, "--config-path", str(config),
                       "--grouping-path", str(tmp_path / "missing"), "--output-dir", str(output)])
    fail.assert_not_called()
    assert not output.exists()


def test_config_enables_prepare(tmp_path):
    config = tmp_path / "settings.jsonc"
    config.write_text('{"agent_distill_args": {"merge_enabled": true}}')
    grouping = write_grouping(tmp_path)
    canonicalize.main(["prepare", "--config-path", str(config),
                       "--grouping-path", str(grouping), "--output-dir", str(tmp_path)])
    assert len(canonicalize.read_jsonl(tmp_path / "person.llm_merge_tasks.jsonl")) == 1


def test_grouping_runs_when_llm_merge_is_disabled(tmp_path, monkeypatch):
    from weclone.data.agent import group_state_memories

    config = tmp_path / "settings.jsonc"
    config.write_text('{"agent_distill_args": {"merge_enabled": false}}')
    source = tmp_path / "person.json"
    source.write_text(json.dumps([{"id": 1, "time": "2025-01-01", "state_memories": [
        {"type": "preference", "content": "B偏好简短直接的回应。", "tags": ["沟通"],
         "importance": 3, "confidence": 4},
        {"type": "preference", "content": "B喜欢直接简短的回复。", "tags": ["沟通"],
         "importance": 3, "confidence": 4},
    ]}]))
    original = source.read_bytes()
    output = tmp_path / "output"
    service = Mock(return_value=({}, None))
    embedding = Mock(return_value=[[1.0, 0.0], [1.0, 0.0]])
    monkeypatch.setattr(group_state_memories, "ensure_embedding_service", service)
    monkeypatch.setattr(group_state_memories, "embed_texts", embedding)
    monkeypatch.setattr(sys, "argv", ["group_state_memories", "--config-path", str(config),
                                     "--input-path", str(source), "--output-dir", str(output)])
    group_state_memories.main()
    data = json.loads((output / "person.prefilter_candidates.json").read_text())
    assert len(data["records"]) == 2
    assert len(data["candidate_groups"]) == 1
    assert set(data["candidate_groups"][0]["memory_ids"]) == {"1:0", "1:1"}
    assert source.read_bytes() == original
    service.assert_called_once()
    embedding.assert_called_once()
