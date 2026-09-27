import json
from unittest.mock import Mock

import pytest

from weclone.core.inference.llm_client import LLMResponse
from weclone.data.agent import distill_event, distill_profile
from weclone.data.agent.distill_windows import (
    ChatSample,
    content_chars,
    group_samples,
    render_sample,
    split_window_response,
)
from weclone.prompts.chat_distill import (
    EVENT_EXTRACT_PROMPT,
    build_state_extract_prompt,
    build_window_extract_prompt,
)


def sample(sid, size=20, recent=False):
    item = {
        "id": sid,
        "time": "2025-01-20" if recent else "2025-01-01",
        "messages": [{"role": "assistant", "content": "字" * size}],
    }
    return ChatSample(int(sid), str(sid), item, recent)


def result(sid, task):
    common = {"tags": ["职业发展"], "importance": 2, "confidence": 4}
    if task == "state":
        value = [{"type": "stable_fact", "content": f"画像 {sid}", **common}]
    else:
        value = {
            "surface_events": [{"surface_event": f"事件 {sid}", "event_types": ["daily_event"], **common}]
        }
    return {"sample_id": sid, f"{task}_memories": value}


def response(ids, task):
    return LLMResponse(
        ok=True,
        parsed_json={"results": [result(sid, task) for sid in reversed(ids)]},
        metadata={"usage": {"input_tokens": 100}},
    )


def test_window_limits_recency_and_oversized_singletons():
    samples = [sample(0, 250), sample(1, 150), sample(2, 1), sample(3, 401)]
    samples += [sample(i, 1, recent=True) for i in range(4, 14)]
    groups = group_samples(samples, max_samples=8, max_content_chars=400)
    assert [[s.sample_id for s in g] for g in groups] == [
        ["0", "1"],
        ["2"],
        [str(i) for i in range(4, 12)],
        ["12", "13"],
        ["3"],
    ]
    assert groups[-1][0].item["messages"][0]["content"] == "字" * 401


def test_plain_format_counts_only_normalized_chat_content():
    item = {
        "time": "2025-01-01",
        "messages": [
            {"role": "system", "content": "ignored" * 100},
            {"role": "user", "content": " 甲\r\n乙 "},
            {"role": "assistant", "content": " 丙 "},
        ],
    }
    assert content_chars(item) == 4
    assert render_sample(item, sample_id="12", target_role="user", include_time=True) == (
        "#12\ntime: 2025-01-01\nB:甲\n乙\nA:丙"
    )


@pytest.mark.parametrize("recent", [False, True])
def test_window_prompt_preserves_original_rules(recent):
    state_rules = build_state_extract_prompt(include_current_state=recent).split(
        "只输出 JSON，不要输出解释文字。", 1
    )[0]
    assert state_rules in build_window_extract_prompt("state", include_current_state=recent)
    event_rules = EVENT_EXTRACT_PROMPT.split("输出 JSON：\n", 1)[0].replace(
        "10. 只输出 JSON，不要解释。没有对应事件时，不输出对应顶层字段。事件对象内没有值的可选字段也直接省略，不要输出空字符串、空数组或 null。",
        "10. 没有对应事件时，在 event_memories 内省略对应事件数组。事件对象内没有值的可选字段直接省略，不输出空字符串、空数组或 null。",
    )
    assert event_rules in build_window_extract_prompt("event")


@pytest.mark.parametrize("defect", ["duplicate", "missing", "unknown", "tags", "type", "recent", "truncated"])
def test_invalid_state_windows_are_rejected(defect):
    samples = [sample(1), sample(2)]
    r = response(["1", "2"], "state")
    rows = r.parsed_json["results"]
    if defect == "duplicate":
        rows[1]["sample_id"] = rows[0]["sample_id"]
    elif defect == "missing":
        rows.pop()
    elif defect == "unknown":
        rows[0]["sample_id"] = "999"
    elif defect == "tags":
        del rows[0]["state_memories"][0]["tags"]
    elif defect == "type":
        rows[0]["state_memories"][0]["type"] = []
    elif defect == "recent":
        rows[0]["state_memories"][0]["type"] = "current_state"
    else:
        r.finish_reason = "length"
    with pytest.raises(ValueError):
        split_window_response(r, samples, "state")


@pytest.mark.parametrize("task,module", [("state", distill_profile), ("event", distill_event)])
def test_process_writeback_retry_resume_and_input_preservation(tmp_path, task, module):
    source = tmp_path / "input.json"
    items = [sample(i, 100).item for i in range(9)]
    source.write_text(json.dumps(items))
    original = source.read_bytes()
    state = {"entries": {}}
    received = []
    failed_once = False

    def generate(requests):
        nonlocal failed_once
        requests = list(requests)
        received.append(requests)
        assert len(requests) <= 2
        responses = []
        for request in requests:
            assert request.max_tokens is None
            assert request.model == "configured-model"
            ids = request.metadata["sample_ids"]
            r = response(ids, task)
            if ids[0] == "0" and not failed_once:
                failed_once = True
                r.parsed_json["results"][1]["sample_id"] = r.parsed_json["results"][0]["sample_id"]
            responses.append(r)
        return responses

    client = Mock(generate_batch=Mock(side_effect=generate))
    options = {
        "output_dir": tmp_path / "output",
        "target_role": "assistant",
        "provider": "codex_exec",
        "model": "configured-model",
        "effort": "low",
        "client": client,
        "max_tokens": None,
        "batch_size": 2,
        "limit_records": None,
        "overwrite": False,
        "dry_run": False,
        "state": state,
        "state_path": tmp_path / "checkpoint.json",
        "indent": 2,
        "progress_factory": Mock(),
    }
    assert module.process_file(source, **options) == (9, 4)
    assert [len(wave) for wave in received] == [2, 1, 1]
    assert received[1][0] is received[0][0]
    output = module.output_path_for(options["output_dir"], source)
    saved = json.loads(output.read_text())
    for item in saved:
        expected = result(str(item["id"]), task)[f"{task}_memories"]
        assert item[f"{task}_memories"] == expected
    assert source.read_bytes() == original
    checkpoint = json.loads(options["state_path"].read_text())
    assert len(checkpoint["entries"]) == 9
    assert all(r["status"] == "done" for r in checkpoint["entries"].values())
    assert sum(len(w["attempts"]) for w in checkpoint["windows"].values()) == 4
    first = checkpoint["entries"][distill_profile.state_key(source, "0")]
    assert "results" not in first["payload"]["result"]
    if task == "state":
        assert first["payload"]["result"]["memories"] == result("0", task)["state_memories"]
    output.unlink()
    options["state"] = checkpoint
    client.generate_batch.reset_mock()
    assert module.process_file(source, **options) == (0, 0)
    client.generate_batch.assert_not_called()
    assert json.loads(output.read_text()) == saved


@pytest.mark.parametrize("task,module", [("state", distill_profile), ("event", distill_event)])
def test_failed_window_never_becomes_successful_empty_memories(tmp_path, task, module):
    source = tmp_path / "input.json"
    source.write_text(json.dumps([sample(0).item, sample(1).item]))
    state = {"entries": {}}
    client = Mock()
    client.generate_batch.side_effect = lambda requests: [
        LLMResponse(ok=True, parsed_json={"results": []}) for _ in requests
    ]
    output_dir = tmp_path / "output"
    assert module.process_file(
        source,
        output_dir=output_dir,
        target_role="assistant",
        provider="codex_exec",
        model="fixture",
        effort="low",
        client=client,
        max_tokens=None,
        batch_size=30,
        limit_records=None,
        overwrite=False,
        dry_run=False,
        state=state,
        state_path=tmp_path / "checkpoint.json",
        indent=2,
        progress_factory=Mock(),
    ) == (2, 2)
    assert client.generate_batch.call_count == 2
    assert all(r["status"] == "failed" and r["payload"]["result"] is None for r in state["entries"].values())
    assert not module.output_path_for(output_dir, source).exists()


@pytest.mark.parametrize("task,module", [("state", distill_profile), ("event", distill_event)])
def test_existing_input_memories_are_written_to_separate_output(tmp_path, task, module):
    item = sample(0).item
    item[f"{task}_memories"] = result("0", task)[f"{task}_memories"]
    source = tmp_path / "input.json"
    source.write_text(json.dumps([item]))
    output_dir = tmp_path / "output"
    client = Mock()
    assert module.process_file(
        source,
        output_dir=output_dir,
        target_role="assistant",
        provider="codex_exec",
        model="fixture",
        effort="low",
        client=client,
        max_tokens=None,
        batch_size=30,
        limit_records=None,
        overwrite=False,
        dry_run=False,
        state={"entries": {}},
        state_path=tmp_path / "checkpoint.json",
        indent=2,
        progress_factory=Mock(),
    ) == (0, 0)
    client.generate_batch.assert_not_called()
    assert json.loads(module.output_path_for(output_dir, source).read_text()) == [item]


@pytest.mark.parametrize("max_tokens", [None, 4096])
def test_window_config_preserves_model_concurrency_and_optional_token_limit(tmp_path, max_tokens):
    config = {
        "agent_distill_args": {
            "llm_provider": "codex_exec",
            "model": "configured-model",
            "effort": "low",
            "command": "codex",
            "sandbox": "read-only",
            "batch_size": 30,
            "timeout": 120,
            "max_tokens": max_tokens,
        },
    }
    path = tmp_path / "settings.jsonc"
    path.write_text(json.dumps(config))
    args = distill_profile.default_args()
    args.config_path = path
    distill_profile.resolve_llm_args(args)
    assert (args.max_samples_per_window, args.max_content_chars) == (8, 400)
    assert (args.model, args.batch_size, args.max_tokens) == ("configured-model", 30, max_tokens)


@pytest.mark.parametrize("task,module", [("state", distill_profile), ("event", distill_event)])
def test_legacy_checkpoint_restores_per_sample_results(tmp_path, task, module):
    source = tmp_path / "input.json"
    items = [sample(0).item, sample(1, recent=True).item]
    items[1][f"{task}_memories"] = [] if task == "state" else {}
    source.write_text(json.dumps(items))
    value = result("0", task)[f"{task}_memories"]
    expected = json.loads(json.dumps(value))
    if task == "state":
        value.append({"type": "current_state", "content": "已过期的近期状态"})
        value = {"memories": value}
    state = {
        "entries": {
            distill_profile.state_key(source, "0"): {
                "status": "done",
                "payload": {"result": value, "response": {"ok": True}},
            }
        }
    }
    client = Mock()
    output_dir = tmp_path / "output"
    assert module.process_file(
        source,
        output_dir=output_dir,
        target_role="assistant",
        provider="codex_exec",
        model="fixture",
        effort="low",
        client=client,
        max_tokens=None,
        batch_size=30,
        limit_records=None,
        overwrite=False,
        dry_run=False,
        state=state,
        state_path=tmp_path / "checkpoint.json",
        indent=2,
        progress_factory=Mock(),
    ) == (0, 0)
    client.generate_batch.assert_not_called()
    assert (
        json.loads(module.output_path_for(output_dir, source).read_text())[0][f"{task}_memories"] == expected
    )


@pytest.mark.parametrize("module", [distill_profile, distill_event])
def test_dry_run_previews_a_whole_window_without_writes(tmp_path, capsys, module):
    source = tmp_path / "input.json"
    source.write_text(json.dumps([sample(i, 100).item for i in range(5)]))
    client = Mock()
    output_dir, checkpoint = tmp_path / "output", tmp_path / "checkpoint.json"
    assert module.process_file(
        source,
        output_dir=output_dir,
        target_role="assistant",
        provider="codex_exec",
        model="fixture",
        effort="low",
        client=client,
        max_tokens=None,
        batch_size=30,
        limit_records=None,
        overwrite=False,
        dry_run=True,
        state={"entries": {}},
        state_path=checkpoint,
        indent=2,
        progress_factory=Mock(),
    ) == (4, 0)
    preview = capsys.readouterr().out
    assert "#0\n" in preview and "#3\n" in preview and "#4\n" not in preview
    client.generate_batch.assert_not_called()
    assert not output_dir.exists() and not checkpoint.exists()
