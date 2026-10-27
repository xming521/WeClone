import json
from unittest.mock import Mock

import pytest

from weclone.core.inference.llm_client import LLMResponse
from weclone.data.agent import distill_event, distill_profile
from weclone.data.agent.distill_profile import FileStats
from weclone.data.agent.distill_windows import (
    ChatSample,
    content_chars,
    group_samples,
    render_sample,
    sample_cache_key,
    split_window_response,
)
from weclone.data.agent.llm_cache import LLMCache
from weclone.prompts.distill_memory import (
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


def chats(count, size=20):
    """Samples with different text, so none shares another's cached answer."""
    items = [sample(i, size).item for i in range(count)]
    for i, item in enumerate(items):
        item["messages"][0]["content"] = f"{i}" + "字" * (size - len(str(i)))
    return items


def file_options(tmp_path, client, **changes):
    options = {
        "output_dir": tmp_path / "output",
        "target_role": "assistant",
        "provider": "codex_exec",
        "model": "configured-model",
        "effort": "low",
        "client": client,
        "cache": LLMCache(tmp_path / "cache.sqlite"),
        "max_tokens": None,
        "batch_size": 2,
        "limit_records": None,
        "dry_run": False,
        "indent": 2,
        "progress_factory": Mock(),
    }
    return {**options, **changes}


@pytest.mark.parametrize("task,module", [("state", distill_profile), ("event", distill_event)])
def test_process_writeback_retry_resume_and_input_preservation(tmp_path, task, module):
    source = tmp_path / "input.json"
    items = chats(9, 100)
    source.write_text(json.dumps(items))
    original = source.read_bytes()
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
    options = file_options(tmp_path, client)
    assert module.process_file(source, **options) == FileStats(cached=0, pending=9, windows=3, failed=0, calls=4)
    assert [len(wave) for wave in received] == [2, 1, 1]
    assert received[1][0] is received[0][0]
    output = distill_profile.output_path_for(options["output_dir"], source, module.TASK)
    saved = json.loads(output.read_text())
    for item in saved:
        expected = result(str(item["id"]), task)[f"{task}_memories"]
        assert item[f"{task}_memories"] == expected
    assert source.read_bytes() == original

    # A rerun, here into another output directory, reads every answer from the cache.
    client.generate_batch.reset_mock()
    rerun = file_options(tmp_path, client, output_dir=tmp_path / "rerun")
    assert module.process_file(source, **rerun) == FileStats(cached=9, pending=0, windows=0, failed=0, calls=0)
    client.generate_batch.assert_not_called()
    assert json.loads(distill_profile.output_path_for(rerun["output_dir"], source, module.TASK).read_text()) == saved

    # Only a sample whose text changed is asked again, under its new text.
    items[4]["messages"][0]["content"] = "改" * 100
    source.write_text(json.dumps(items))
    received.clear()
    failed_once = True
    assert module.process_file(source, **rerun).pending == 1
    [[request]] = received
    assert request.metadata["sample_ids"] == ["4"] and "改" * 100 in request.messages[0]["content"]


def test_cache_key_leaves_out_position_and_follows_rules_and_text():
    def key(item, *, recent=False, sid="0", task="state", role="assistant"):
        return sample_cache_key(ChatSample(0, sid, item, recent), task=task, target_role=role)

    item = sample(0).item
    assert key(item) == key(item, sid="17")
    assert key(item) != key(item, recent=True)
    assert key(item) != key(item, task="event")
    assert key(item) != key(item, role="user")
    assert key(item) != key({**item, "messages": [{"role": "assistant", "content": "别的"}]})
    # Event rules see the chat time, profile rules do not.
    assert key(item) == key({**item, "time": "2025-02-01"})
    assert key(item, task="event") != key({**item, "time": "2025-02-01"}, task="event")


@pytest.mark.parametrize("task,module", [("state", distill_profile), ("event", distill_event)])
def test_failed_window_never_becomes_successful_empty_memories(tmp_path, task, module):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(chats(2)))
    client = Mock()
    client.generate_batch.side_effect = lambda requests: [
        LLMResponse(ok=True, parsed_json={"results": []}) for _ in requests
    ]
    options = file_options(tmp_path, client, model="fixture", batch_size=30)
    output = distill_profile.output_path_for(options["output_dir"], source, module.TASK)
    output.parent.mkdir(parents=True)
    output.write_text("[]")
    assert module.process_file(source, **options) == FileStats(cached=0, pending=2, windows=1, failed=2, calls=2)
    assert client.generate_batch.call_count == 2
    # No answer is cached for a failure, and no output file is left that would look complete.
    assert not output.exists()
    assert module.process_file(source, **options).pending == 2
    assert client.generate_batch.call_count == 4


def test_failed_samples_stop_the_run_and_a_rerun_asks_only_them(tmp_path, monkeypatch):
    from weclone.core.inference import llm_client

    config = tmp_path / "settings.jsonc"
    config.write_text(json.dumps({
        "codex_exec_args": {
            "model": "fixture", "effort": "low", "batch_size": 1,
            "timeout": 30, "command": "codex", "sandbox": "read-only",
        },
        "agent_distill_args": {"llm_provider": "codex_exec", "max_samples_per_window": 1},
    }))
    input_dir = tmp_path / "people"
    input_dir.mkdir()
    (input_dir / "chat.json").write_text(json.dumps(chats(2)))
    asked = []
    broken = {"1"}

    def generate(requests):
        rows = []
        for request in requests:
            ids = request.metadata["sample_ids"]
            asked.append(ids)
            rows.append(LLMResponse(ok=False, error="boom") if set(ids) & broken else response(ids, "state"))
        return rows

    client = Mock(generate_batch=Mock(side_effect=generate))
    monkeypatch.setattr(llm_client, "build_llm_client", lambda *a, **kw: client)
    run = {"input_dir": input_dir, "output_dir": tmp_path / "distill", "config_path": config, "confirmed": True}
    with pytest.raises(RuntimeError, match="1 个样本失败"):
        distill_profile.main(**run)
    assert asked == [["0"], ["1"], ["1"]]
    broken.clear()
    assert distill_profile.main(**run) == FileStats(cached=1, pending=1, windows=1, failed=0, calls=1)
    assert asked[3:] == [["1"]]


@pytest.mark.parametrize("max_tokens", [None, 4096])
def test_window_config_preserves_model_concurrency_and_optional_token_limit(tmp_path, max_tokens):
    config = {
        "codex_exec_args": {
            "model": "configured-model",
            "effort": "low",
            "command": "codex",
            "sandbox": "read-only",
            "batch_size": 30,
            "timeout": 120,
            "max_tokens": max_tokens,
        },
        "agent_distill_args": {"llm_provider": "codex_exec"},
    }
    path = tmp_path / "settings.jsonc"
    path.write_text(json.dumps(config))
    args = distill_profile.default_args()
    args.config_path = path
    distill_profile.resolve_llm_args(args)
    assert (args.max_samples_per_window, args.max_content_chars) == (8, 400)
    assert (args.model, args.batch_size, args.max_tokens) == ("configured-model", 30, max_tokens)


@pytest.mark.parametrize("module", [distill_profile, distill_event])
def test_dry_run_counts_pending_and_previews_a_whole_window_without_writes(tmp_path, capsys, module):
    source = tmp_path / "input.json"
    source.write_text(json.dumps([sample(i, 100).item for i in range(5)]))
    client = Mock()
    options = file_options(tmp_path, client, model="fixture", batch_size=30, dry_run=True)
    assert module.process_file(source, **options) == FileStats(cached=0, pending=5, windows=2)
    preview = capsys.readouterr().out
    assert "#0\n" in preview and "#3\n" in preview and "#4\n" not in preview
    client.generate_batch.assert_not_called()
    assert not options["output_dir"].exists() and not (tmp_path / "cache.sqlite").exists()
