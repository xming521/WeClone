"""End-to-end regression check for style extraction, with a fake LLM judge that always fails.

The snapshot pins every output file and the set of LLM requests. After an intended change, regenerate it with
UPDATE_STYLE_SNAPSHOT=1 and review which files moved.
"""

import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from weclone.data.agent.distill_style import data, extraction, register_style, run, variation, wire
from weclone.data.agent.distill_style.batching import JOINT_FIELDS, BatchDispatcher, restore_chat_output

ROOT = Path(__file__).resolve().parents[1]
LTP_MODEL = ROOT / "models/LTPbase"
SNAPSHOT = Path(__file__).parent / "tests_data/distill_style_snapshot.json"
# Code digests change with every edit; repair cache keys and batch IDs carry random UUIDs; the discovery
# artifact digest covers its absolute input path.
VOLATILE_KEYS = {"implementation_sha256", "batch_id", "artifact_digest"}

REPLIES = [
    "哈哈哈哈好的",
    "emmm 我想想",
    "[捂脸]\n不是吧",
    "可以可以😂",
    "嗯嗯",
    "你在干嘛？",
    "我觉得这个 update 一下就好了。",
    "今天2点开会\n记得带电脑",
    "啊啊啊啊啊 真的太好了！！",
    "看看再说吧...",
    "我们明天去吃火锅吧，那个店人不多",
    "非常开心，谢谢你们",
    "OK",
    "笑死\n哈哈哈",
    "这个价格3.5元一斤，还挺便宜的",
    "等等我马上来",
    "好吧，那就这样定了，周末见",
    "不知道诶，你问问他",
]
PROMPTS = ["在吗", "你觉得呢", "明天有空吗", "晚上吃什么", "那个文件发我一下", "周末去哪"]


def chat_file(contact, count, offset):
    items = []
    for i in range(count):
        n = offset + i
        messages = []
        for turn in range(1 + n % 3):
            messages.append({"role": "user", "content": PROMPTS[(n + turn) % len(PROMPTS)]})
            messages.append({"role": "assistant", "content": REPLIES[(3 * n + turn) % len(REPLIES)]})
        items.append(
            {
                "id": f"{contact}-{i}",
                "time": f"2025-{1 + n % 6:02d}-{1 + n % 27:02d} 1{n % 10}:00:00",
                "messages": messages,
                "chat_with_id": contact,
            }
        )
    return items


def normalized(value, tmp):
    if isinstance(value, dict):
        return {
            k: normalized(v, tmp) for k, v in value.items() if k not in VOLATILE_KEYS and "cache_key" not in k
        }
    if isinstance(value, list):
        return [normalized(v, tmp) for v in value]
    if isinstance(value, str):
        return value.replace(str(tmp), "<tmp>")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def outputs(directory, tmp):
    """Digest of every normalized JSON output; batch logs and progress counters depend on thread timing."""
    return {
        str(path.relative_to(directory)): digest(normalized(json.loads(path.read_text()), tmp))
        for path in sorted(directory.rglob("*.json"))
        if "llm" not in path.relative_to(directory).parts
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    if not LTP_MODEL.exists():
        pytest.skip("models/LTPbase is not available")
    (tmp_path / "in").mkdir()
    for index, contact in enumerate(("chat_with_a", "chat_with_b", "chat_with_c")):
        rows = chat_file(contact, 8 if index < 2 else 3, 10 * index)
        (tmp_path / "in" / f"{contact}.json").write_text(json.dumps(rows, ensure_ascii=False))
    settings = tmp_path / "settings.jsonc"
    settings.write_text(
        json.dumps(
            {
                "security_args": {"storage_mode": "plaintext", "state_dir": str(tmp_path / "state")},
                "agent_distill_args": {"llm_provider": "codex_exec"},
                "codex_exec_args": {
                    "model": "fake",
                    "effort": "low",
                    "timeout": 1,
                    "command": "codex",
                    "sandbox": "read-only",
                },
            }
        )
    )
    monkeypatch.setenv("WECLONE_CONFIG_PATH", str(settings))
    # Relative input paths keep tmp_path out of the records, so record digests are stable.
    monkeypatch.chdir(tmp_path)
    requests, lock = [], threading.Lock()

    def generate(self, prompt, payload, cache_key=None):
        with lock:
            requests.append(digest([prompt, payload]))
        return {
            "state": "failure",
            "error": "FakeJudge",
            "raw": None,
            "model": self.model,
            "provider": self.provider,
            "cache_key": cache_key,
        }

    monkeypatch.setattr(extraction.Judge, "_generate", generate)

    def extract(name, *extra):
        requests.clear()
        output = tmp_path / name
        run.main(
            [
                *extra,
                "--input-dir", "in",
                "--output", str(output),
                "--cache", "cache",
                "--settings", str(settings),
                "--ltp-model", str(LTP_MODEL),
                "--workers", "4",
                "--min-group-messages", "8",
            ]
        )  # fmt: skip
        return {
            "files": outputs(output, tmp_path),
            "requests": {"count": len(requests), "digest": digest(sorted(requests))},
        }

    return extract


def frozen_expressions(path):
    value = {
        "version": 1,
        "state": "complete",
        "person_ids": ["p0"],
        "candidates": [
            {"item": "004,006", "expression": expression, "kind": "expression", "accepted": True}
            for expression in ("哈哈哈", "可以可以")
        ],
    }
    value["artifact_digest"] = data.digest(value)
    path.write_text(json.dumps(value, ensure_ascii=False))
    return str(path)


def test_extraction_matches_snapshot(workspace, tmp_path):
    frozen = frozen_expressions(tmp_path / "frozen.json")
    result = {
        # Sampled semantic pass with group top-ups, a surface pass over all windows and QA repeats.
        "extract": workspace(
            "extract",
            "--llm-samples",
            "12",
            "--group-floor",
            "2",
            "--qa-samples",
            "1",
            "--llm-batch-size",
            "1",
        ),
        "no_llm": workspace("no_llm", "--no-llm"),
        "one_chapter": workspace("one_chapter", "--no-llm", "--chapter", "syntactic_style", "--no-groups"),
        "discover": workspace("discover", "discover"),
        "frozen": workspace("frozen", "--no-llm", "--chapter", "lexical_choice", "--frozen", frozen),
    }
    # Batched requests depend on timing, so only the outputs are pinned.
    result["batched"] = {
        "files": workspace("batched", "--llm-samples", "0", "--chapter", "pragmatic_style")["files"]
    }
    if os.environ.get("UPDATE_STYLE_SNAPSHOT"):
        SNAPSHOT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    expected = json.loads(SNAPSHOT.read_text())
    for name, value in expected.items():
        files = result[name]["files"]
        changed = sorted(k for k in set(files) | set(value["files"]) if files.get(k) != value["files"].get(k))
        assert not changed, f"{name}: changed outputs {changed}"
        assert result[name].get("requests") == value.get("requests"), f"{name}: LLM requests changed"


def test_restore_chat_output_follows_module_after_protocol_bump(monkeypatch):
    protocol = "paralinguistic-style-v99"
    monkeypatch.setitem(wire.MODULES, protocol, "paralinguistic_style")
    chat = {
        "context": [],
        "messages": [["m0", "assistant", "哈哈哈哈"]],
        "anchors": {"repeats": [{"anchor_id": "r0"}], "emoji": []},
    }
    restored = restore_chat_output({"repeat_usage": [{"use": "emphasis"}]}, chat, protocol)
    assert restored["repeat_usage"] == [{"use": "emphasis", "anchor_id": "r0"}]


def test_expand_output_accepts_states_split_per_item():
    schema = {
        "module": "pragmatic_style",
        "keys": ("x", "y"),
        "extra": (),
        "root": False,
        "per_message": True,
        "message_ids": ["m0", "m1"],
        "degree_anchors": [],
        "messages": [],
    }
    split = wire.expand_output({"states": ["aa", "an"]}, schema)
    assert split == wire.expand_output({"states": "aaan"}, schema)
    with pytest.raises(ValueError):
        wire.expand_output({"states": ["aaa", "n"]}, schema)


def test_joint_batch_requests_and_keeps_only_its_categories(tmp_path):
    protocol = next(p for p, field in JOINT_FIELDS.items() if field == "词汇")
    prompts = []

    def generate(prompt, payload):
        prompts.append(prompt)
        result = {"语体": {}, "词汇": {"states": "a"}, "修辞": {}}
        return {"state": "ok", "data": {"results": [result]}, "raw": "", "model": "fake"}

    dispatcher = BatchDispatcher(generate, tmp_path, 1, 8, 16000, 32000)
    payload = {"context": [], "messages": [["m0", "assistant", "好的"]]}
    future = dispatcher.submit("rules", payload, "k0", "c0", protocol)
    dispatcher.close()
    assert future.result()["data"] == {"states": "a"}
    assert '"词汇":该类结果' in prompts[0] and "语体" not in prompts[0]


def test_repair_state_strings_drops_replace_suffix():
    raw = '{"a":{"states":"pa ap".replace(" ",""),"items":{}},"b":{"states":"p a".replace}}'
    assert json.loads(extraction.repair_state_strings(raw)) == {
        "a": {"states": "paap", "items": {}},
        "b": {"states": "pa"},
    }


def test_joint_batch_keeps_one_rule_per_category(tmp_path):
    protocol = next(p for p, field in JOINT_FIELDS.items() if field == "词汇")
    requests = []

    def generate(prompt, payload):
        requests.append(prompt)
        result = [{"词汇": {"states": "a"}}]
        return {"state": "ok", "data": {"results": result}, "raw": "", "model": "fake"}

    dispatcher = BatchDispatcher(generate, tmp_path, 1, 8, 16000, 32000)
    futures = [
        dispatcher.submit(rule, {"context": [], "messages": [[f"m{i}", "assistant", "好的"]]}, f"k{i}", f"c{i}", protocol)
        for i, rule in enumerate(("eight keys", "nine keys"))
    ]
    dispatcher.close()
    assert len(requests) == 2 and all(f.result()["state"] == "ok" for f in futures)


def test_register_labels_are_stratified_by_window_length():
    def window(sample_id, *texts):
        messages = [{"message_id": f"m{i}", "text": t} for i, t in enumerate(texts)]
        return {"sample_id": sample_id, "context": [], "messages": messages}

    records = [window("w0", "好的"), window("w1", "这个" * 10, "那个" * 10, "")]
    items = [{"state": "present", "label": "spoken"}, {"state": "present", "label": "mixed"}]
    summary = register_style.label_summary(records, items, "233")
    # Each nonempty message lands in its whole window's bucket: 2 characters vs 40 characters.
    assert summary["by_length"]["message_weighted_proportions"] == {
        "spoken": {"1-8": {"hits": 1, "messages": 1, "rate": 1.0}, "33-64": {"hits": 0, "messages": 2, "rate": 0.0}},
        "mixed": {"1-8": {"hits": 0, "messages": 1, "rate": 0.0}, "33-64": {"hits": 2, "messages": 2, "rate": 1.0}},
    }
    leaves = variation.collect({"233": {"measurements": summary}})["233"]
    assert leaves["message_weighted_proportions/mixed"] == (2, 3, {"1-8": (0, 1), "33-64": (2, 2)})


def test_repair_calls_are_cached_apart_from_the_first_call(tmp_path, monkeypatch):
    settings = tmp_path / "settings.jsonc"
    backend = {"model": "fake", "effort": "low", "timeout": 1, "command": "codex", "sandbox": "read-only"}
    security = {"storage_mode": "plaintext", "state_dir": str(tmp_path / "state")}
    settings.write_text(json.dumps({"security_args": security, "codex_exec_args": backend}))
    monkeypatch.setenv("WECLONE_CONFIG_PATH", str(settings))
    calls = []

    def generate(self, prompt, payload, cache_key=None):
        calls.append(cache_key)
        return {"state": "ok", "data": {"n": len(calls)}, "raw": "", "model": "fake", "cache_key": cache_key}

    monkeypatch.setattr(extraction.Judge, "_generate", generate)
    judge = extraction.Judge(settings, tmp_path / "cache", protocol="test-protocol")
    first = judge.call("p", {"x": 1})
    repair = judge.call("p", {"x": 1}, attempt=1)
    assert repair["cache_key"] != first["cache_key"] and len(calls) == 2
    # A rerun reads the repair from the cache; only QA repeats draw a new sample.
    assert judge.call("p", {"x": 1}, attempt=1)["data"] == repair["data"] and len(calls) == 2
    judge.call("p", {"x": 1}, refresh=True, attempt=1)
    assert len(calls) == 3


def test_codex_pacing_comes_from_the_given_settings(tmp_path):
    settings = tmp_path / "elsewhere.jsonc"
    backend = {"model": "fake", "effort": "low", "timeout": 1, "command": "codex", "sandbox": "read-only"}
    backend |= {"request_interval_seconds": 7, "capacity_cooldown_seconds": 3}
    settings.write_text(json.dumps({"codex_exec_args": backend}))
    client = extraction.Judge(settings, tmp_path / "cache", protocol="test-protocol").client
    assert (client.request_interval_seconds, client.capacity_cooldown_seconds) == (7, 3)
