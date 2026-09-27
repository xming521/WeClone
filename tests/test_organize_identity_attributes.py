import json
from types import SimpleNamespace

import pytest

from weclone.core.inference.llm_client import LLMResponse
from weclone.data.agent import organize_identity_attributes as identity
from weclone.data.agent import organize_memories as organize
from weclone.prompts.memory_organization import ATTRIBUTE_HIERARCHY_SCHEMA


def test_identity_pipeline_only_sends_names_and_preserves_every_fact(tmp_path, monkeypatch):
    source = tmp_path / "organized.json"
    output = tmp_path / "output"
    facts = [
        {"dim": 1, "attr": attr, "value": f"PRIVATE_FACT_{i}", "source_ids": [f"s{i}"], "support_count": i + 1}
        for i, attr in enumerate(["职业", "职业岗位", "职业", "本科专业", "研究方向"])
    ]
    sources = {f"s{i}": {"content": f"PRIVATE_SOURCE_{i}"} for i in range(len(facts))}
    data = {"facts": facts + [{"dim": 8, "attr": "偏好", "value": "PRIVATE_PREFERENCE"}], "sources": sources}
    organize.save(source, data)
    before = source.read_bytes()
    calls = []

    class Runner:
        def __init__(self, args):
            pass

        def run(self, tasks, validate):
            calls.extend(tasks)
            prompt = tasks[0]["prompt"]
            assert "PRIVATE_" not in prompt
            rows = json.loads(prompt.split("\n属性：\n")[1])
            assert len(rows) == 4
            payload = {"items": [
                {"id": row["id"], "topic": "工作" if row["attr"].startswith("职业") else "教育",
                 "attr": "职业" if row["attr"].startswith("职业") else row["attr"]}
                for row in reversed(rows)
            ]}
            return [validate(payload, tasks[0])]

        def close(self):
            pass

    monkeypatch.setattr(identity, "LLMTasks", Runner)
    identity.main(["--input-path", str(source), "--output-dir", str(output)])
    result = organize.load(output / "identity_profile.json")
    groups = [g for topic in result["topics"] for g in topic["attributes"]]
    restored = [f for group in groups for f in group["facts"]]
    assert sorted(restored, key=lambda f: f["value"]) == facts
    assert len(next(g for g in groups if g["attr"] == "职业")["facts"]) == 3
    assert {g["attr"] for g in groups} == {"职业", "本科专业", "研究方向"}
    assert result["sources"] == sources
    assert source.read_bytes() == before
    assert len(calls) == 1


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unknown", "boolean", "blank", "split_topic"])
def test_mapping_rejects_incomplete_or_ambiguous_links(defect):
    attributes = [{"id": 1, "attr": "职业"}, {"id": 2, "attr": "职业岗位"}]
    items = [{"id": 1, "topic": "工作", "attr": "职业"}, {"id": 2, "topic": "工作", "attr": "职业"}]
    if defect == "missing":
        items.pop()
    elif defect == "duplicate":
        items[1]["id"] = 1
    elif defect == "unknown":
        items[1]["id"] = 3
    elif defect == "boolean":
        items[0]["id"] = True
    elif defect == "blank":
        items[0]["topic"] = " "
    else:
        items[1]["topic"] = "教育"
    with pytest.raises(ValueError):
        identity.validate_mapping({"items": items}, attributes)


def test_hierarchy_request_uses_its_schema_and_existing_checkpoint_runner(tmp_path):
    runner = organize.LLMTasks.__new__(organize.LLMTasks)
    runner.path, runner.entries = tmp_path / "checkpoint.json", {}
    runner.budget = organize.InputBudget(8192, "o200k_base")
    runner.options = SimpleNamespace(
        batch_size=50, llm_provider="api", model=None, effort=None, max_tokens=200, timeout=10, overwrite=False,
    )
    payload = {"items": [{"id": 1, "topic": "工作", "attr": "职业"}]}

    class Client:
        def generate_batch(self, requests):
            assert len(requests) == 1
            assert requests[0].json_schema == ATTRIBUTE_HIERARCHY_SCHEMA
            return [LLMResponse(ok=True, parsed_json=payload, finish_reason="stop")]

    runner.client = Client()
    result = runner.run([{"stage": "attribute_hierarchy", "prompt": "属性名"}],
                        lambda value, _: identity.validate_mapping(value, [{"id": 1, "attr": "职业"}]))
    assert result == [payload["items"]]
