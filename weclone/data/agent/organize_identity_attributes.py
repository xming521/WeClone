"""Group identity attributes into topics without sending fact values to the model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from weclone.data.agent.organize_memories import LLMTasks, chinese_attr, load, save
from weclone.prompts.memory_organization import attribute_hierarchy_prompt


def validate_mapping(payload: object, attributes: list[dict]) -> list[dict]:
    if not isinstance(payload, dict) or set(payload) != {"items"} or not isinstance(payload["items"], list):
        raise ValueError("Attribute mapping requires items array")
    expected = {row["id"] for row in attributes}
    seen, topics = set(), {}
    result = []
    for item in payload["items"]:
        if not isinstance(item, dict) or set(item) != {"id", "topic", "attr"}:
            raise ValueError("Attribute mapping requires id, topic and attr")
        rid = item["id"]
        if type(rid) is not int or rid not in expected or rid in seen:
            raise ValueError("Unknown or duplicate attribute id")
        if not chinese_attr(item["topic"]) or not chinese_attr(item["attr"]):
            raise ValueError("Topic and attribute must be nonempty Chinese names")
        topic, attr = item["topic"].strip(), item["attr"].strip()
        if attr in topics and topics[attr] != topic:
            raise ValueError("The same canonical attribute has different topics")
        topics[attr] = topic
        seen.add(rid)
        result.append({"id": rid, "topic": topic, "attr": attr})
    if seen != expected:
        raise ValueError("Missing attribute ids")
    return sorted(result, key=lambda row: row["id"])


def build_hierarchy(facts: list[dict], attributes: list[dict], mapping: list[dict]) -> list[dict]:
    by_id = {row["id"]: row for row in mapping}
    by_name = {row["attr"]: by_id[row["id"]] for row in attributes}
    topics = {}
    for fact in facts:
        item = by_name[fact["attr"]]
        groups = topics.setdefault(item["topic"], {})
        group = groups.setdefault(item["attr"], {"attr": item["attr"], "original_attrs": [], "facts": []})
        if fact["attr"] not in group["original_attrs"]:
            group["original_attrs"].append(fact["attr"])
        group["facts"].append(fact)
    return [{"topic": topic, "attributes": list(groups.values())} for topic, groups in topics.items()]


def run(args: argparse.Namespace) -> dict:
    data = load(args.input_path)
    facts = [fact for fact in data["facts"] if fact["dim"] == 1]
    if not facts:
        raise ValueError("Input has no identity facts (dimension 1)")
    attributes = [{"id": i, "attr": name} for i, name in enumerate(sorted({f["attr"] for f in facts}), 1)]
    counts = {"facts": len(facts), "attributes": len(attributes)}
    if args.dry_run:
        return counts
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task = {"stage": "attribute_hierarchy", "prompt": attribute_hierarchy_prompt(attributes)}
    runner = LLMTasks(args)
    try:
        mapping = runner.run([task], lambda payload, _: validate_mapping(payload, attributes))[0]
    finally:
        runner.close()
    topics = build_hierarchy(facts, attributes, mapping)
    source_ids = {sid for fact in facts for sid in fact["source_ids"]}
    result = {
        "input_file": str(args.input_path.resolve()),
        "topics": topics,
        "sources": {sid: data["sources"][sid] for sid in sorted(source_ids)},
    }
    save(args.output_dir / "attribute_mapping.json", [
        {**item, "original_attr": attributes[item["id"] - 1]["attr"]} for item in mapping
    ])
    save(args.output_dir / "identity_profile.json", result)
    return {**counts, "topics": len(topics),
            "canonical_attributes": sum(len(topic["attributes"]) for topic in topics),
            "output_dir": str(args.output_dir)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path,
                        default=Path("dataset/res_csv/agent/memory_organization/organized_memories.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config-path", type=Path, default=Path("settings.jsonc"))
    parser.add_argument("--llm-provider", choices=("api", "codex_exec"))
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--token-encoding", default="o200k_base")
    parser.add_argument("--tokenizer-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_context_tokens is not None and args.max_context_tokens < 1:
        parser.error("--max-context-tokens must be positive")
    if args.output_dir is None:
        args.output_dir = Path("dataset/res_csv/agent") / (
            "identity_attributes_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
