"""Request one attribute-only hierarchy per dimension using the configured LLM client."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from weclone.core.inference.llm_client import LLMRequest
from weclone.data.agent.organize_memories import LLMTasks, save
from weclone.prompts.memory_organization import DIMENSIONS
from weclone.prompts.profile_hierarchy import SCHEMA, prompt
from weclone.utils import secure_storage


def validate(payload, rows):
    seen = []

    def node(value, level):
        if not isinstance(value, dict) or set(value) != {"name", "items"}:
            raise ValueError("A topic needs name and items")
        if not isinstance(value["name"], str) or not value["name"].strip():
            raise ValueError("Empty topic name")
        if not isinstance(value["items"], list) or not value["items"]:
            raise ValueError("Empty topic")
        child_names = []
        for item in value["items"]:
            if type(item) is int:
                seen.append(item)
            elif level == 2:
                node(item, 3)
                child_names.append(item["name"])
            else:
                raise ValueError("Only attribute ids may appear under level 3")
        if len(child_names) != len(set(child_names)):
            raise ValueError("Duplicate sibling topic names")

    if not isinstance(payload, dict) or set(payload) != {"groups"} or not isinstance(payload["groups"], list):
        raise ValueError("Expected groups array")
    for group in payload["groups"]:
        node(group, 2)
    names = [group["name"] for group in payload["groups"]]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate level 2 topics")
    if sorted(seen) != sorted(row["id"] for row in rows):
        raise ValueError("Missing, duplicate or invented attribute ids")
    return payload


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config-path", type=Path, default=Path("settings.jsonc"))
    parser.add_argument("--llm-provider", choices=("api", "codex_exec"))
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--token-encoding", default="o200k_base")
    parser.add_argument("--tokenizer-file", type=Path)
    args = parser.parse_args(argv)
    secure_storage.configure(args.config_path)
    if args.output_dir is None:
        args.output_dir = Path("dataset/res_csv/agent/memory_organization")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = secure_storage.read_json(args.input_path)
    dims = (1, 3, 4, 5, 8)
    facts = [f for f in data["facts"] if f["dim"] in dims]
    attributes = {
        dim: [
            {"id": i, "attr": name}
            for i, name in enumerate(sorted({f["attr"] for f in facts if f["dim"] == dim}), 1)
        ]
        for dim in dims
    }
    tasks = [
        {"dim": dim, "attributes": rows, "prompt": prompt(dim, rows)}
        for dim, rows in attributes.items()
        if rows
    ]
    save(args.output_dir / "requests.json", {"schema": SCHEMA, "tasks": tasks})
    runner = LLMTasks(args)
    entries = []
    started = perf_counter()
    try:
        requests = [
            LLMRequest.from_prompt(
                task["prompt"],
                provider=runner.options.llm_provider,
                model=runner.options.model,
                effort=runner.options.effort,
                max_tokens=runner.options.max_tokens,
                timeout=runner.options.timeout,
                json_mode=True,
                json_schema=SCHEMA,
                metadata={"stage": "profile_hierarchy", "dim": task["dim"]},
            )
            for task in tasks
        ]
        input_stats = [runner.budget.check(task["prompt"]) for task in tasks]
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "requests": len(tasks),
                    "provider": runner.options.llm_provider,
                    "model": runner.options.model,
                    "prompt_tokens": [x["tokens"] for x in input_stats],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        responses = list(runner.client.generate_batch(requests))
        for i, task in enumerate(tasks):
            entry = {"dim": task["dim"], "input": input_stats[i]}
            if i >= len(responses):
                entry.update(status="failed", error="Missing response")
            else:
                response = responses[i]
                entry.update(
                    model=response.model,
                    usage=response.metadata.get("usage"),
                    elapsed_s=response.elapsed_s,
                    cost_usd=response.cost_usd,
                )
                try:
                    if not response.ok or response.finish_reason in {"length", "max_tokens"}:
                        raise ValueError(response.error or "Failed or truncated response")
                    entry.update(status="done", result=validate(response.parsed_json, task["attributes"]))
                except ValueError as exc:
                    entry.update(status="failed", error=str(exc), result=response.parsed_json)
            entries.append(entry)
        save(args.output_dir / "llm_checkpoint.json", entries)
    finally:
        runner.close()
    summary = {
        "wall_s": perf_counter() - started,
        "requests": len(tasks),
        "provider": runner.options.llm_provider,
        "model": runner.options.model,
        "dimensions": [],
    }
    if any(row["status"] != "done" for row in entries):
        raise RuntimeError(f"Some requests failed; results saved at {args.output_dir}; no retries sent")
    dimensions = []
    for entry in entries:
        dim = entry["dim"]
        rows = attributes[dim]
        by_id = {row["id"]: row["attr"] for row in rows}

        def expand(group):
            return {
                "name": group["name"],
                "items": [
                    {
                        "attr": by_id[item],
                        "fact_indices": [
                            i for i, f in enumerate(facts) if f["dim"] == dim and f["attr"] == by_id[item]
                        ],
                    }
                    if type(item) is int
                    else expand(item)
                    for item in group["items"]
                ],
            }

        groups = entry["result"]["groups"]
        dimensions.append(
            {"dim": dim, "name": DIMENSIONS[dim].split("：")[0], "groups": [expand(g) for g in groups]}
        )
        summary["dimensions"].append(
            {
                "dim": dim,
                "attributes": len(rows),
                "level2": len(groups),
                "level3": sum(isinstance(item, dict) for g in groups for item in g["items"]),
                "usage": entry["usage"],
                "elapsed_s": entry["elapsed_s"],
            }
        )
    used = {sid for f in facts for sid in f["source_ids"]}
    save(
        args.output_dir / "profile_hierarchy.json",
        {
            "input_path": str(args.input_path.resolve()),
            "input_sha256": hashlib.sha256(
                secure_storage.resolve_path(args.input_path).read_bytes()
            ).hexdigest(),
            "dimensions": dimensions,
            "facts": facts,
            "sources": {sid: data["sources"][sid] for sid in sorted(used)},
        },
    )
    save(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
