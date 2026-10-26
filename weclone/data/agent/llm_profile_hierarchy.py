"""Build the browsable profile hierarchy (domain → topic → profile facts) from organized memories.

Topics with more than --leaf-facts facts are split into smaller topics, and each domain description is rewritten
from its final topics. The output also keeps a per-dimension view (dimension → domain → topic → attribute) for the
review UI.
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from weclone.data.agent.organize_memories import LLMTasks, save, validate_topics
from weclone.prompts.memory_organization import PROFILE_DIMS, dimension_name
from weclone.prompts.profile_hierarchy import DESCRIBE_SCHEMA, SPLIT_SCHEMA, describe_prompt, split_prompt
from weclone.utils import secure_storage

# Chat peer ids as organize_memories assigns them: P1, P2, ...
PEER_ID = re.compile(r"(?<![A-Za-z0-9])P\d+(?![A-Za-z0-9])")


def topic_layout(data: dict) -> tuple[list[dict], list[dict]]:
    """Profile facts and the domains holding them, each with its topics' fact indices in organizer order."""
    facts = [fact for fact in data["facts"] if fact["dim"] in PROFILE_DIMS]
    members = {domain["name"]: {} for domain in data["domains"]}
    for topic in data["topics"]:
        members[topic["domain"]][topic["name"]] = []
    for index, fact in enumerate(facts):
        indices = members.get(fact.get("domain"), {}).get(fact.get("topic"))
        if indices is None:
            raise ValueError(f"Profile fact {index} is not in a known domain topic")
        indices.append(index)
    domains = []
    for domain in data["domains"]:
        topics = [
            {"name": name, "fact_indices": rows} for name, rows in members[domain["name"]].items() if rows
        ]
        if topics:
            domains.append({**domain, "topics": topics})
    return facts, domains


def validate_split(payload, count: int, taken: set[str]) -> list[dict]:
    topics = validate_topics(payload, count)
    if len(topics) < 2 or any(topic["name"] in taken for topic in topics):
        raise ValueError("A split needs at least two topics with new names")
    # The router reads topic names against the incoming message, where a chat peer id never appears.
    if any(PEER_ID.search(topic["name"]) for topic in topics):
        raise ValueError("Split topic names must not contain chat peer ids")
    return topics


def unique_name(name: str, prefix: str, used: set[str]) -> str:
    candidate, n = (name if name not in used else f"{prefix}·{name}"), 2
    while candidate in used:
        candidate, n = f"{prefix}·{name}{n}", n + 1
    return candidate


def split_topics(
    domains: list[dict], facts: list[dict], runner, leaf_facts: int
) -> tuple[list[dict], list[dict]]:
    """Replace each topic holding more than leaf_facts facts by the smaller topics the model splits it into."""
    tasks = []
    for d, domain in enumerate(domains):
        names = [topic["name"] for topic in domain["topics"]]
        for t, topic in enumerate(domain["topics"]):
            if len(topic["fact_indices"]) > leaf_facts:
                tasks.append(
                    {
                        "stage": "split_topic",
                        "schema": SPLIT_SCHEMA,
                        "position": [d, t],
                        "count": len(topic["fact_indices"]),
                        "taken": names,
                        "prompt": split_prompt(
                            domain,
                            [name for name in names if name != topic["name"]],
                            topic["name"],
                            [facts[i] for i in topic["fact_indices"]],
                            leaf_facts,
                        ),
                    }
                )
    results = (
        runner.run(tasks, lambda payload, task: validate_split(payload, task["count"], set(task["taken"])))
        if tasks
        else []
    )
    splits = {tuple(task["position"]): result for task, result in zip(tasks, results)}
    refined, report = [], []
    for d, domain in enumerate(domains):
        used = {topic["name"] for t, topic in enumerate(domain["topics"]) if (d, t) not in splits}
        topics = []
        for t, topic in enumerate(domain["topics"]):
            if (d, t) not in splits:
                topics.append(topic)
                continue
            parts = []
            for part in splits[(d, t)]:
                name = unique_name(part["name"], topic["name"], used)
                used.add(name)
                parts.append(
                    {
                        "name": name,
                        "fact_indices": sorted(topic["fact_indices"][i - 1] for i in part["items"]),
                    }
                )
            topics.extend(parts)
            report.append(
                {
                    "domain": domain["name"],
                    "topic": topic["name"],
                    "facts": len(topic["fact_indices"]),
                    "into": {part["name"]: len(part["fact_indices"]) for part in parts},
                }
            )
        refined.append({**domain, "topics": topics})
    return refined, report


def validate_descriptions(payload, count: int) -> list[str]:
    if (
        not isinstance(payload, dict)
        or set(payload) != {"domains"}
        or not isinstance(payload["domains"], list)
    ):
        raise ValueError("Descriptions require domains array")
    descriptions = {}
    for item in payload["domains"]:
        if not isinstance(item, dict) or set(item) != {"id", "description"}:
            raise ValueError("Description requires id and description")
        if type(item["id"]) is not int or not 1 <= item["id"] <= count or item["id"] in descriptions:
            raise ValueError("Unknown or duplicate domain id")
        if not isinstance(item["description"], str) or not item["description"].strip():
            raise ValueError("Domain description is empty")
        descriptions[item["id"]] = item["description"].strip()
    if len(descriptions) != count:
        raise ValueError("Missing domain ids")
    return [descriptions[i] for i in range(1, count + 1)]


def dimension_view(facts: list[dict], domains: list[dict]) -> list[dict]:
    """Per-dimension tree for the review UI: dimension → domain → topic → attribute → fact indices."""
    dimensions = []
    for dim in PROFILE_DIMS:
        groups = []
        for domain in domains:
            topics = []
            for topic in domain["topics"]:
                attrs = defaultdict(list)
                for index in topic["fact_indices"]:
                    if facts[index]["dim"] == dim:
                        attrs[facts[index]["attr"]].append(index)
                if attrs:
                    items = [{"attr": attr, "fact_indices": attrs[attr]} for attr in sorted(attrs)]
                    topics.append({"name": topic["name"], "items": items})
            if topics:
                groups.append({"name": domain["name"], "items": topics})
        if groups:
            dimensions.append({"dim": dim, "name": dimension_name(dim), "groups": groups})
    return dimensions


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for profile_hierarchy.json; the review server reads the newest "
        "dataset/res_csv/agent/profile_runs/<run>/profile_hierarchy",
    )
    parser.add_argument("--config-path", type=Path, default=Path("settings.jsonc"))
    parser.add_argument("--llm-provider", choices=("api", "codex_exec"))
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--token-encoding", default="o200k_base")
    parser.add_argument("--tokenizer-file", type=Path)
    parser.add_argument(
        "--leaf-facts", type=int, default=30, help="Topics holding more facts are split into smaller topics"
    )
    args = parser.parse_args(argv)
    if args.leaf_facts < 1:
        parser.error("--leaf-facts must be positive")
    secure_storage.configure(args.config_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_bytes = secure_storage.read_bytes(args.input_path)
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    data = json.loads(input_bytes)
    del input_bytes
    facts, domains = topic_layout(data)
    runner = LLMTasks(args)
    try:
        domains, splits = split_topics(domains, facts, runner, args.leaf_facts)
        if domains:
            task = {
                "stage": "describe_domains",
                "schema": DESCRIBE_SCHEMA,
                "prompt": describe_prompt(domains),
            }
            descriptions = runner.run(
                [task], lambda payload, _: validate_descriptions(payload, len(domains))
            )[0]
            domains = [{**domain, "description": text} for domain, text in zip(domains, descriptions)]
    finally:
        runner.close()
    facts = list(facts)
    for domain in domains:
        for topic in domain["topics"]:
            for index in topic["fact_indices"]:
                facts[index] = {**facts[index], "topic": topic["name"]}
    used = {sid for fact in facts for sid in fact["source_ids"]}
    save(
        args.output_dir / "profile_hierarchy.json",
        {
            "input_path": str(args.input_path.resolve()),
            "input_sha256": input_sha256,
            "domains": domains,
            "dimensions": dimension_view(facts, domains),
            "facts": facts,
            "sources": {sid: data["sources"][sid] for sid in sorted(used)},
        },
    )
    summary = {
        "facts": len(facts),
        "domains": [
            {
                "name": domain["name"],
                "topics": len(domain["topics"]),
                "facts": sum(len(topic["fact_indices"]) for topic in domain["topics"]),
                "max_topic_facts": max(len(topic["fact_indices"]) for topic in domain["topics"]),
            }
            for domain in domains
        ],
        "splits": splits,
    }
    save(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
