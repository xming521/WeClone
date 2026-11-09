"""Build the browsable profile hierarchy (domain → topic → profile facts) from organized memories.

Each domain's profile facts are grouped into topics of at most --leaf-facts facts, and the domain description
is rewritten from them in the same request. Topics that still hold more facts are split into smaller topics.
The output also keeps a per-dimension view (dimension → domain → topic → attribute) for the review UI.
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from weclone.data.agent.organize_memories import LLMTasks, chinese_attr, save
from weclone.prompts.memory_organization import PROFILE_DIMS, dimension_name
from weclone.prompts.profile_hierarchy import SPLIT_SCHEMA, TOPIC_SCHEMA, split_prompt, topic_prompt
from weclone.utils import secure_storage
from weclone.utils.log import logger

# Chat peer ids as organize_memories assigns them: P1, P2, ...
PEER_ID = re.compile(r"(?<![A-Za-z0-9])P\d+(?![A-Za-z0-9])")


def domain_facts(data: dict) -> tuple[list[dict], list[tuple[dict, list[int]]]]:
    """Profile facts, and each domain holding some with their fact indices in organizer order."""
    facts = [fact for fact in data["facts"] if fact["dim"] in PROFILE_DIMS]
    members: dict[str, list[int]] = {domain["name"]: [] for domain in data["domains"]}
    for index, fact in enumerate(facts):
        if fact.get("domain") not in members:
            raise ValueError(f"Profile fact {index} is not in a known domain")
        members[fact["domain"]].append(index)
    return facts, [(domain, members[domain["name"]]) for domain in data["domains"] if members[domain["name"]]]


def validate_topics(payload: Any, count: int) -> list[dict]:
    if not isinstance(payload, dict) or set(payload) != {"topics"} or not isinstance(payload["topics"], list):
        raise ValueError("Topics require topics array")
    topics, names, seen = [], set(), []
    for topic in payload["topics"]:
        if not isinstance(topic, dict) or set(topic) != {"name", "items"}:
            raise ValueError("Topic requires name and items")
        name = topic["name"].strip() if isinstance(topic["name"], str) else ""
        if not chinese_attr(name) or name in names:
            raise ValueError("Topic names must be unique Chinese names")
        # The router reads topic names against the incoming message, where a chat peer id never appears.
        if PEER_ID.search(name):
            raise ValueError("Topic names must not contain chat peer ids")
        items = topic["items"]
        if not isinstance(items, list) or not items or any(type(i) is not int for i in items):
            raise ValueError("Topic items must be fact ids")
        names.add(name)
        seen.extend(items)
        topics.append({"name": name, "items": items})
    if sorted(seen) != list(range(1, count + 1)):
        raise ValueError("Every fact id must appear in exactly one topic")
    return topics


def validate_domain_topics(payload: Any, count: int) -> dict:
    if not isinstance(payload, dict) or set(payload) != {"description", "topics"}:
        raise ValueError("Domain topics require description and topics")
    description = payload["description"].strip() if isinstance(payload["description"], str) else ""
    if not description:
        raise ValueError("Domain description is empty")
    return {"description": description, "topics": validate_topics({"topics": payload["topics"]}, count)}


def group_topics(
    domains: list[dict],
    layout: list[tuple[dict, list[int]]],
    facts: list[dict],
    runner,
    min_facts: int,
    max_facts: int,
) -> list[dict]:
    """Each domain of the layout with its facts grouped into topics and its description rewritten."""
    tasks = [
        {
            "stage": "domain_topics",
            "schema": TOPIC_SCHEMA,
            "count": len(indices),
            "prompt": topic_prompt(domains, domain, [facts[i] for i in indices], min_facts, max_facts),
        }
        for domain, indices in layout
    ]
    results = (
        runner.run(tasks, lambda payload, task: validate_domain_topics(payload, task["count"]))
        if tasks
        else []
    )
    return [
        {
            **domain,
            "description": result["description"],
            "topics": [
                {"name": topic["name"], "fact_indices": sorted(indices[i - 1] for i in topic["items"])}
                for topic in result["topics"]
            ],
        }
        for (domain, indices), result in zip(layout, results)
    ]


def validate_split(payload, count: int, taken: set[str]) -> list[dict]:
    topics = validate_topics(payload, count)
    if len(topics) < 2 or any(topic["name"] in taken for topic in topics):
        raise ValueError("A split needs at least two topics with new names")
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
        help="Directory for profile_hierarchy.json; the review server reads <run>/profile_hierarchy of the run "
        "given by server --run, by default the newest one in dataset/res_csv/agent/profile_runs",
    )
    parser.add_argument("--config-path", type=Path, default=Path("settings.jsonc"))
    parser.add_argument("--llm-provider", choices=("api", "codex_exec"))
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--token-encoding", default="o200k_base")
    parser.add_argument("--tokenizer-file", type=Path)
    parser.add_argument(
        "--leaf-facts",
        type=int,
        default=30,
        help="Facts a topic should stay within; topics still holding more are split into smaller topics",
    )
    # 8 facts cite about 20 memories (2.4 per profile fact on average); smaller topics are fragments that only
    # lengthen the topic list.
    parser.add_argument(
        "--topic-min-facts",
        type=int,
        default=8,
        help="Facts a topic should reach; sparse facts join the closest topic",
    )
    args = parser.parse_args(argv)
    if args.leaf_facts < 1 or args.topic_min_facts < 1:
        parser.error("--leaf-facts and --topic-min-facts must be positive")
    if args.topic_min_facts > args.leaf_facts:
        parser.error("--topic-min-facts must not exceed --leaf-facts")
    secure_storage.configure(args.config_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_bytes = secure_storage.read_bytes(args.input_path)
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    data = json.loads(input_bytes)
    del input_bytes
    facts, layout = domain_facts(data)
    runner = LLMTasks(args)
    try:
        domains = group_topics(data["domains"], layout, facts, runner, args.topic_min_facts, args.leaf_facts)
        domains, splits = split_topics(domains, facts, runner, args.leaf_facts)
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
    logger.info(
        "Profile hierarchy: {counts[facts]} facts, {counts[domains]} domains, {counts[topics]} topics",
        counts={
            "facts": len(facts),
            "domains": len(domains),
            "topics": sum(len(domain["topics"]) for domain in domains),
        },
    )


if __name__ == "__main__":
    main()
