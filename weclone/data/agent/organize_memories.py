"""Organize extracted profile/event memories by life domain, topic and dimension without reading chat messages.

Stages: prepare reads the extracted memories and notes which chat each peer id stands for (the stable contact id,
P1, P2, ..., from the people files). domains induces the person's life domains from extraction tags
and maps the tags to them. peers writes one line per chat peer on how they relate to B, from the memories that
mention them. classify assigns each memory a dimension, a domain and an attribute name, in batches grouped by the
tags' domains, places what passes between B and a peer by that relation, and gives every memory a privacy level.
topics groups each domain's profile attributes into conversation topics. summarize writes facts per (topic,
dimension), each with its own privacy level and time scope (stable, current, superseded or dated); dimension 6 is
summarized per chat peer and dimension 7 per domain. Identity facts of every domain share one request, so the model
can tell which candidate a later one replaced, and profile groups of different domains that share a memory share
one request, so its fact is written once.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from statistics import fmean
from typing import Any

from weclone.data.agent import distill_profile
from weclone.data.agent.contacts import CONTACT_ID
from weclone.data.agent.llm_cache import LLMCache, cache_key
from weclone.prompts.memory_organization import (
    BATCH_SUMMARY_SCHEMA,
    CLASSIFY_SCHEMA,
    DOMAIN_SCHEMA,
    PEER_SCHEMA,
    PROFILE_DIMS,
    SENSITIVITY_LEVELS,
    SUMMARY_SCHEMA,
    TAG_DOMAIN_SCHEMA,
    TIME_SCOPES,
    TOPIC_SCHEMA,
    batch_summary_prompt,
    classify_prompt,
    domain_prompt,
    peer_prompt,
    summarize_prompt,
    tag_domain_prompt,
    topic_prompt,
)
from weclone.utils import secure_storage

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STAGES = ("prepare", "domains", "peers", "classify", "topics", "summarize")
# Domain induction and tag mapping read the tags used at least DOMAIN_TAG_MIN_COUNT times, keeping at least
# DOMAIN_TAG_FLOOR and at most DOMAIN_TAG_LIMIT of the most used tags.
DOMAIN_TAG_MIN_COUNT = 3
DOMAIN_TAG_FLOOR = 200
DOMAIN_TAG_LIMIT = 1000
TAG_BATCH = 250
MAX_DOMAINS = 20
EXAMPLE_CHARS = 30
# A peer's relation is read from at most PEER_MEMORY_LIMIT of the memories that mention the peer.
PEER_MEMORY_LIMIT = 200
PEER_RELATION_CHARS = 100
# Extraction writes the chat peer as A; prompts show the peer id in its place.
PEER_PLACEHOLDER = re.compile(r"(?<![A-Za-zＡ-Ｚａ-ｚ0-9０-９_])[AＡ](?![A-Za-zＡ-Ｚａ-ｚ0-9０-９_]|股)")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path: Path, value: Any) -> None:
    distill_profile.atomic_save_any_json(path, value, indent=2)


def load(path: Path) -> Any:
    return secure_storage.read_json(path)


def source_statistics(source_ids: list[str], by_id: dict[str, dict]) -> dict:
    """同一聊天内先平均，再对不同聊天等权平均。"""
    samples = defaultdict(list)
    for sid in source_ids:
        record = by_id[sid]
        samples[(record["peer"], record["sample"])].append(record)
    return {
        "support_count": len(samples),
        **{
            f"source_{field}_mean": round(
                fmean(fmean(record[field] for record in records) for records in samples.values()), 4
            )
            for field in ("confidence", "importance")
        },
    }


def allowed_dimensions(record: dict) -> tuple[int, ...]:
    if record["kind"] != "S":
        return (5, 7)
    if record["type"] == "preference":
        return (8,)
    if record["type"] == "goal":
        return (1, 3, 4, 5, 6)
    # Habits arrive as stable facts, recent ones as current states.
    return (1, 3, 4, 6, 8)


def read_memories(input_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    """Memories passing the score filters, and the chat_with name behind each peer id."""
    records: dict[str, dict] = {}
    peers: dict[str, dict] = {}
    sample_ids: dict[str, str] = {}
    for directory, field in (("state_people", "state_memories"), ("event_people", "event_memories")):
        for path in secure_storage.iter_files(input_dir / directory):
            if "manifest" in path.name:
                continue
            samples = load(path)
            if not isinstance(samples, list):
                raise ValueError(f"Expected sample array: {path}")
            for index, sample in enumerate(samples):
                sid = distill_profile.sample_id_for(sample, index)
                peer = sample.get("chat_with_id")
                if not isinstance(peer, str) or not CONTACT_ID.fullmatch(peer):
                    raise ValueError(f"chat_with_id must be a contact id such as P3: {path}, sample {sid}")
                peers.setdefault(peer, {"chat_with": sample.get("chat_with")})
                sample_key = sample_ids.setdefault(
                    digest([peer, sid, sample.get("time", "")]),
                    f"C{len(sample_ids) + 1}",
                )
                payload = sample.get(field)
                if payload is None:
                    raise ValueError(f"Missing {field}: {path}, sample {sid}")
                branches = (
                    [("S", payload, "content")]
                    if field == "state_memories"
                    else [
                        ("ES", payload.get("surface_events", []), "surface_event"),
                        ("EI", payload.get("inferred_events", []), "inferred_event"),
                    ]
                )
                for kind, memories, content_field in branches:
                    for memory_index, memory in enumerate(memories):
                        content = memory.get(content_field)
                        if not isinstance(content, str) or not content.strip():
                            raise ValueError(f"Empty memory: {path}, sample {sid}, index {memory_index}")
                        memory_key = digest([kind, sample_key, memory])
                        origin = {
                            "file": str(secure_storage.logical_path(path)),
                            "sample_id": sid,
                            "sample_index": index,
                            "kind": kind,
                            "memory_index": memory_index,
                        }
                        if memory_key in records:
                            records[memory_key]["origins"].append(origin)
                            continue
                        record = {
                            "id": f"{kind}:{len(records) + 1}",
                            "sample": sample_key,
                            "peer": peer,
                            "kind": kind,
                            "sample_time": sample.get("time", ""),
                            "content": content,
                            "type": memory.get("type", kind),
                            "tags": memory.get("tags", []),
                            "importance": memory["importance"],
                            "confidence": memory["confidence"],
                            "origins": [origin],
                        }
                        for key in (
                            "preference_type",
                            "status",
                            "event_time",
                            "event_types",
                            "people",
                            "locations",
                        ):
                            if key in memory:
                                record[key] = memory[key]
                        records[memory_key] = record
    selected = [row for row in records.values() if row["importance"] >= 2 and row["confidence"] >= 3]
    if not selected:
        raise ValueError(f"No extracted memories match the input and score filters in {input_dir}")
    return sorted(selected, key=lambda row: (row["sample_time"], row["sample"], row["id"])), peers


def prompt_record(record: dict, *, include_sample_time: bool = False) -> dict:
    row = {key: record[key] for key in ("id", "content")}
    for key in ("event_time", "status"):
        if record.get(key):
            row[key] = record[key]
    if include_sample_time and record.get("sample_time"):
        row["sample_time"] = record["sample_time"]
    if record.get("sensitivity"):
        row["sensitivity"] = record["sensitivity"]
    if record.get("peer"):
        row["content"] = PEER_PLACEHOLDER.sub(lambda _: record["peer"], row["content"])
        if record["type"] == "preference":
            row["peer"] = record["peer"]
    return row


async def codex_config(command: str) -> dict:
    process = await asyncio.create_subprocess_exec(
        command,
        "app-server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    async def read_config():
        messages = [
            {
                "id": 1,
                "method": "initialize",
                "params": {
                    "clientInfo": {"name": "weclone-context", "version": "1.0"},
                },
            },
            {
                "id": 2,
                "method": "config/read",
                "params": {
                    "cwd": str(PROJECT_ROOT),
                    "includeLayers": False,
                },
            },
        ]
        for message in messages:
            process.stdin.write((json.dumps(message) + "\n").encode())
            await process.stdin.drain()
            while line := await process.stdout.readline():
                response = json.loads(line)
                if response.get("id") == message["id"]:
                    if "error" in response:
                        raise ValueError(f"Codex {message['method']} failed: {response['error']}")
                    break
            else:
                raise ValueError("Codex exited before returning its configuration")
        return response["result"]["config"]

    try:
        return await asyncio.wait_for(read_config(), timeout=30)
    finally:
        if process.returncode is None:
            process.terminate()
        await process.wait()


def resolve_context_window(args: argparse.Namespace) -> int:
    if args.max_context_tokens is not None:
        return args.max_context_tokens
    provider = args.llm_provider or distill_profile.load_agent_distill_config(args.config_path).get(
        "llm_provider"
    )
    if distill_profile.normalize_llm_provider(provider) == "codex_exec":
        config = distill_profile.load_codex_exec_config(args.config_path)
        command = config.get("command", "codex")
        effective_config = asyncio.run(codex_config(command))
        catalog = json.loads(
            subprocess.run(
                [command, "debug", "models", "--bundled"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ).stdout
        )
        for model in catalog.get("models", []):
            if model.get("slug") == config.get("model"):
                window = effective_config.get("model_context_window") or model.get("context_window")
                percent = model.get("effective_context_window_percent", 95)
                if type(window) is int and window > 0:
                    return window * percent // 100
    raise ValueError("Codex context window is unknown; specify --max-context-tokens explicitly")


def input_budget(args: argparse.Namespace) -> InputBudget:
    args.max_context_tokens = resolve_context_window(args)
    return InputBudget(args.max_context_tokens, args.token_encoding, args.tokenizer_file)


class InputBudget:
    def __init__(self, max_tokens: int, encoding: str, tokenizer_file: Path | None = None):
        self.max_tokens = max_tokens
        if tokenizer_file is not None:
            from weclone.utils.token_counter import count_tokens

            self.tokenizer = str(tokenizer_file.resolve())
            self.count = lambda text: count_tokens(text, tokenizer_file=tokenizer_file)
        else:
            import tiktoken

            tokenizer = tiktoken.get_encoding(encoding)
            self.tokenizer = encoding
            self.count = lambda text: len(tokenizer.encode(text, disallowed_special=()))

    def fits(self, prompt: str) -> bool:
        return self.count(prompt) <= self.max_tokens

    def check(self, prompt: str) -> dict:
        tokens = self.count(prompt)
        if tokens > self.max_tokens:
            raise ValueError(
                "Prompt exceeds input budget: "
                f"{tokens}/{self.max_tokens} tokens ({self.tokenizer}); rerun prepare or increase the budget"
            )
        return {"chars": len(prompt), "tokens": tokens, "tokenizer": self.tokenizer}


def record_tags(record: dict) -> list[str]:
    return list(dict.fromkeys(tag.strip() for tag in record["tags"] if isinstance(tag, str) and tag.strip()))


def tag_counts(records: list[dict]) -> Counter:
    return Counter(tag for row in records for tag in record_tags(row))


def domain_tags(counts: Counter) -> list[str]:
    ranked = sorted(counts, key=lambda tag: (-counts[tag], tag))
    frequent = sum(counts[tag] >= DOMAIN_TAG_MIN_COUNT for tag in ranked)
    return ranked[: min(DOMAIN_TAG_LIMIT, max(frequent, DOMAIN_TAG_FLOOR))]


def validate_domains(payload: Any) -> list[dict]:
    if (
        not isinstance(payload, dict)
        or set(payload) != {"domains"}
        or not isinstance(payload["domains"], list)
    ):
        raise ValueError("Domain induction requires domains array")
    if not 1 <= len(payload["domains"]) <= MAX_DOMAINS:
        raise ValueError(f"Domain induction must return 1 to {MAX_DOMAINS} domains")
    domains, names = [], set()
    for domain in payload["domains"]:
        if not isinstance(domain, dict) or set(domain) != {"name", "description"}:
            raise ValueError("Domain requires name and description")
        name = domain["name"].strip() if isinstance(domain["name"], str) else ""
        if not chinese_attr(name) or name in names:
            raise ValueError("Domain names must be unique Chinese names")
        if not isinstance(domain["description"], str) or not domain["description"].strip():
            raise ValueError("Domain description is empty")
        names.add(name)
        domains.append({"name": name, "description": domain["description"].strip()})
    return domains


def validate_tag_domains(payload: Any, ids: set[int], domain_count: int) -> dict[int, int]:
    if not isinstance(payload, dict) or set(payload) != {"items"} or not isinstance(payload["items"], list):
        raise ValueError("Tag mapping requires items array")
    mapping = {}
    for item in payload["items"]:
        if not isinstance(item, dict) or set(item) != {"id", "domain"}:
            raise ValueError("Tag mapping item requires id and domain")
        if type(item["id"]) is not int or item["id"] not in ids or item["id"] in mapping:
            raise ValueError("Unknown or duplicate tag id")
        if type(item["domain"]) is not int or not 1 <= item["domain"] <= domain_count:
            raise ValueError("Unknown domain id")
        mapping[item["id"]] = item["domain"]
    if len(mapping) != len(ids):
        raise ValueError("Missing tag ids")
    return mapping


def peer_memories(records: list[dict], limit: int = PEER_MEMORY_LIMIT) -> dict[str, list[dict]]:
    """Each peer's memories that mention the peer: state memories first, then events by importance, oldest first."""
    mentions = defaultdict(list)
    for row in records:
        if PEER_PLACEHOLDER.search(row["content"]):
            mentions[row["peer"]].append(row)
    selected = {}
    for peer in sorted(mentions, key=lambda peer: (len(peer), peer)):
        ranked = sorted(
            mentions[peer],
            key=lambda row: (row["kind"] != "S", -row["importance"], row["sample_time"], row["id"]),
        )
        selected[peer] = sorted(ranked[:limit], key=lambda row: (row["sample_time"], row["id"]))
    return selected


def validate_peer(payload: Any) -> str:
    """The peer's relation to B in one line; empty when the memories do not show it."""
    if (
        not isinstance(payload, dict)
        or set(payload) != {"relation"}
        or not isinstance(payload["relation"], str)
    ):
        raise ValueError("Peer relation requires relation")
    relation = " ".join(payload["relation"].split())
    if len(relation) > PEER_RELATION_CHARS:
        raise ValueError("Peer relation is too long")
    return relation


def provisional_domains(records: list[dict], mapping: dict[str, int], counts: Counter) -> list[int | None]:
    """Majority domain of each memory's mapped tags; ties go to the domain of the rarest tag."""
    domains = []
    for row in records:
        tags = [tag for tag in record_tags(row) if tag in mapping]
        votes = Counter(mapping[tag] for tag in tags)
        if not votes:
            domains.append(None)
            continue
        best = max(votes.values())
        rarest = min((tag for tag in tags if votes[mapping[tag]] == best), key=lambda tag: (counts[tag], tag))
        domains.append(mapping[rarest])
    return domains


def classification_batches(
    records: list[dict],
    domains: list[int | None],
    counts: Counter,
    *,
    max_records: int,
    prompt: Callable[[list[dict], tuple[int, ...]], str],
    budget: InputBudget | None = None,
) -> list[list[dict]]:
    """Batch memories by allowed dimensions, ordered by provisional domain, rarest shared tag and time."""

    def rarest(row):
        tags = [tag for tag in record_tags(row) if counts[tag] >= 2]
        return min(tags, key=lambda tag: (counts[tag], tag)) if tags else None

    def fit(rows, dims):
        if budget is None or budget.fits(prompt(rows, dims)):
            return [rows]
        batches, current = [], []
        for row in rows:
            if not budget.fits(prompt([row], dims)):
                raise ValueError(f"Record {row['id']} exceeds the token input budget")
            if current and not budget.fits(prompt(current + [row], dims)):
                batches.append(current)
                current = []
            current.append(row)
        return batches + [current]

    partitions: dict[tuple[int, ...], list[tuple[tuple, dict]]] = defaultdict(list)
    for row, domain in zip(records, domains, strict=True):
        tag = rarest(row)
        key = (domain is None, domain or 0, tag is None, tag or "", row["sample_time"], row["id"])
        partitions[allowed_dimensions(row)].append((key, row))
    batches = []
    for dims, rows in partitions.items():
        ordered = [row for _, row in sorted(rows, key=lambda pair: pair[0])]
        for start in range(0, len(ordered), max_records):
            batches.extend(fit(ordered[start : start + max_records], dims))
    return batches


def chinese_attr(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and bool(re.search(r"[\u3400-\u9fff]", value))


def validate_classification(payload: Any, records: list[dict], domain_count: int) -> list[dict]:
    if not isinstance(payload, dict) or set(payload) != {"items"} or not isinstance(payload["items"], list):
        raise ValueError("Classification requires items array")
    expected = {row["id"]: row for row in records}
    seen = set()
    for item in payload["items"]:
        if not isinstance(item, dict) or set(item) != {"id", "fields", "sensitivity"}:
            raise ValueError("Classification item requires id, fields and sensitivity")
        if item["sensitivity"] not in SENSITIVITY_LEVELS:
            raise ValueError("Unknown sensitivity level")
        rid = item["id"]
        if not isinstance(rid, str) or rid not in expected or rid in seen:
            raise ValueError("Unknown or duplicate classification id")
        seen.add(rid)
        if not isinstance(item["fields"], list):
            raise ValueError("fields must be an array")
        pairs = set()
        for field in item["fields"]:
            if (
                not isinstance(field, dict)
                or set(field) != {"dim", "domain", "attr"}
                or type(field["dim"]) is not int
                or field["dim"] not in allowed_dimensions(expected[rid])
                or type(field["domain"]) is not int
                or not 1 <= field["domain"] <= domain_count
                or not chinese_attr(field["attr"])
            ):
                raise ValueError("Invalid dimension/source combination, domain or Chinese attribute")
            field["attr"] = field["attr"].strip()
            pair = (field["dim"], field["attr"])
            if pair in pairs:
                raise ValueError("Duplicate attribute assignment")
            pairs.add(pair)
    if seen != set(expected):
        raise ValueError("Missing classification ids")
    return payload["items"]


def validate_summary(payload: Any, source_ids: set[str]) -> dict:
    if not isinstance(payload, dict) or set(payload) != {"facts"}:
        raise ValueError("Summary requires facts")
    if not isinstance(payload["facts"], list):
        raise ValueError("Summary facts must be an array")
    for fact in payload["facts"]:
        if not isinstance(fact, dict) or set(fact) != {"attr", "value", "source_ids", "sensitivity", "time_scope"}:
            raise ValueError("Fact requires attr, value, source_ids, sensitivity and time_scope")
        if fact["sensitivity"] not in SENSITIVITY_LEVELS:
            raise ValueError("Unknown sensitivity level")
        if fact["time_scope"] not in TIME_SCOPES:
            raise ValueError("Unknown time scope")
        if not chinese_attr(fact["attr"]) or not isinstance(fact["value"], str) or not fact["value"].strip():
            raise ValueError("Missing Chinese attribute or value")
        ids = fact["source_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(i, str) for i in ids)
            or len(ids) != len(set(ids))
            or set(ids) - source_ids
        ):
            raise ValueError("Duplicate or unknown fact sources")
    return payload


def topic_inputs(classifications: list[dict], by_id: dict[str, dict]) -> dict[int, list[dict]]:
    """Profile attributes of each domain, numbered for its topic request, with memory counts and one example."""
    members: dict[tuple[int, str, int], list[str]] = defaultdict(list)
    for item in classifications:
        for field in item["fields"]:
            if field["dim"] in PROFILE_DIMS:
                members[(field["domain"], field["attr"], field["dim"])].append(item["id"])
    inputs: dict[int, list[dict]] = defaultdict(list)
    for (domain, attr, dim), ids in sorted(members.items()):
        example = min(
            (by_id[rid] for rid in ids), key=lambda row: (-row["importance"], row["sample_time"], row["id"])
        )
        rows = inputs[domain]
        rows.append(
            {
                "id": len(rows) + 1,
                "dim": dim,
                "attr": attr,
                "count": len(ids),
                "example": " ".join(prompt_record(example)["content"].split())[:EXAMPLE_CHARS],
            }
        )
    return dict(inputs)


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
        items = topic["items"]
        if not isinstance(items, list) or not items or any(type(i) is not int for i in items):
            raise ValueError("Topic items must be attribute ids")
        names.add(name)
        seen.extend(items)
        topics.append({"name": name, "items": items})
    if sorted(seen) != list(range(1, count + 1)):
        raise ValueError("Every attribute id must appear in exactly one topic")
    return topics


def summary_groups(classifications: list[dict], by_id: dict[str, dict], topics: list[dict]) -> list[dict]:
    """Summary units: (topic, dimension) for profile dimensions, chat peer for 6, domain for 7."""
    topic_of = {}
    for topic in topics:
        for dim, attr in topic["attrs"]:
            key = (topic["domain"], dim, attr)
            if key in topic_of:
                raise ValueError("Topics do not match classifications")
            topic_of[key] = topic["name"]
    members: dict[tuple, dict[str, set[str]]] = defaultdict(lambda: {"attrs": set(), "source_ids": set()})
    expected = set()
    for item in classifications:
        for field in item["fields"]:
            dim, domain, attr = field["dim"], field["domain"], field["attr"]
            if dim == 6:
                key = (6, 0, "", by_id[item["id"]]["peer"])
            elif dim in PROFILE_DIMS:
                expected.add((domain, dim, attr))
                if (domain, dim, attr) not in topic_of:
                    raise ValueError("Topics do not match classifications")
                key = (dim, domain, topic_of[(domain, dim, attr)], "")
            else:
                key = (dim, domain, "", "")
            members[key]["attrs"].add(attr)
            members[key]["source_ids"].add(item["id"])
    if expected != set(topic_of):
        raise ValueError("Topics do not match classifications")
    groups = []
    for (dim, domain, topic, peer), group in sorted(members.items()):
        entry = {"dim": dim}
        if dim == 6:
            entry["peer"] = peer
        else:
            entry["domain"] = domain
            if topic:
                entry["topic"] = topic
        groups.append({**entry, "attrs": sorted(group["attrs"]), "source_ids": sorted(group["source_ids"])})
    return groups


def task_schema(task: dict) -> dict:
    if "schema" in task:
        return task["schema"]
    if "members" in task:
        return BATCH_SUMMARY_SCHEMA
    return {
        "domains": DOMAIN_SCHEMA,
        "tag_domains": TAG_DOMAIN_SCHEMA,
        "peers": PEER_SCHEMA,
        "classify": CLASSIFY_SCHEMA,
        "topics": TOPIC_SCHEMA,
    }.get(task["stage"], SUMMARY_SCHEMA)


class LLMTasks:
    """Runs prompt tasks on the configured model. Answers live in an LLM cache keyed by stage, prompt and schema, so
    tasks answered by an earlier run, or before a failure, are not asked again."""

    def __init__(self, args: argparse.Namespace, cache_path: Path | None = None):
        from weclone.core.inference.llm_client import build_llm_client

        options = distill_profile.default_args()
        options.config_path = args.config_path
        options.llm_provider = args.llm_provider
        options = self.options = distill_profile.resolve_llm_args(options)
        self.budget = input_budget(args)
        self.cache = LLMCache(cache_path)
        self.client = build_llm_client(
            options.llm_provider,
            config_path=options.config_path,
            model=options.model,
            max_workers=options.batch_size,
            timeout=options.timeout,
            effort=options.effort,
            command=options.codex_command,
            sandbox=options.codex_sandbox,
        )

    def run(self, tasks: list[dict], validate) -> list[Any]:
        from weclone.core.inference.llm_client import LLMRequest

        input_stats = [self.budget.check(task["prompt"]) for task in tasks]
        schemas = [task_schema(task) for task in tasks]
        keys = [cache_key(task["stage"], task["prompt"], schema) for task, schema in zip(tasks, schemas)]
        answers = self.cache.get_many(keys)
        # One request per question, even when several tasks ask the same one.
        first: dict[str, int] = {}
        for i, key in enumerate(keys):
            if key not in answers:
                first.setdefault(key, i)
        pending = list(first.values())
        if tasks:
            print(
                f"{tasks[0]['stage']}: tasks={len(tasks)} pending={len(pending)} "
                f"max_input_tokens={max(item['tokens'] for item in input_stats)}",
                flush=True,
            )
        for batch in distill_profile.batched(pending, self.options.batch_size):
            retry = list(batch)
            errors: Counter[str] = Counter()
            for _ in range(2):
                requests = [
                    LLMRequest.from_prompt(
                        tasks[i]["prompt"],
                        provider=self.options.llm_provider,
                        model=self.options.model,
                        effort=self.options.effort,
                        max_tokens=self.options.max_tokens,
                        timeout=self.options.timeout,
                        json_mode=True,
                        metadata={"stage": tasks[i]["stage"], "task_id": keys[i]},
                        json_schema=schemas[i],
                    )
                    for i in retry
                ]
                responses = list(self.client.generate_batch(requests))
                failed, answered = [], []
                errors.clear()
                for position, i in enumerate(retry):
                    try:
                        if position >= len(responses):
                            raise ValueError("Missing response")
                        response = responses[position]
                        if not response.ok or response.finish_reason in {"length", "max_tokens"}:
                            raise ValueError(response.error or "Failed or truncated response")
                        validate(response.parsed_json, tasks[i])
                        answers[keys[i]] = response.parsed_json
                        answered.append((keys[i], tasks[i]["stage"], response.parsed_json, response.model))
                    except ValueError as exc:
                        errors[str(exc)] += 1
                        failed.append(i)
                self.cache.put_many(answered)
                retry = failed
                if not retry:
                    break
            if retry:
                reasons = "; ".join(f"{error} ×{count}" for error, count in errors.most_common(3))
                raise RuntimeError(
                    f"{len(retry)} LLM tasks failed ({reasons}). Rerun the same command to retry them, "
                    f"answered tasks are read from {self.cache.path}"
                )
        return [validate(answers[key], task) for key, task in zip(keys, tasks)]

    def close(self) -> None:
        self.client.close()


def pack_summary(
    dim: int,
    attrs: list[str],
    rows: list[dict],
    *,
    reduced: bool,
    budget: InputBudget | None = None,
    peer: str | None = None,
    topic: str | None = None,
    domain: int | None = None,
    max_records: int = 300,
) -> list[dict]:
    def fits(items):
        if not reduced and len(items) > max_records:
            return False
        prompt = summarize_prompt(dim, attrs, items, reduced=reduced, peer=peer, topic=topic)
        return budget is None or budget.fits(prompt)

    batches, current = [], []
    for row in rows:
        proposed = current + [row]
        if not fits(proposed):
            if not current:
                raise ValueError("Single summary record exceeds the token input budget")
            batches.append(current)
            current = [row]
            if not fits(current):
                raise ValueError("Single summary record exceeds the token input budget")
        else:
            current = proposed
    if current:
        batches.append(current)
    return [
        {
            "stage": "reduce" if reduced else "summarize",
            "dim": dim,
            "domain": domain,
            "topic": topic,
            "attrs": attrs,
            "rows": batch,
            "prompt": summarize_prompt(dim, attrs, batch, reduced=reduced, peer=peer, topic=topic),
            "source_ids": sorted(
                {sid for row in batch for sid in (row["source_ids"] if reduced else [row["id"]])}
            ),
        }
        for batch in batches
    ]


def validate_summary_batch(payload: Any, task: dict) -> list[dict]:
    if not isinstance(payload, dict) or set(payload) != {"groups"} or not isinstance(payload["groups"], list):
        raise ValueError("Batched summary requires groups array")
    results = {}
    request_sources = {sid for member in task["members"] for sid in member["source_ids"]}
    for group in payload["groups"]:
        if not isinstance(group, dict) or set(group) != {"id", "facts"}:
            raise ValueError("Batched summary group requires id and facts")
        i = group["id"]
        if type(i) is not int or i not in range(len(task["members"])) or i in results:
            raise ValueError("Unknown or duplicate summary group id")
        results[i] = validate_summary({"facts": group["facts"]}, request_sources)
        used = {sid for fact in group["facts"] for sid in fact["source_ids"]}
        cross_group = used - set(task["members"][i]["source_ids"])
        if cross_group:
            logging.getLogger(__name__).warning(
                "Summary group %s references sources from other groups in this request: %s",
                i,
                sorted(cross_group),
            )
    if len(results) != len(task["members"]):
        raise ValueError("Missing summary groups")
    return [results[i] for i in range(len(task["members"]))]


def summary_partitions(tasks: list[dict]) -> dict[tuple, list[tuple[int, dict]]]:
    """Summary tasks that may share a request: one stage and dimension, and one domain unless memories link them."""
    keys = []
    for task in tasks:
        # Identity candidates of one person spread over domains (study status under background and under
        # study, employment under job search and under work), and only one request can see which replaced which.
        domain = None if task["dim"] == 1 else task.get("domain")
        keys.append((task["stage"], task["dim"], domain))
    # A memory classified under several domains feeds a group in each, and separate requests each write its fact.
    # Profile groups linked by shared memories leave their domains for one partition, so one request writes the fact
    # once. Dimension 7 is summarized per domain, and linking it would merge whole domains.
    parent = list(range(len(tasks)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[tuple, int] = {}
    for i, task in enumerate(tasks):
        if task["dim"] in PROFILE_DIMS:
            for sid in task["source_ids"]:
                parent[root(i)] = root(first.setdefault((task["stage"], task["dim"], sid), i))
    spans = defaultdict(set)
    for i, key in enumerate(keys):
        spans[root(i)].add(key)
    partitions = defaultdict(list)
    for i, task in enumerate(tasks):
        linked = len(spans[root(i)]) > 1
        partitions[(*keys[i][:2], ("linked", root(i))) if linked else keys[i]].append((i, task))
    return partitions


def run_summary_tasks(
    tasks: list[dict],
    runner: LLMTasks,
    max_groups: int,
    budget: InputBudget | None,
    max_records: int = 300,
) -> list[dict]:
    partitions = summary_partitions(tasks)
    requests, positions = [], []

    def append(batch):
        members = [task for _, task in batch]
        request = (
            members[0]
            if len(members) == 1
            else {
                "stage": members[0]["stage"],
                "members": members,
                "prompt": batch_summary_prompt(members),
            }
        )
        requests.append(request)
        positions.append([i for i, _ in batch])

    for entries in partitions.values():
        if entries[0][1]["dim"] == 6:
            for entry in entries:
                append([entry])
            continue
        current = []
        for entry in entries:
            proposed = current + [entry]
            source_count = len({sid for _, task in proposed for sid in task["source_ids"]})
            if current and (
                len(proposed) > max_groups
                or (entry[1]["stage"] == "summarize" and source_count > max_records)
                or (budget and not budget.fits(batch_summary_prompt([task for _, task in proposed])))
            ):
                append(current)
                current = []
            current.append(entry)
        if current:
            append(current)
    results = runner.run(
        requests,
        lambda payload, task: (
            validate_summary_batch(payload, task)
            if "members" in task
            else [validate_summary(payload, set(task["source_ids"]))]
        ),
    )
    ordered = [None] * len(tasks)
    for indices, batch in zip(positions, results):
        for i, result in zip(indices, batch):
            ordered[i] = result
    return ordered


def summarize_groups(
    groups: list[dict],
    by_id: dict[str, dict],
    runner: LLMTasks,
    budget: InputBudget | None = None,
    max_groups: int = 30,
    max_records: int = 300,
) -> list[dict]:
    states = []
    for group in groups:
        records = sorted(
            (by_id[rid] for rid in group["source_ids"]), key=lambda row: (row["sample_time"], row["id"])
        )
        rows = [prompt_record(row, include_sample_time=True) for row in records]
        states.append({"rows": rows, "reduced": False, "result": None})
    while any(state["result"] is None for state in states):
        tasks, owners = [], []
        for i, (group, state) in enumerate(zip(groups, states)):
            if state["result"] is not None:
                continue
            batch = pack_summary(
                group["dim"],
                group["attrs"],
                state["rows"],
                reduced=state["reduced"],
                budget=budget,
                peer=group.get("peer"),
                topic=group.get("topic"),
                domain=group.get("domain"),
                max_records=max_records,
            )
            tasks.extend(batch)
            owners.extend([i] * len(batch))
        results = run_summary_tasks(tasks, runner, max_groups, budget, max_records=max_records)
        by_group = defaultdict(list)
        for i, result in zip(owners, results):
            by_group[i].append(result)
        for i, batch_results in by_group.items():
            state = states[i]
            facts = [fact for result in batch_results for fact in result["facts"]]
            if len(batch_results) == 1 or not facts:
                state["result"] = {"facts": facts}
            elif state["reduced"] and len(json.dumps(facts, ensure_ascii=False)) >= len(
                json.dumps(state["rows"], ensure_ascii=False)
            ):
                raise ValueError("Cross-batch results do not shrink within the model context window")
            else:
                state["rows"], state["reduced"] = facts, True
    return [state["result"] for state in states]


def run(args: argparse.Namespace) -> dict:
    if args.dry_run:
        records, _ = read_memories(args.input_dir)
        return {"records": len(records), "by_kind": dict(Counter(row["kind"] for row in records))}
    budget = input_budget(args)
    print(f"codex_context_tokens={args.max_context_tokens}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.output_dir / "records.json"
    peer_path = args.output_dir / "peers.json"
    domain_path = args.output_dir / "domains.json"
    relation_path = args.output_dir / "peer_relations.json"
    task_path = args.output_dir / "classification_tasks.json"
    classification_path = args.output_dir / "classifications.json"
    topic_path = args.output_dir / "topics.json"
    stages = STAGES if args.stage == "all" else (args.stage,)
    runner = None
    try:
        if "prepare" in stages:
            records, peers = read_memories(args.input_dir)
            if secure_storage.file_exists(record_path) and load(record_path) != records:
                raise ValueError("Input snapshot changed; use a new output directory")
            save(record_path, records)
            save(peer_path, peers)
        else:
            records = load(record_path)
        by_id = {row["id"]: row for row in records}
        counts = tag_counts(records)
        if "domains" in stages:
            tags = domain_tags(counts)
            if not tags:
                raise ValueError("Extracted memories carry no tags; domains cannot be induced")
            runner = runner or LLMTasks(args)
            domains = runner.run(
                [{"stage": "domains", "prompt": domain_prompt([(tag, counts[tag]) for tag in tags])}],
                lambda payload, _: validate_domains(payload),
            )[0]
            tasks = [
                {
                    "stage": "tag_domains",
                    "ids": [i for i, _ in batch],
                    "prompt": tag_domain_prompt(domains, batch),
                }
                for batch in distill_profile.batched(list(enumerate(tags, 1)), TAG_BATCH)
            ]
            results = runner.run(
                tasks, lambda payload, task: validate_tag_domains(payload, set(task["ids"]), len(domains))
            )
            mapping = {tags[i - 1]: domain for result in results for i, domain in result.items()}
            save(domain_path, {"domains": domains, "tags": mapping})
        if "peers" in stages:
            tasks = [
                {
                    "stage": "peers",
                    "peer": peer,
                    "prompt": peer_prompt(
                        peer, [prompt_record(row, include_sample_time=True) for row in rows]
                    ),
                }
                for peer, rows in peer_memories(records).items()
            ]
            results = []
            if tasks:
                runner = runner or LLMTasks(args)
                results = runner.run(tasks, lambda payload, _: validate_peer(payload))
            save(relation_path, {task["peer"]: relation for task, relation in zip(tasks, results)})
        if set(stages) & {"classify", "topics", "summarize"}:
            state = load(domain_path)
            domains, mapping = state["domains"], state["tags"]
        if "classify" in stages:
            if not secure_storage.file_exists(relation_path):
                raise ValueError("Run the peers stage before classify")
            relations = {peer: relation for peer, relation in load(relation_path).items() if relation}

            def prompt(rows, dims):
                return classify_prompt([prompt_record(row) for row in rows], dims, domains, relations)

            batches = classification_batches(
                records,
                provisional_domains(records, mapping, counts),
                counts,
                max_records=args.max_records,
                prompt=prompt,
                budget=budget,
            )
            tasks = [
                {
                    "stage": "classify",
                    "ids": [row["id"] for row in batch],
                    "prompt": prompt(batch, allowed_dimensions(batch[0])),
                }
                for batch in batches
            ]
            save(task_path, tasks)
            runner = runner or LLMTasks(args)
            results = runner.run(
                tasks,
                lambda payload, task: validate_classification(
                    payload, [by_id[rid] for rid in task["ids"]], len(domains)
                ),
            )
            classifications = [item for batch in results for item in batch]
            validate_classification({"items": classifications}, records, len(domains))
            save(classification_path, classifications)
        if "topics" in stages:
            classifications = validate_classification(
                {"items": load(classification_path)}, records, len(domains)
            )
            inputs = topic_inputs(classifications, by_id)
            tasks = [
                {
                    "stage": "topics",
                    "domain": domain,
                    "prompt": topic_prompt(
                        domains, domain, rows, args.topic_min_memories, args.topic_max_memories
                    ),
                }
                for domain, rows in sorted(inputs.items())
            ]
            results = []
            if tasks:
                runner = runner or LLMTasks(args)
                results = runner.run(
                    tasks, lambda payload, task: validate_topics(payload, len(inputs[task["domain"]]))
                )
            topics = []
            for task, result in zip(tasks, results):
                rows = inputs[task["domain"]]
                topics.extend(
                    {
                        "domain": task["domain"],
                        "name": topic["name"],
                        "attrs": [[rows[i - 1]["dim"], rows[i - 1]["attr"]] for i in topic["items"]],
                    }
                    for topic in result
                )
            save(topic_path, topics)
        if "summarize" in stages:
            classifications = validate_classification(
                {"items": load(classification_path)}, records, len(domains)
            )
            topics = load(topic_path)
            relations = load(relation_path)
            # Summaries see each memory's privacy level, and the output sources keep it.
            levels = {item["id"]: item["sensitivity"] for item in classifications}
            by_id = {rid: {**row, "sensitivity": levels[rid]} for rid, row in by_id.items()}
            groups = summary_groups(classifications, by_id, topics)
            runner = runner or LLMTasks(args)
            results = summarize_groups(
                groups,
                by_id,
                runner,
                budget=budget,
                max_groups=args.max_summary_groups,
                max_records=args.max_summary_records,
            )
            referenced_by_dim = defaultdict(set)
            for group, result in zip(groups, results):
                referenced_by_dim[group["dim"]].update(
                    sid for fact in result["facts"] for sid in fact["source_ids"]
                )
            facts, unused = [], []
            for group, result in zip(groups, results):
                labels = {
                    "dim": group["dim"],
                    "domain": domains[group["domain"] - 1]["name"] if "domain" in group else None,
                    "topic": group.get("topic"),
                }
                if "peer" in group:
                    labels["peer"] = group["peer"]
                facts.extend(
                    {**labels, **fact, **source_statistics(fact["source_ids"], by_id)}
                    for fact in result["facts"]
                )
                unreferenced = sorted(set(group["source_ids"]) - referenced_by_dim[group["dim"]])
                if unreferenced:
                    unused.append({**labels, "attrs": group["attrs"], "source_ids": unreferenced})
            order = {domain["name"]: i for i, domain in enumerate(domains, 1)}
            facts.sort(
                key=lambda row: (
                    row["dim"],
                    order.get(row["domain"], 0),
                    row["topic"] or "",
                    row["attr"],
                    row["value"],
                )
            )
            save(
                args.output_dir / "organized_memories.json",
                {
                    "domains": domains,
                    "topics": [{**topic, "domain": domains[topic["domain"] - 1]["name"]} for topic in topics],
                    "facts": facts,
                    "unused": unused,
                    "unassigned_ids": [item["id"] for item in classifications if not item["fields"]],
                    "peers": {
                        peer: {**info, "relation": relations.get(peer, "")}
                        for peer, info in load(peer_path).items()
                    },
                    "sources": by_id,
                },
            )
        return {"stage": args.stage, "records": len(records), "output_dir": str(args.output_dir)}
    finally:
        if runner is not None:
            runner.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="distill directory of a build-profile run"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("settings.jsonc"))
    parser.add_argument("--llm-provider", choices=("api", "codex_exec"))
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        help="Local token limit; defaults to Codex's effective context window",
    )
    parser.add_argument("--token-encoding", default="o200k_base", help="tiktoken encoding used for budgeting")
    parser.add_argument("--tokenizer-file", type=Path, help="Use a local tokenizer.json instead of tiktoken")
    parser.add_argument("--max-records", type=int, default=50)
    parser.add_argument("--max-summary-groups", type=int, default=30)
    parser.add_argument(
        "--max-summary-records",
        type=int,
        default=300,
        help="Maximum unique source memories per initial summary request",
    )
    parser.add_argument(
        "--topic-min-memories",
        type=int,
        default=20,
        help="Memory count each topic should reach; sparse attributes join the closest topic",
    )
    parser.add_argument(
        "--topic-max-memories",
        type=int,
        default=100,
        help="Memory count each topic should stay under when a domain is split into topics",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Count extracted memories without model calls or writes"
    )
    args = parser.parse_args(argv)
    secure_storage.configure(args.config_path)
    for name in (
        "max_context_tokens",
        "max_records",
        "max_summary_groups",
        "max_summary_records",
        "topic_min_memories",
        "topic_max_memories",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.topic_min_memories > args.topic_max_memories:
        parser.error("--topic-min-memories must not exceed --topic-max-memories")
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
