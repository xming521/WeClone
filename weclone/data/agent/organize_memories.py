"""Organize extracted profile/event memories by dimension without reading chat messages."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

import numpy as np

from weclone.data.agent import distill_profile, group_state_memories
from weclone.prompts.memory_organization import (
    ATTRIBUTE_HIERARCHY_SCHEMA,
    BATCH_SUMMARY_SCHEMA,
    CLASSIFY_SCHEMA,
    PREFERENCE_ATTRIBUTE_SCHEMA,
    SUMMARY_SCHEMA,
    batch_summary_prompt,
    classify_prompt,
    summarize_prompt,
)
from weclone.utils import secure_storage

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
    dims = [1, 3, 4, 6]
    if record["type"] == "goal":
        dims.append(5)
    return tuple(sorted(dims))


def read_memories(input_dir: Path, *, preferences_only: bool = False) -> list[dict]:
    records: dict[str, dict] = {}
    peers: dict[str, str] = {}
    sample_ids: dict[str, str] = {}
    for directory, field in (("state_people", "state_memories"), ("event_people", "event_memories")):
        if preferences_only and field != "state_memories":
            continue
        for path in secure_storage.iter_files(input_dir / directory):
            if "manifest" in path.name:
                continue
            samples = load(path)
            if not isinstance(samples, list):
                raise ValueError(f"Expected sample array: {path}")
            for index, sample in enumerate(samples):
                sid = distill_profile.sample_id_for(sample, index)
                peer_key = digest(
                    sample.get("chat_with_id")
                    or sample.get("chat_with")
                    or secure_storage.logical_path(path).stem
                )
                peer = peers.setdefault(peer_key, f"P{len(peers) + 1}")
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
    selected = [
        row
        for row in records.values()
        if row["importance"] >= 2
        and row["confidence"] >= 3
        and (not preferences_only or row["type"] == "preference")
    ]
    if not selected:
        raise ValueError(f"No extracted memories match the input and score filters in {input_dir}")
    return sorted(selected, key=lambda row: (row["sample_time"], row["sample"], row["id"]))


def prompt_record(record: dict, *, include_sample_time: bool = False) -> dict:
    row = {key: record[key] for key in ("id", "content")}
    for key in ("event_time", "status"):
        if record.get(key):
            row[key] = record[key]
    if include_sample_time and record.get("sample_time"):
        row["sample_time"] = record["sample_time"]
    if record.get("peer"):
        row["content"] = re.sub(
            r"(?<![A-Za-zＡ-Ｚａ-ｚ0-9０-９_])[AＡ](?![A-Za-zＡ-Ｚａ-ｚ0-9０-９_]|股)",
            lambda _: record["peer"],
            row["content"],
        )
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
    if provider == "codex_exec":
        config = distill_profile.load_agent_distill_config(args.config_path)
        if not config.get("model"):
            config = {**distill_profile.load_codex_exec_config(args.config_path), **config}
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


def normalized_vectors(vectors: list[list[float]]) -> np.ndarray:
    array = np.asarray(vectors, dtype=float)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Invalid embedding matrix")
    lengths = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(lengths == 0):
        raise ValueError("Zero embedding vector")
    return array / lengths


class Embeddings:
    def __init__(self, args: argparse.Namespace):
        self.args = group_state_memories.default_args()
        self.args.config_path = args.config_path
        self.args.embedding_url = args.embedding_url
        self.args.auto_start_embedding_service = not args.no_auto_start_embedding_service
        if args.embedding_batch_size is not None:
            self.args.embedding_batch_size = args.embedding_batch_size
        self.args.embedding_service_log_path = args.output_dir / "embedding_service.log"
        self.batch_dir = args.output_dir / "embeddings"
        self.process = None
        self.cache = None

    def get(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.cache is None:
            self.url = group_state_memories.service_base_url(self.args)
            health, self.process = group_state_memories.ensure_embedding_service(self.args, base_url=self.url)
            identity = {key: health.get(key) for key in ("model", "max_length", "device")}
            cache = {"identity": identity, "vectors": {}}
            for path in secure_storage.iter_files(self.batch_dir):
                shard = load(path)
                if shard["identity"] != identity:
                    raise ValueError("Embedding model changed; use a new output directory")
                cache["vectors"].update(shard["vectors"])
            self.cache = cache
        missing = list(dict.fromkeys(text for text in texts if digest(text) not in self.cache["vectors"]))
        total = len(set(texts))
        completed = total - len(missing)
        print(f"embedding: cached={completed}/{total} pending={len(missing)}", flush=True)
        for batch in distill_profile.batched(missing, self.args.embedding_batch_size):
            started = perf_counter()
            vectors = group_state_memories.embed_texts(
                batch,
                base_url=self.url,
                batch_size=self.args.embedding_batch_size,
                timeout=self.args.request_timeout,
            )
            if len(vectors) != len(batch):
                raise ValueError("Embedding response count mismatch")
            normalized_vectors(vectors)
            updates = {digest(text): vector for text, vector in zip(batch, vectors)}
            embedded = perf_counter()
            save(
                self.batch_dir / f"{digest(list(updates))}.json",
                {
                    "identity": self.cache["identity"],
                    "vectors": updates,
                },
            )
            self.cache["vectors"].update(updates)
            completed += len(batch)
            print(
                f"embedding: {completed}/{total} compute={embedded - started:.2f}s "
                f"save={perf_counter() - embedded:.2f}s",
                flush=True,
            )
        return [self.cache["vectors"][digest(text)] for text in texts]

    def close(self) -> None:
        if self.process is not None:
            group_state_memories.stop_embedding_service_process(
                self.process,
                timeout=self.args.embedding_service_stop_timeout,
            )


def classification_batches(
    records: list[dict],
    vectors: list[list[float]],
    *,
    max_records: int,
    neighbors: int,
    budget: InputBudget | None = None,
) -> list[list[dict]]:
    matrix = normalized_vectors(vectors)
    if len(matrix) != len(records):
        raise ValueError("Embedding count does not match records")
    tags = [set(row["tags"]) for row in records]
    counts = Counter(tag for row in tags for tag in row)
    weights = {tag: float(np.log((len(records) + 1) / (count + 1))) for tag, count in counts.items()}
    partitions: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i, row in enumerate(records):
        partitions[allowed_dimensions(row)].append(i)
    batches = []
    for dims, indices in partitions.items():
        remaining = set(indices)
        while remaining:
            seed = min(remaining)
            candidates = sorted(remaining - {seed})
            scores = matrix[candidates] @ matrix[seed] if candidates else []
            nearest = sorted(zip(candidates, scores), key=lambda pair: (-pair[1], pair[0]))[:neighbors]
            overlap = {i: sum(weights[tag] for tag in tags[i] & tags[seed]) for i in candidates}
            tagged = sorted((i for i in candidates if overlap[i] > 0), key=lambda i: (-overlap[i], i))[
                :neighbors
            ]
            ranks: dict[int, float] = defaultdict(float)
            for channel in ([i for i, _ in nearest], tagged):
                for rank, i in enumerate(channel, 1):
                    ranks[i] += 1 / rank
            same_sample = [i for i in candidates if records[i]["sample"] == records[seed]["sample"]]
            rest = [i for i, _ in sorted(zip(candidates, scores), key=lambda pair: (-pair[1], pair[0]))]
            order = list(
                dict.fromkeys([seed] + same_sample + sorted(ranks, key=lambda i: (-ranks[i], i)) + rest)
            )
            selected = []
            for i in order:
                if len(selected) >= max_records:
                    break
                proposed = selected + [i]
                prompt = classify_prompt([prompt_record(records[j]) for j in proposed], dims)
                if budget and not budget.fits(prompt):
                    if not selected:
                        raise ValueError(f"Record {records[i]['id']} exceeds the token input budget")
                    continue
                selected = proposed
            batches.append([records[i] for i in selected])
            remaining.difference_update(selected)
    return batches


def chinese_attr(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and bool(re.search(r"[\u3400-\u9fff]", value))


def validate_classification(payload: Any, records: list[dict]) -> list[dict]:
    if not isinstance(payload, dict) or set(payload) != {"items"} or not isinstance(payload["items"], list):
        raise ValueError("Classification requires items array")
    expected = {row["id"]: row for row in records}
    seen = set()
    for item in payload["items"]:
        if not isinstance(item, dict) or set(item) != {"id", "fields"}:
            raise ValueError("Classification item requires id and fields")
        rid = item["id"]
        if not isinstance(rid, str) or rid not in expected or rid in seen:
            raise ValueError("Unknown or duplicate classification id")
        seen.add(rid)
        if not isinstance(item["fields"], list):
            raise ValueError("fields must be an array")
        fields = set()
        for field in item["fields"]:
            if (
                not isinstance(field, list)
                or len(field) != 2
                or type(field[0]) is not int
                or field[0] not in allowed_dimensions(expected[rid])
                or not chinese_attr(field[1])
            ):
                raise ValueError("Invalid dimension/source combination or Chinese attribute")
            pair = (field[0], field[1].strip())
            if pair in fields:
                raise ValueError("Duplicate attribute assignment")
            fields.add(pair)
            field[1] = pair[1]
    if seen != set(expected):
        raise ValueError("Missing classification ids")
    return payload["items"]


def validate_summary(payload: Any, source_ids: set[str]) -> dict:
    if not isinstance(payload, dict) or set(payload) != {"facts"}:
        raise ValueError("Summary requires facts")
    if not isinstance(payload["facts"], list):
        raise ValueError("Summary facts must be an array")
    for fact in payload["facts"]:
        if not isinstance(fact, dict) or set(fact) != {"attr", "value", "source_ids"}:
            raise ValueError("Fact requires attr, value and source_ids")
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


def validate_preference_attributes(payload: Any, records: list[dict]) -> list[dict]:
    if not isinstance(payload, dict) or set(payload) != {"items"} or not isinstance(payload["items"], list):
        raise ValueError("Preference attributes require items array")
    items = []
    for item in payload["items"]:
        if not isinstance(item, dict) or set(item) != {"id", "attrs"} or not isinstance(item["attrs"], list):
            raise ValueError("Preference attribute item requires id and attrs")
        items.append({"id": item["id"], "fields": [[8, attr] for attr in item["attrs"]]})
    return validate_classification({"items": items}, records)


class LLMTasks:
    def __init__(self, args: argparse.Namespace):
        from weclone.core.inference.llm_client import build_llm_client

        options = distill_profile.default_args()
        options.config_path = args.config_path
        options.llm_provider = args.llm_provider
        options = self.options = distill_profile.resolve_llm_args(options)
        self.budget = input_budget(args)
        self.path = args.output_dir / "llm_checkpoint.json"
        self.entries = load(self.path) if secure_storage.file_exists(self.path) else {}
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
        schemas = [
            ATTRIBUTE_HIERARCHY_SCHEMA
            if task["stage"] == "attribute_hierarchy"
            else PREFERENCE_ATTRIBUTE_SCHEMA
            if task["stage"] == "attributes"
            else CLASSIFY_SCHEMA
            if task["stage"] == "classify"
            else BATCH_SUMMARY_SCHEMA
            if "members" in task
            else SUMMARY_SCHEMA
            for task in tasks
        ]
        keys = [digest([task["stage"], task["prompt"], schema]) for task, schema in zip(tasks, schemas)]
        pending = [
            i
            for i, key in enumerate(keys)
            if self.options.overwrite or self.entries.get(key, {}).get("status") != "done"
        ]
        if tasks:
            print(
                f"{tasks[0]['stage']}: tasks={len(tasks)} pending={len(pending)} "
                f"max_input_tokens={max(item['tokens'] for item in input_stats)}",
                flush=True,
            )
        for batch in distill_profile.batched(pending, self.options.batch_size):
            retry = list(batch)
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
                failed = []
                for position, i in enumerate(retry):
                    error = "Missing response"
                    try:
                        if position >= len(responses):
                            raise ValueError(error)
                        response = responses[position]
                        if not response.ok or response.finish_reason in {"length", "max_tokens"}:
                            raise ValueError(response.error or "Failed or truncated response")
                        validate(response.parsed_json, tasks[i])
                        self.entries[keys[i]] = {
                            "status": "done",
                            "stage": tasks[i]["stage"],
                            "result": response.parsed_json,
                            "input": input_stats[i],
                            "usage": response.metadata.get("usage"),
                            "elapsed_s": response.elapsed_s,
                            "model": response.model,
                        }
                    except ValueError as exc:
                        self.entries[keys[i]] = {"status": "failed", "error": str(exc)}
                        failed.append(i)
                save(self.path, self.entries)
                retry = failed
                if not retry:
                    break
            if retry:
                raise RuntimeError(f"{len(retry)} LLM tasks failed; rerun to resume from {self.path}")
        results = []
        for key, task in zip(keys, tasks):
            payload = self.entries[key]["result"]
            results.append(validate(payload, task))
        return results

    def close(self) -> None:
        self.client.close()


def attribute_groups(
    assignments: list[dict], embed, *, threshold: float, by_id: dict[str, dict]
) -> list[dict]:
    attributes: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    relations: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for item in assignments:
        for dim, attr in item["fields"]:
            members = relations[by_id[item["id"]]["peer"]] if dim == 6 else attributes[dim]
            members[attr].add(item["id"])
    groups = []
    for dim, members in sorted(attributes.items()):
        names = sorted(members)
        vectors = normalized_vectors(embed(names))
        similarity = vectors @ vectors.T
        parents = list(range(len(names)))

        def root(i: int) -> int:
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        for i in range(len(names)):
            for j in np.flatnonzero(similarity[i, :i] >= threshold):
                left, right = root(i), root(int(j))
                parents[max(left, right)] = min(left, right)
        clusters: dict[int, list[int]] = defaultdict(list)
        for i in range(len(names)):
            clusters[root(i)].append(i)
        for cluster in clusters.values():
            attrs = [names[i] for i in cluster]
            ids = sorted(set().union(*(members[attr] for attr in attrs)))
            groups.append({"dim": dim, "attrs": attrs, "source_ids": ids})
    for peer, members in sorted(relations.items()):
        groups.append(
            {
                "dim": 6,
                "peer": peer,
                "attrs": sorted(members),
                "source_ids": sorted(set().union(*members.values())),
            }
        )
    return sorted(groups, key=lambda group: group["dim"])


def pack_summary(
    dim: int,
    attrs: list[str],
    rows: list[dict],
    *,
    reduced: bool,
    budget: InputBudget | None = None,
    peer: str | None = None,
    max_records: int = 300,
) -> list[dict]:
    def fits(items):
        if not reduced and len(items) > max_records:
            return False
        prompt = summarize_prompt(dim, attrs, items, reduced=reduced, peer=peer)
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
            "attrs": attrs,
            "rows": batch,
            "prompt": summarize_prompt(dim, attrs, batch, reduced=reduced, peer=peer),
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


def run_summary_tasks(
    tasks: list[dict],
    runner: LLMTasks,
    max_groups: int,
    budget: InputBudget | None,
    max_records: int = 300,
) -> list[dict]:
    partitions = defaultdict(list)
    for i, task in enumerate(tasks):
        partitions[(task["stage"], task["dim"])].append((i, task))
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
        records = read_memories(args.input_dir, preferences_only=args.preferences_only)
        return {"records": len(records), "by_kind": dict(Counter(row["kind"] for row in records))}
    budget = input_budget(args)
    print(f"codex_context_tokens={args.max_context_tokens}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.output_dir / "records.json"
    task_path = args.output_dir / "classification_tasks.json"
    classification_path = args.output_dir / "classifications.json"
    group_path = args.output_dir / "attribute_groups.json"
    embedding = None
    runner = None
    try:
        if args.stage in {"prepare", "all"}:
            records = read_memories(args.input_dir, preferences_only=args.preferences_only)
            if secure_storage.file_exists(record_path) and load(record_path) != records:
                raise ValueError("Input snapshot changed; use a new output directory")
            save(record_path, records)
            texts = [
                group_state_memories.embedding_text_for(
                    content=row["content"],
                    memory_type=row["type"],
                    tags=row["tags"],
                )
                for row in records
            ]
            embedding = Embeddings(args)
            batches = classification_batches(
                records,
                embedding.get(texts),
                max_records=args.max_records,
                neighbors=args.neighbor_top_k,
                budget=budget,
            )
            tasks = [
                {
                    "stage": "attributes" if allowed_dimensions(batch[0]) == (8,) else "classify",
                    "ids": [row["id"] for row in batch],
                    "prompt": classify_prompt(
                        [prompt_record(row) for row in batch], allowed_dimensions(batch[0])
                    ),
                }
                for batch in batches
            ]
            save(task_path, tasks)
        else:
            records = load(record_path)
            if args.preferences_only and any(
                row["kind"] != "S" or row["type"] != "preference" for row in records
            ):
                raise ValueError(
                    "Snapshot contains non-preference memories; use the preferences output directory"
                )
        by_id = {row["id"]: row for row in records}
        if args.stage in {"classify", "all"}:
            tasks = load(task_path)
            results = []
            if tasks:
                runner = LLMTasks(args)
                results = runner.run(
                    tasks,
                    lambda payload, task: (
                        validate_preference_attributes
                        if task["stage"] == "attributes"
                        else validate_classification
                    )(
                        payload,
                        [by_id[rid] for rid in task["ids"]],
                    ),
                )
            classifications = [item for batch in results for item in batch]
            validate_classification({"items": classifications}, records)
            save(classification_path, classifications)
        if args.stage in {"group", "all"}:
            classifications = load(classification_path)
            validate_classification({"items": classifications}, records)
            if any(dim != 6 for item in classifications for dim, _ in item["fields"]):
                embedding = embedding or Embeddings(args)
            groups = attribute_groups(
                classifications,
                embedding.get if embedding else None,
                threshold=args.attribute_similarity,
                by_id=by_id,
            )
            save(group_path, groups)
        if args.stage in {"summarize", "all"}:
            runner = runner or LLMTasks(args)
            classifications = validate_classification({"items": load(classification_path)}, records)
            expected = defaultdict(set)
            for item in classifications:
                for dim, attr in item["fields"]:
                    peer = by_id[item["id"]]["peer"] if dim == 6 else None
                    expected[(dim, peer, attr)].add(item["id"])
            groups = load(group_path)
            represented = set()
            facts, unused = [], []
            for group in groups:
                peer = group.get("peer") if group["dim"] == 6 else None
                keys = {(group["dim"], peer, attr) for attr in group["attrs"]}
                if (
                    not keys
                    or represented & keys
                    or not keys <= expected.keys()
                    or set(group["source_ids"]) != set().union(*(expected[key] for key in keys))
                ):
                    raise ValueError("Attribute group does not match classifications")
                represented.update(keys)
            if represented != set(expected):
                raise ValueError("Missing attribute groups")
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
            for group, result in zip(groups, results):
                facts.extend(
                    {
                        "dim": group["dim"],
                        **fact,
                        **source_statistics(fact["source_ids"], by_id),
                    }
                    for fact in result["facts"]
                )
                unreferenced = sorted(set(group["source_ids"]) - referenced_by_dim[group["dim"]])
                if unreferenced:
                    unused.append({"dim": group["dim"], "attrs": group["attrs"], "source_ids": unreferenced})
            facts.sort(key=lambda row: (row["dim"], row["attr"], row["value"]))
            save(
                args.output_dir / "organized_memories.json",
                {
                    "facts": facts,
                    "unused": unused,
                    "unassigned_ids": [item["id"] for item in classifications if not item["fields"]],
                    "sources": by_id,
                },
            )
        return {"stage": args.stage, "records": len(records), "output_dir": str(args.output_dir)}
    finally:
        if runner is not None:
            runner.close()
        if embedding is not None:
            embedding.close()


def main(argv: list[str] | None = None) -> None:
    defaults = group_state_memories.default_args()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "classify", "group", "summarize", "all"), default="all"
    )
    parser.add_argument("--input-dir", type=Path, default=Path("dataset/res_csv/agent/distill"))
    parser.add_argument(
        "--preferences-only", action="store_true", help="Only organize preference profile memories"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to memory_organization_preferences for --preferences-only, otherwise memory_organization",
    )
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
    parser.add_argument("--neighbor-top-k", type=int, default=defaults.neighbor_top_k)
    parser.add_argument("--attribute-similarity", type=float, default=defaults.min_similarity)
    parser.add_argument("--embedding-url")
    parser.add_argument("--embedding-batch-size", type=int)
    parser.add_argument("--no-auto-start-embedding-service", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Count extracted memories without model calls or writes"
    )
    args = parser.parse_args(argv)
    secure_storage.configure(args.config_path)
    if args.output_dir is None:
        name = "memory_organization_preferences" if args.preferences_only else "memory_organization"
        args.output_dir = Path("dataset/res_csv/agent") / name
    for name in (
        "max_context_tokens",
        "max_records",
        "max_summary_groups",
        "max_summary_records",
        "neighbor_top_k",
        "embedding_batch_size",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.attribute_similarity <= 1:
        parser.error("--attribute-similarity must be between 0 and 1")
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
