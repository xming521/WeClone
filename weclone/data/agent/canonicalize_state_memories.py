"""Canonicalize extracted state memories into a retrieval-ready memory bank.

Workflow:
1. prepare: build one LLM task per candidate group.
2. run: call the configured LLM backend and save group merge JSONL results.
3. apply: turn merge results plus high-value singletons into canonical_memories.
Use apply --skip-merge to skip this stage without reading or converting memories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import pyjson5

from weclone.prompts.chat_distill import MERGE_PROMPT_TEMPLATE

DEFAULT_OUTPUT_DIR = Path("dataset/res_csv/agent/memory_grouping")
DEFAULT_GROUPING_PATH = DEFAULT_OUTPUT_DIR
DEFAULT_CONFIG_PATH = Path("settings.jsonc")
DEFAULT_GROUPING_SUFFIX = ".prefilter_candidates.json"
DEFAULT_TASK_SUFFIX = ".llm_merge_tasks.jsonl"
DEFAULT_RESULT_SUFFIX = ".llm_merge_results.jsonl"
DEFAULT_BANK_SUFFIX = ".canonical_memories.json"
DEFAULT_STATE_SUFFIX = ".llm_merge_state.json"
DEFAULT_SINGLETON_MIN_IMPORTANCE = 2
DEFAULT_SINGLETON_MIN_CONFIDENCE = 3
DEFAULT_INDENT = 2

VALID_MEMORY_TYPES = {"stable_fact", "preference", "goal", "current_state"}
VALID_PREFERENCE_TYPES = {"like", "dislike", "emotional_preference", "none"}
VALID_DECISIONS = {"archived", "discarded"}
VALID_TIME_SCOPES = {"long_term", "time_sensitive", "historical_only"}
VALID_STATUSES = {"active", "historical", "expired", "uncertain", "none", ""}


@dataclass
class CanonicalMemory:
    id: str
    type: str
    content: str
    tags: list[str]
    source_ids: list[str]
    evidence_count: int
    source_time_range: dict[str, str]
    last_evidence_time: str
    importance: int
    confidence: int
    preference_type: str
    status: str
    time_scope: str
    parent_id: str
    children_ids: list[str]
    origin: str
    group_id: str


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def parse_time(value: Any) -> datetime | None:
    text = normalize_text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed


def format_time(value: datetime | None) -> str:
    return value.isoformat() if value else ""


def unique_texts(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = normalize_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_save_json(path: Path, payload: Any, *, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent, default=str) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1
    tmp_path.replace(path)
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(parsed)
    return rows


def person_stem(path: Path) -> str:
    name = path.name
    if name.endswith(DEFAULT_GROUPING_SUFFIX):
        return name[: -len(DEFAULT_GROUPING_SUFFIX)]
    return path.stem


def grouping_paths_for(grouping_path: Path, *, allow_extracted: bool = False) -> list[Path]:
    if grouping_path.is_file():
        return [grouping_path]
    if grouping_path.is_dir():
        paths = sorted(path for path in grouping_path.glob(f"*{DEFAULT_GROUPING_SUFFIX}") if path.is_file())
        if not paths and allow_extracted:
            paths = sorted(path for path in grouping_path.glob("*.json") if path.is_file())
        if paths:
            return paths
        raise FileNotFoundError(
            f"No {DEFAULT_GROUPING_SUFFIX} files found in grouping directory: {grouping_path}"
        )
    raise FileNotFoundError(f"Grouping path does not exist: {grouping_path}")


def default_task_path(grouping_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{person_stem(grouping_path)}{DEFAULT_TASK_SUFFIX}"


def default_result_path(grouping_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{person_stem(grouping_path)}{DEFAULT_RESULT_SUFFIX}"


def default_bank_path(grouping_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{person_stem(grouping_path)}{DEFAULT_BANK_SUFFIX}"


def default_state_path(grouping_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{person_stem(grouping_path)}{DEFAULT_STATE_SUFFIX}"


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": normalize_text(record.get("memory_id")),
        "type": normalize_text(record.get("type")),
        "content": normalize_text(record.get("content")),
        "tags": unique_texts(record.get("tags") or []),
        "preference_type": normalize_text(record.get("preference_type")) or "none",
        "status": normalize_text(record.get("status")),
        "importance": coerce_int(record.get("importance")),
        "confidence": coerce_int(record.get("confidence")),
        "time": normalize_text(record.get("sample_time")),
        "chat_with": normalize_text(record.get("chat_with")),
        "prefilter_reasons": unique_texts(record.get("prefilter_reasons") or []),
    }


def compact_llm_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record_id(record),
        "type": normalize_text(record.get("type")),
        "content": normalize_text(record.get("content")),
        "time": normalize_text(record.get("sample_time") or record.get("time")),
    }


def record_id(record: dict[str, Any]) -> str:
    return normalize_text(record.get("memory_id") or record.get("id"))


def group_lookup(data: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    groups = {normalize_text(group.get("group_id")): group for group in data.get("candidate_groups", [])}
    member_to_group: dict[str, str] = {}
    for group_id, group in groups.items():
        for memory_id in group.get("memory_ids", []):
            member_to_group[normalize_text(memory_id)] = group_id
    return groups, member_to_group


def build_merge_task(group: dict[str, Any]) -> dict[str, Any]:
    group_id = normalize_text(group.get("group_id"))
    records = [compact_llm_record(member) for member in group.get("members", [])]
    task_input = {
        "records": records,
    }
    prompt = MERGE_PROMPT_TEMPLATE.replace(
        "{{TASK_JSON}}",
        json.dumps(task_input, ensure_ascii=False, indent=2),
    )
    return {
        "task_id": group_id,
        "group_id": group_id,
        "kind": "group_merge",
        "input": task_input,
        "prompt": prompt,
        "stats": {
            "records": len(records),
        },
    }


def prepare_tasks(args: SimpleNamespace) -> dict[str, Any]:
    grouping_path = Path(args.grouping_path)
    data = load_json(grouping_path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected grouping payload object: {grouping_path}")
    output_path = (
        Path(args.output_path)
        if args.output_path
        else default_task_path(grouping_path, Path(args.output_dir))
    )

    tasks = [build_merge_task(group) for group in data.get("candidate_groups", [])]
    count = write_jsonl(output_path, tasks)
    report = {
        "created_at": now_ts(),
        "grouping_path": str(grouping_path),
        "output_path": str(output_path),
        "tasks": count,
        "candidate_groups": len(data.get("candidate_groups", [])),
    }
    if args.report_path:
        atomic_save_json(Path(args.report_path), report, indent=args.indent)
    return report


def memory_merge_enabled(config_path: Path) -> bool:
    config = pyjson5.loads(config_path.read_text(encoding="utf-8"))
    enabled = config.get("agent_distill_args", {}).get("merge_enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("agent_distill_args.merge_enabled must be a boolean")
    return enabled


def load_codex_exec_config(config_path: Path) -> dict[str, Any]:
    config_data = pyjson5.loads(config_path.read_text(encoding="utf-8"))
    codex_config = config_data.get("codex_exec_args", {})
    if not isinstance(codex_config, dict):
        raise ValueError(f"codex_exec_args must be an object in {config_path}")
    return codex_config


def load_agent_distill_config(config_path: Path) -> dict[str, Any]:
    config_data = pyjson5.loads(config_path.read_text(encoding="utf-8"))
    config = config_data.get("agent_distill_args", {})
    if not isinstance(config, dict):
        raise ValueError(f"agent_distill_args must be an object in {config_path}")
    return config


def required_config_value(config: dict[str, Any], key: str, config_path: Path) -> Any:
    value = config.get(key)
    if value is None or value == "":
        raise ValueError(f"agent_distill_args.{key} is required in {config_path}")
    return value


def required_config_str(config: dict[str, Any], key: str, config_path: Path) -> str:
    text = str(required_config_value(config, key, config_path)).strip()
    if not text:
        raise ValueError(f"agent_distill_args.{key} is required in {config_path}")
    return text


def required_config_int(config: dict[str, Any], key: str, config_path: Path) -> int:
    value = int(required_config_value(config, key, config_path))
    if value < 1:
        raise ValueError(f"agent_distill_args.{key} must be >= 1 in {config_path}")
    return value


def resolve_llm_args(args: SimpleNamespace) -> SimpleNamespace:
    distill_config = load_agent_distill_config(args.config_path)
    args.llm_provider = args.llm_provider or distill_config.get("llm_provider") or "codex_exec"
    legacy_codex = load_codex_exec_config(args.config_path) if args.llm_provider == "codex_exec" else {}
    shared_config = {**legacy_codex, **distill_config}
    args.batch_size = args.batch_size or int(shared_config.get("batch_size", 10))
    if args.max_tokens is None:
        max_tokens = shared_config.get("max_tokens")
        args.max_tokens = None if max_tokens is None else int(max_tokens)
    args.timeout = args.timeout or int(shared_config.get("timeout", 120))
    if args.llm_provider != "codex_exec":
        return args
    args.model = args.model or required_config_str(shared_config, "model", args.config_path)
    args.effort = args.effort or required_config_str(shared_config, "effort", args.config_path)
    args.codex_command = args.codex_command or required_config_str(shared_config, "command", args.config_path)
    args.codex_sandbox = args.codex_sandbox or required_config_str(shared_config, "sandbox", args.config_path)
    return args


def load_state(state_path: Path, *, overwrite: bool) -> dict[str, Any]:
    if overwrite or not state_path.exists():
        return {"version": 1, "entries": {}, "created_at": now_ts()}
    state = load_json(state_path)
    if not isinstance(state, dict) or not isinstance(state.get("entries"), dict):
        raise ValueError(f"State must be an object with entries: {state_path}")
    return state


def response_payload(response: Any) -> dict[str, Any]:
    if hasattr(response, "__dataclass_fields__"):
        return {
            field.name: getattr(response, field.name)
            for field in response.__dataclass_fields__.values()
            if field.name != "raw"
        }
    return {"text": str(response)}


def validate_merge_result(result: Any, source_ids: set[str]) -> None:
    if not isinstance(result, dict) or set(result) != {"canonical_memories", "source_decisions"}:
        raise ValueError("Merge result must contain canonical_memories and source_decisions only")
    memories = result["canonical_memories"]
    decisions = result["source_decisions"]
    if not isinstance(memories, list) or not isinstance(decisions, list):
        raise ValueError("canonical_memories and source_decisions must be arrays")
    covered: set[str] = set()
    parents: dict[int, int] = {}
    allowed = {"content", "source_ids", "parent_index", "status", "time_scope"}
    for index, memory in enumerate(memories):
        if not isinstance(memory, dict) or set(memory) - allowed:
            raise ValueError(f"Invalid memory fields at index {index}")
        if not isinstance(memory.get("content"), str) or not memory["content"].strip():
            raise ValueError(f"Empty memory content at index {index}")
        ids = memory.get("source_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(sid, str) for sid in ids):
            raise ValueError(f"Invalid source_ids at index {index}")
        if len(ids) != len(set(ids)) or set(ids) - source_ids:
            raise ValueError(f"Duplicate or unknown source_ids at index {index}")
        covered.update(ids)
        parent = memory.get("parent_index")
        if parent is not None:
            if type(parent) is not int or not 0 <= parent < len(memories) or parent == index:
                raise ValueError(f"Invalid parent_index at index {index}")
            parents[index] = parent
        for field, values in (("status", VALID_STATUSES), ("time_scope", VALID_TIME_SCOPES | {""})):
            if field in memory and (not isinstance(memory[field], str) or memory[field] not in values):
                raise ValueError(f"Invalid {field} at index {index}")
    for index in parents:
        seen: set[int] = set()
        while index in parents:
            if index in seen:
                raise ValueError("Cyclic parent_index references")
            seen.add(index)
            index = parents[index]
    excluded: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != {"source_id", "decision", "reason"}:
            raise ValueError("Source decisions require source_id, decision and reason only")
        sid = decision["source_id"]
        if not isinstance(sid, str) or sid not in source_ids or sid in covered or sid in excluded:
            raise ValueError("Source decision must identify one unretained input source")
        if not isinstance(decision["decision"], str) or decision["decision"] not in VALID_DECISIONS:
            raise ValueError("Source decision must be archived or discarded")
        if not isinstance(decision["reason"], str) or not decision["reason"].strip():
            raise ValueError("Source decision requires a reason")
        excluded.add(sid)
    if covered | excluded != source_ids:
        raise ValueError("Merge result does not account for every input source")


def run_llm_merge(args: SimpleNamespace) -> dict[str, Any]:
    args = resolve_llm_args(args)
    tasks = read_jsonl(Path(args.tasks_path))
    if args.limit is not None:
        tasks = tasks[: args.limit]

    output_path = Path(args.output_path)
    state_path = Path(args.state_path)
    state = load_state(state_path, overwrite=args.overwrite)
    state["updated_at"] = now_ts()
    state["tasks_path"] = str(args.tasks_path)
    state["output_path"] = str(output_path)
    state["provider"] = args.llm_provider
    state["model"] = args.model
    state["effort"] = args.effort
    state["batch_size"] = args.batch_size

    entries = state["entries"]
    pending_tasks: list[dict[str, Any]] = []
    for task in tasks:
        task_id = normalize_text(task.get("task_id") or task.get("group_id"))
        if not task_id:
            continue
        existing = entries.get(task_id)
        if existing and existing.get("status") == "done" and not args.overwrite:
            continue
        pending_tasks.append(task)

    if args.dry_run:
        preview = {
            "tasks": len(tasks),
            "pending": len(pending_tasks),
            "first_prompt": normalize_text(pending_tasks[0].get("prompt")) if pending_tasks else "",
        }
        return preview

    from weclone.core.inference.llm_client import LLMRequest, build_llm_client

    request_rows: list[tuple[dict[str, Any], LLMRequest]] = []
    for task in pending_tasks:
        task_id = normalize_text(task.get("task_id") or task.get("group_id"))
        request = LLMRequest.from_prompt(
            normalize_text(task.get("prompt")),
            provider=args.llm_provider,
            model=args.model if args.llm_provider == "codex_exec" else None,
            effort=args.effort if args.llm_provider == "codex_exec" else None,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            json_mode=True,
            metadata={
                "task_id": task_id,
                "group_id": task.get("group_id"),
                "kind": task.get("kind"),
            },
        )
        request_rows.append((task, request))

    client = build_llm_client(
        args.llm_provider,
        config_path=args.config_path,
        model=args.model if args.llm_provider == "codex_exec" else None,
        max_workers=args.batch_size,
        timeout=args.timeout,
        effort=args.effort,
        command=args.codex_command,
        sandbox=args.codex_sandbox,
    )
    try:
        for start in range(0, len(request_rows), args.batch_size):
            batch = request_rows[start : start + args.batch_size]
            responses = client.generate_batch(request for _task, request in batch)
            for (task, _request), response in zip(batch, responses):
                task_id = normalize_text(task.get("task_id") or task.get("group_id"))
                error = response.error or ""
                ok = bool(response.ok)
                if ok:
                    try:
                        validate_merge_result(
                            response.parsed_json,
                            {record_id(record) for record in task["input"]["records"]},
                        )
                    except ValueError as exc:
                        ok = False
                        error = str(exc)
                row = {
                    "task_id": task_id,
                    "group_id": task.get("group_id"),
                    "ok": ok,
                    "result": response.parsed_json,
                    "response": response_payload(response),
                }
                entries[task_id] = {
                    "status": "done" if ok else "failed",
                    "updated_at": now_ts(),
                    "row": row,
                    "last_error": error,
                }
            atomic_save_json(state_path, state, indent=args.indent)
    finally:
        client.close()

    done_rows = [
        entry["row"]
        for entry in entries.values()
        if entry.get("status") == "done" and isinstance(entry.get("row"), dict)
    ]
    count = write_jsonl(output_path, sorted(done_rows, key=lambda row: normalize_text(row.get("task_id"))))
    return {
        "tasks": len(tasks),
        "written_results": count,
        "pending": len(request_rows),
        "failed": sum(
            entries.get(normalize_text(task.get("task_id") or task.get("group_id")), {}).get("status")
            == "failed"
            for task in tasks
        ),
        "output_path": str(output_path),
        "state_path": str(state_path),
    }


def extract_result(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("result", "output", "payload"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    if "canonical_memories" in row and "group_id" in row:
        return row
    raise ValueError(f"Cannot find merge result in row for task={row.get('task_id') or row.get('group_id')}")


def source_time_range(source_ids: Sequence[str], records_by_id: dict[str, dict[str, Any]]) -> dict[str, str]:
    times = [
        parsed
        for source_id in source_ids
        if (parsed := parse_time(records_by_id.get(source_id, {}).get("sample_time"))) is not None
    ]
    if not times:
        return {"first": "", "last": ""}
    return {"first": min(times).isoformat(), "last": max(times).isoformat()}


def source_tags(source_ids: Sequence[str], records_by_id: dict[str, dict[str, Any]]) -> list[str]:
    return unique_texts(
        tag for source_id in source_ids for tag in records_by_id.get(source_id, {}).get("tags") or []
    )


def source_field_values(
    source_ids: Sequence[str],
    records_by_id: dict[str, dict[str, Any]],
    field: str,
    *,
    valid_values: set[str] | None = None,
) -> list[str]:
    values: list[str] = []
    for source_id in source_ids:
        value = normalize_text(records_by_id.get(source_id, {}).get(field))
        if not value:
            continue
        if valid_values is not None and value not in valid_values:
            continue
        values.append(value)
    return values


def dominant_source_value(
    source_ids: Sequence[str],
    records_by_id: dict[str, dict[str, Any]],
    field: str,
    *,
    valid_values: set[str] | None = None,
    fallback: str,
) -> str:
    values = source_field_values(source_ids, records_by_id, field, valid_values=valid_values)
    if not values:
        return fallback
    counts = Counter(values)
    return max(counts, key=lambda value: (counts[value], -values.index(value)))


def source_memory_type(
    source_ids: Sequence[str],
    records_by_id: dict[str, dict[str, Any]],
    *,
    fallback: Any = "",
) -> str:
    fallback_type = normalize_text(fallback)
    if fallback_type not in VALID_MEMORY_TYPES:
        fallback_type = "stable_fact"
    return dominant_source_value(
        source_ids,
        records_by_id,
        "type",
        valid_values=VALID_MEMORY_TYPES,
        fallback=fallback_type,
    )


def source_preference_type(
    source_ids: Sequence[str],
    records_by_id: dict[str, dict[str, Any]],
    *,
    memory_type: str,
    fallback: Any = "",
) -> str:
    if memory_type != "preference":
        return "none"
    fallback_type = normalize_text(fallback) or "none"
    if fallback_type not in VALID_PREFERENCE_TYPES:
        fallback_type = "none"
    return dominant_source_value(
        source_ids,
        records_by_id,
        "preference_type",
        valid_values=VALID_PREFERENCE_TYPES,
        fallback=fallback_type,
    )


def normalize_status(value: Any) -> str:
    status = normalize_text(value) or "none"
    return status if status in VALID_STATUSES else "uncertain"


def normalize_time_scope(value: Any) -> str:
    time_scope = normalize_text(value)
    return time_scope if time_scope in VALID_TIME_SCOPES else ""


def stable_slug(text: str, *, fallback: str) -> str:
    ascii_text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    if ascii_text:
        return ascii_text
    normalized = normalize_text(text)
    if normalized:
        return "p" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return fallback


def build_canonical_memory(
    *,
    person_id: str,
    group_id: str,
    index: int,
    raw_memory: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
    origin: str,
) -> CanonicalMemory:
    source_ids = [
        source_id
        for source_id in unique_texts(raw_memory.get("source_ids") or [])
        if source_id in records_by_id
    ]
    memory_type = source_memory_type(source_ids, records_by_id, fallback=raw_memory.get("type"))
    preference_type = source_preference_type(
        source_ids,
        records_by_id,
        memory_type=memory_type,
        fallback=raw_memory.get("preference_type"),
    )
    source_range = source_time_range(source_ids, records_by_id)
    last_evidence_time = source_range["last"]

    status = normalize_status(raw_memory.get("status"))
    time_scope = normalize_time_scope(raw_memory.get("time_scope"))
    canonical_id = normalize_text(raw_memory.get("id"))
    if not canonical_id:
        canonical_id = f"{stable_slug(person_id, fallback='person')}:{group_id}:c{index:03d}"

    return CanonicalMemory(
        id=canonical_id,
        type=memory_type,
        content=normalize_text(raw_memory.get("content")),
        tags=source_tags(source_ids, records_by_id),
        source_ids=source_ids,
        evidence_count=len(source_ids),
        source_time_range=source_range,
        last_evidence_time=last_evidence_time,
        importance=max(coerce_int(records_by_id[sid].get("importance"), 1) for sid in source_ids),
        confidence=min(coerce_int(records_by_id[sid].get("confidence"), 2) for sid in source_ids),
        preference_type=preference_type,
        status=status,
        time_scope=time_scope,
        parent_id="",
        children_ids=[],
        origin=origin,
        group_id=group_id,
    )


def build_singleton_memory(
    *,
    person_id: str,
    record: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
) -> CanonicalMemory:
    memory_id = record_id(record)
    raw_memory = {
        "id": f"{stable_slug(person_id, fallback='person')}:singleton:{stable_slug(memory_id, fallback='source')}",
        "content": record.get("content"),
        "source_ids": [memory_id],
        "status": record.get("status") or "none",
    }
    return build_canonical_memory(
        person_id=person_id,
        group_id="singleton",
        index=0,
        raw_memory=raw_memory,
        records_by_id=records_by_id,
        origin="singleton",
    )


def record_is_high_value(record: dict[str, Any], *, min_importance: int, min_confidence: int) -> bool:
    return (
        coerce_int(record.get("importance")) >= min_importance
        and coerce_int(record.get("confidence")) >= min_confidence
    )


def apply_merge_results(args: SimpleNamespace) -> dict[str, Any]:
    grouping_path = Path(args.grouping_path)
    if getattr(args, "skip_merge", False):
        return {"skipped": True, "input_path": str(grouping_path)}
    result_path = Path(args.results_path)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else default_bank_path(grouping_path, Path(args.output_dir))
    )
    data = load_json(grouping_path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected grouping payload object: {grouping_path}")
    records_by_id = {record_id(record): record for record in data.get("records", [])}
    groups, _member_to_group = group_lookup(data)
    result_rows = read_jsonl(result_path)
    person_id = person_stem(grouping_path)

    canonical_memories: list[CanonicalMemory] = []
    source_decisions: list[dict[str, Any]] = []
    covered_source_ids: set[str] = set()
    llm_archived_source_ids: set[str] = set()
    warnings: list[str] = []

    for row in result_rows:
        result = extract_result(row)
        group_id = normalize_text(result.get("group_id") or row.get("group_id"))
        if group_id not in groups:
            warnings.append(f"Unknown group_id in result: {group_id}")
            continue
        validate_merge_result(result, {record_id(member) for member in groups[group_id]["members"]})
        raw_memories = result["canonical_memories"]
        group_start_index = len(canonical_memories) + 1
        group_memories = [
            build_canonical_memory(
                person_id=person_id,
                group_id=group_id,
                index=group_start_index + local_index,
                raw_memory=raw_memory,
                records_by_id=records_by_id,
                origin="llm_group_merge",
            )
            for local_index, raw_memory in enumerate(raw_memories)
        ]
        for local_index, raw_memory in enumerate(raw_memories):
            parent_index = raw_memory.get("parent_index")
            if parent_index is not None:
                child = group_memories[local_index]
                parent = group_memories[parent_index]
                child.parent_id = parent.id
                parent.children_ids.append(child.id)
        canonical_memories.extend(group_memories)
        for memory in group_memories:
            covered_source_ids.update(memory.source_ids)
        for decision in result["source_decisions"]:
            source_decisions.append({**decision, "group_id": group_id})
            llm_archived_source_ids.add(decision["source_id"])

    singletons = [normalize_text(memory_id) for memory_id in data.get("singletons", [])]
    singleton_kept = 0
    singleton_archived = 0
    archive_items: list[dict[str, Any]] = []
    for source_id in sorted(llm_archived_source_ids - covered_source_ids):
        record = records_by_id.get(source_id)
        if record is not None:
            archive_items.append(
                {
                    "source_id": source_id,
                    "reason": "llm_source_decision",
                    "record": compact_record(record),
                }
            )
    for memory_id in singletons:
        if memory_id in covered_source_ids:
            continue
        record = records_by_id.get(memory_id)
        if not record:
            continue
        if memory_id in llm_archived_source_ids:
            singleton_archived += 1
            continue
        if record_is_high_value(
            record,
            min_importance=args.singleton_min_importance,
            min_confidence=args.singleton_min_confidence,
        ):
            memory = build_singleton_memory(
                person_id=person_id,
                record=record,
                records_by_id=records_by_id,
            )
            canonical_memories.append(memory)
            covered_source_ids.update(memory.source_ids)
            singleton_kept += 1
        else:
            archive_items.append(
                {
                    "source_id": memory_id,
                    "reason": "low_singleton_value",
                    "record": compact_record(record),
                }
            )
            singleton_archived += 1

    for record in data.get("archive_records", []):
        archive_items.append(
            {
                "source_id": record_id(record),
                "reason": "prefilter_archive",
                "record": compact_record(record),
            }
        )

    all_candidate_ids = {
        record_id(record)
        for record in data.get("records", [])
        if normalize_text(record.get("prefilter_status")) == "candidate"
    }
    unresolved_candidate_ids = sorted(
        all_candidate_ids - covered_source_ids - {item["source_id"] for item in archive_items}
    )
    if unresolved_candidate_ids:
        warnings.append(f"Unresolved candidate source_ids: {len(unresolved_candidate_ids)}")

    payload = {
        "created_at": now_ts(),
        "input_path": str(grouping_path),
        "results_path": str(result_path),
        "person_id": person_id,
        "canonical_memories": [
            {
                field: getattr(memory, field)
                for field in (
                    "content",
                    "type",
                    "tags",
                    "importance",
                    "confidence",
                    "preference_type",
                    "status",
                    "source_time_range",
                )
            }
            for memory in canonical_memories
        ],
        "memory_provenance": [
            {
                "memory_index": index,
                **{
                    field: getattr(memory, field)
                    for field in (
                        "id",
                        "source_ids",
                        "parent_id",
                        "children_ids",
                        "origin",
                        "group_id",
                        "time_scope",
                    )
                },
            }
            for index, memory in enumerate(canonical_memories)
        ],
        "source_decisions": source_decisions,
        "archive_items": archive_items,
        "unresolved_source_ids": unresolved_candidate_ids,
        "stats": {
            "canonical_memories": len(canonical_memories),
            "source_ids_covered": len(covered_source_ids),
            "singletons_kept": singleton_kept,
            "singletons_archived": singleton_archived,
            "archive_items": len(archive_items),
            "warnings": len(warnings),
            "by_type": dict(Counter(memory.type for memory in canonical_memories).most_common()),
            "by_status": dict(Counter(memory.status for memory in canonical_memories).most_common()),
            "by_time_scope": dict(Counter(memory.time_scope for memory in canonical_memories).most_common()),
        },
        "warnings": warnings,
    }
    atomic_save_json(output_path, payload, indent=args.indent)
    return {
        "output_path": str(output_path),
        **payload["stats"],
        "unresolved_source_ids": len(unresolved_candidate_ids),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare LLM merge tasks and apply them into a canonical state memory bank."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build group-batched LLM merge tasks.")
    prepare.add_argument(
        "--grouping-path",
        type=Path,
        default=DEFAULT_GROUPING_PATH,
        help="Grouping JSON file, or a directory containing *.prefilter_candidates.json files.",
    )
    prepare.add_argument("--output-path", type=Path, default=None)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare.add_argument("--report-path", type=Path, default=None)
    prepare.add_argument("--indent", type=int, default=DEFAULT_INDENT)
    prepare.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)

    run = subparsers.add_parser("run", help="Run LLM merge tasks and save JSONL results.")
    run.add_argument(
        "--grouping-path",
        type=Path,
        default=DEFAULT_GROUPING_PATH,
        help="Grouping JSON file, or a directory containing *.prefilter_candidates.json files.",
    )
    run.add_argument("--tasks-path", type=Path, default=None)
    run.add_argument("--output-path", type=Path, default=None)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--state-path", type=Path, default=None)
    run.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    run.add_argument("--llm-provider", default=None)
    run.add_argument("--model", default=None)
    run.add_argument("--effort", default=None)
    run.add_argument("--batch-size", type=int, default=None)
    run.add_argument("--max-tokens", type=int, default=None)
    run.add_argument("--timeout", type=int, default=None)
    run.add_argument("--codex-command", default=None)
    run.add_argument("--codex-sandbox", default=None)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--indent", type=int, default=DEFAULT_INDENT)

    apply = subparsers.add_parser("apply", help="Apply LLM merge results into canonical memory bank.")
    apply.add_argument(
        "--grouping-path",
        "--input-path",
        type=Path,
        default=DEFAULT_GROUPING_PATH,
        help="Grouping JSON file/directory, or the original input to leave unchanged with --skip-merge.",
    )
    apply.add_argument("--results-path", type=Path, default=None)
    apply.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip this stage; leave the input unchanged and write no output.",
    )
    apply.add_argument("--output-path", type=Path, default=None)
    apply.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    apply.add_argument("--singleton-min-importance", type=int, default=DEFAULT_SINGLETON_MIN_IMPORTANCE)
    apply.add_argument("--singleton-min-confidence", type=int, default=DEFAULT_SINGLETON_MIN_CONFIDENCE)
    apply.add_argument("--indent", type=int, default=DEFAULT_INDENT)
    apply.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)

    return parser


def fill_default_paths(args: argparse.Namespace) -> SimpleNamespace:
    namespace = SimpleNamespace(**vars(args))
    grouping_path = Path(namespace.grouping_path)
    output_dir = Path(namespace.output_dir)
    if namespace.command == "run":
        namespace.tasks_path = (
            Path(namespace.tasks_path)
            if namespace.tasks_path
            else default_task_path(grouping_path, output_dir)
        )
        namespace.output_path = (
            Path(namespace.output_path)
            if namespace.output_path
            else default_result_path(grouping_path, output_dir)
        )
        namespace.state_path = (
            Path(namespace.state_path)
            if namespace.state_path
            else default_state_path(grouping_path, output_dir)
        )
    elif namespace.command == "apply":
        namespace.results_path = (
            Path(namespace.results_path)
            if namespace.results_path
            else default_result_path(grouping_path, output_dir)
        )
        namespace.output_path = (
            Path(namespace.output_path)
            if namespace.output_path
            else default_bank_path(grouping_path, output_dir)
        )
    return namespace


def single_file_only_path_fields(command: str) -> tuple[str, ...]:
    if command == "prepare":
        return ("output_path", "report_path")
    if command == "run":
        return ("tasks_path", "output_path", "state_path")
    if command == "apply":
        return ("results_path", "output_path")
    return ()


def validate_directory_mode_args(args: argparse.Namespace, requested_grouping_path: Path) -> None:
    if not requested_grouping_path.is_dir():
        return
    for field in single_file_only_path_fields(args.command):
        if getattr(args, field, None):
            option = "--" + field.replace("_", "-")
            raise ValueError(
                f"{option} can only be used when --grouping-path points to one grouping JSON file."
            )


def run_command(args: SimpleNamespace) -> dict[str, Any]:
    if args.command == "prepare":
        return prepare_tasks(args)
    if args.command == "run":
        return run_llm_merge(args)
    if args.command == "apply":
        return apply_merge_results(args)
    raise ValueError(f"Unknown command: {args.command}")


def print_command_report(args: SimpleNamespace, report: dict[str, Any]) -> None:
    if args.command == "prepare":
        print(
            f"完成: input={args.grouping_path} tasks={report['tasks']} output={report['output_path']}",
            flush=True,
        )
        return
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    parsed_args = parser.parse_args(argv)
    try:
        if getattr(parsed_args, "skip_merge", False) or not memory_merge_enabled(parsed_args.config_path):
            print(
                json.dumps({"skipped": True, "command": parsed_args.command}, ensure_ascii=False), flush=True
            )
            return
        requested_grouping_path = Path(parsed_args.grouping_path)
        validate_directory_mode_args(parsed_args, requested_grouping_path)
        grouping_paths = grouping_paths_for(
            requested_grouping_path,
            allow_extracted=getattr(parsed_args, "skip_merge", False),
        )
        if getattr(parsed_args, "skip_merge", False) and parsed_args.results_path:
            raise ValueError("--skip-merge cannot be combined with --results-path")
        if parsed_args.command == "run":
            for field in ("batch_size", "max_tokens", "timeout", "limit"):
                value = getattr(parsed_args, field)
                if value is not None and value < 1:
                    raise ValueError(f"--{field.replace('_', '-')} must be >= 1")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    failed = False
    for grouping_path in grouping_paths:
        file_args = argparse.Namespace(**vars(parsed_args))
        file_args.grouping_path = grouping_path
        args = fill_default_paths(file_args)
        report = run_command(args)
        print_command_report(args, report)
        if args.command == "run" and report.get("failed", 0):
            failed = True

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
