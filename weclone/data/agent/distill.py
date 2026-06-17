import json
import threading
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import pyjson5
from tqdm import tqdm

from weclone.prompts.chat_distill import STATE_EXTRACT_PROMPT
from weclone.utils.log import logger

DEFAULT_INPUT_DIR = Path("dataset/res_csv/agent/people")
DEFAULT_OUTPUT_DIR = Path("dataset/res_csv/agent/distill")
DEFAULT_STATE_PATH = None
DEFAULT_CONFIG_PATH = Path("settings.jsonc")
DEFAULT_TARGET_ROLE = "assistant"
DEFAULT_PROVIDER = "codex_exec"
DEFAULT_MODEL = None
DEFAULT_EFFORT = "low"
DEFAULT_BATCH_SIZE = 1
DEFAULT_CODEX_COMMAND = "codex"
DEFAULT_CODEX_SANDBOX = "read-only"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = None
DEFAULT_LIMIT_FILES = None
DEFAULT_LIMIT_RECORDS = None
DEFAULT_OVERWRITE = False
DEFAULT_DRY_RUN = False
DEFAULT_INDENT = 2

PROMPT_PLACEHOLDER = "{{CHAT_JSON}}"
TERMINAL_STATUSES = {"done", "failed"}
WRITEBACK_FIELD = "state_memories"
_print_lock = threading.Lock()


def default_args() -> SimpleNamespace:
    return SimpleNamespace(
        input_dir=DEFAULT_INPUT_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        state_path=DEFAULT_STATE_PATH,
        config_path=DEFAULT_CONFIG_PATH,
        target_role=DEFAULT_TARGET_ROLE,
        llm_provider=None,
        model=DEFAULT_MODEL,
        effort=None,
        batch_size=None,
        codex_command=None,
        codex_sandbox=None,
        max_tokens=DEFAULT_MAX_TOKENS,
        timeout=DEFAULT_TIMEOUT,
        limit_files=DEFAULT_LIMIT_FILES,
        limit_records=DEFAULT_LIMIT_RECORDS,
        overwrite=DEFAULT_OVERWRITE,
        dry_run=DEFAULT_DRY_RUN,
        indent=DEFAULT_INDENT,
    )


def log(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_codex_exec_config(config_path: Path) -> dict[str, Any]:
    config_data = pyjson5.loads(config_path.read_text(encoding="utf-8"))
    codex_config = config_data.get("codex_exec_args", {})
    if not isinstance(codex_config, dict):
        raise ValueError(f"codex_exec_args must be an object in {config_path}")
    return codex_config


def optional_str(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def optional_int(*values: Any) -> int | None:
    for value in values:
        if value is None or value == "":
            continue
        return int(value)
    return None


def resolve_llm_args(args: SimpleNamespace) -> SimpleNamespace:
    args.llm_provider = args.llm_provider or DEFAULT_PROVIDER
    if args.llm_provider != "codex_exec":
        args.model = None
        args.batch_size = max(1, optional_int(args.batch_size, DEFAULT_BATCH_SIZE) or 1)
        return args

    codex_config = load_codex_exec_config(args.config_path)
    args.model = optional_str(args.model, codex_config.get("model"), DEFAULT_MODEL)
    if args.model is None:
        raise ValueError(
            f"codex_exec model is required. Set codex_exec_args.model in {args.config_path} "
            "or pass --model."
        )
    args.effort = optional_str(args.effort, codex_config.get("effort"), DEFAULT_EFFORT)
    args.batch_size = max(
        1,
        optional_int(args.batch_size, codex_config.get("batch_size"), DEFAULT_BATCH_SIZE) or 1,
    )
    args.timeout = optional_int(args.timeout, codex_config.get("timeout"), DEFAULT_TIMEOUT)
    args.codex_command = optional_str(
        args.codex_command,
        codex_config.get("command"),
        DEFAULT_CODEX_COMMAND,
    )
    args.codex_sandbox = optional_str(
        args.codex_sandbox,
        codex_config.get("sandbox"),
        DEFAULT_CODEX_SANDBOX,
    )
    return args


def atomic_save_json(path: Path, payload: dict[str, Any], *, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent, default=str) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def atomic_save_any_json(path: Path, payload: Any, *, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent, default=str) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def iter_chat_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.glob("*.json")):
        if "manifest" in path.name:
            continue
        yield path


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    size = max(1, int(batch_size or 1))
    for start in range(0, len(items), size):
        yield items[start : start + size]


def chat_items(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict) and isinstance(item.get("messages"), list)]


def default_state_path(output_dir: Path) -> Path:
    return output_dir / "distill_state.json"


def new_state(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "version": 1,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "entries": {},
    }


def load_state(
    state_path: Path,
    *,
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if overwrite or not state_path.exists():
        return new_state(input_dir, output_dir)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or not isinstance(state.get("entries"), dict):
        raise ValueError(f"State must be an object with entries: {state_path}")
    state["input_dir"] = str(input_dir)
    state["output_dir"] = str(output_dir)
    return state


def state_key(source_path: Path, sample_id: str) -> str:
    return f"{source_path}::{sample_id}"


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def is_terminal(record: Any) -> bool:
    return isinstance(record, dict) and str(record.get("status") or "") in TERMINAL_STATUSES


def make_record(
    source_path: Path,
    item: dict[str, Any],
    *,
    source_index: int,
    sample_id: str,
) -> dict[str, Any]:
    return {
        "source_file": str(source_path),
        "source_index": source_index,
        "sample_id": sample_id,
        "sample_time": item.get("time", ""),
        "chat_with": item.get("chat_with", ""),
        "status": "pending",
        "updated_at": now_ts(),
        "last_error": "",
    }


def memories_from_payload(payload: dict[str, Any]) -> list[Any] | None:
    result = payload.get("result")
    if isinstance(result, dict):
        memories = result.get("memories")
        if isinstance(memories, list):
            return memories
    return None


def apply_payload_to_item(item: dict[str, Any], payload: dict[str, Any]) -> bool:
    memories = memories_from_payload(payload)
    if memories is None:
        return False
    if item.get(WRITEBACK_FIELD) == memories:
        return False
    item[WRITEBACK_FIELD] = memories
    return True


def sample_id_for(item: dict[str, Any], source_index: int) -> str:
    for key in ("id", "local_id"):
        value = item.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return str(source_index)


def role_label(role: str, target_role: str) -> str | None:
    if role == target_role:
        return "B"
    if role in {"user", "assistant"}:
        return "A"
    return None


def other_role(target_role: str) -> str:
    return "assistant" if target_role == "user" else "user"


def render_chat(item: dict[str, Any], *, target_role: str, sample_id: str) -> str:
    a_role = other_role(target_role)
    lines = [
        f"sample_id: {sample_id}",
        f"sample_time: {item.get('time', '')}",
        f"chat_with: {item.get('chat_with', '')}",
        f"role_mapping: A={a_role}, B={target_role}",
        "说明：只抽取 B 的信息；A 的内容只作为上下文证据。",
        "messages:",
    ]

    for message_index, message in enumerate(item.get("messages", [])):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip()
        label = role_label(role, target_role)
        if label is None:
            continue
        content = str(message.get("content", "")).replace("\r\n", "\n").strip()
        if not content:
            continue
        lines.append(f"[{message_index}] {label}：{content}")
    return "\n".join(lines)


def build_prompt(item: dict[str, Any], *, target_role: str, sample_id: str) -> str:
    rendered_chat = render_chat(item, target_role=target_role, sample_id=sample_id)
    return STATE_EXTRACT_PROMPT.replace(PROMPT_PLACEHOLDER, rendered_chat)


def response_payload(response: Any) -> dict[str, Any]:
    if not is_dataclass(response):
        return {"text": str(response)}
    return {
        field.name: getattr(response, field.name)
        for field in fields(response)
        if field.name != "raw"
    }


def run_dry_preview(source_path: Path, item: dict[str, Any], *, target_role: str, sample_id: str) -> None:
    rendered_chat = render_chat(item, target_role=target_role, sample_id=sample_id)
    prompt = STATE_EXTRACT_PROMPT.replace(PROMPT_PLACEHOLDER, rendered_chat)
    preview = rendered_chat[:2000]
    suffix = "" if len(rendered_chat) <= len(preview) else "\n...<truncated>"
    logger.info(
        f"Dry run: {source_path.name} sample_id={sample_id}, "
        f"chat_chars={len(rendered_chat)}, prompt_chars={len(prompt)}"
    )
    print(preview + suffix)


def process_file(
    source_path: Path,
    *,
    output_dir: Path,
    target_role: str,
    provider: str,
    model: str | None,
    effort: str | None,
    client: Any,
    max_tokens: int,
    batch_size: int,
    limit_records: int | None,
    overwrite: bool,
    dry_run: bool,
    state: dict[str, Any],
    state_path: Path,
    indent: int,
    progress_factory: Any = tqdm,
) -> tuple[int, int]:
    source_data = load_json(source_path)
    all_items = chat_items(source_data)
    items = all_items
    if limit_records is not None:
        items = items[:limit_records]
    if not items:
        logger.warning(f"Skip {source_path}: no chat records found")
        return 0, 0

    entries = state.setdefault("entries", {})
    done_count = 0
    call_count = 0
    request_rows = []
    skipped_count = 0
    writeback_changed = False

    for source_index, item in enumerate(items):
        sample_id = sample_id_for(item, source_index)
        key = state_key(source_path, sample_id)
        record = entries.get(key)
        if not isinstance(record, dict):
            record = make_record(source_path, item, source_index=source_index, sample_id=sample_id)
            entries[key] = record

        if is_terminal(record):
            payload = record.get("payload")
            if isinstance(payload, dict) and not dry_run:
                writeback_changed = apply_payload_to_item(item, payload) or writeback_changed
            skipped_count += 1
            continue

        if not overwrite and isinstance(item.get(WRITEBACK_FIELD), list):
            record["status"] = "done"
            record["done_reason"] = "input_writeback"
            record["updated_at"] = now_ts()
            skipped_count += 1
            continue

        if dry_run:
            run_dry_preview(source_path, item, target_role=target_role, sample_id=sample_id)
            return 1, 0

        from weclone.core.inference.llm_client import LLMRequest

        prompt = build_prompt(item, target_role=target_role, sample_id=sample_id)
        request = LLMRequest.from_prompt(
            prompt,
            provider=provider,
            model=model,
            effort=effort,
            max_tokens=max_tokens,
            json_mode=True,
            metadata={
                "source_file": str(source_path),
                "source_index": source_index,
                "sample_id": sample_id,
            },
        )
        request_rows.append((key, source_index, item, sample_id, request))

    log(
        f"{source_path.name}: total={len(items)} pending={len(request_rows)} "
        f"skipped={skipped_count} batch_size={batch_size}"
    )
    if writeback_changed and not dry_run:
        atomic_save_any_json(source_path, source_data, indent=indent)
        writeback_changed = False
    if not dry_run:
        atomic_save_json(state_path, state, indent=indent)

    progress = progress_factory(
        total=len(items),
        initial=skipped_count,
        desc=source_path.name,
        unit="sample",
    )
    try:
        for batch in batched(request_rows, batch_size):
            responses = client.generate_batch(row[4] for row in batch)
            call_count += len(batch)

            for key, source_index, item, sample_id, _request in batch:
                record = entries[key]
                response = responses.pop(0) if responses else None
                if response is None:
                    record["status"] = "failed"
                    record["last_error"] = "missing response"
                    record["updated_at"] = now_ts()
                    continue

                payload = {
                    "source_file": str(source_path),
                    "source_index": source_index,
                    "sample_id": sample_id,
                    "sample_time": item.get("time", ""),
                    "chat_with": item.get("chat_with", ""),
                    "target_role": target_role,
                    "role_mapping": {"A": other_role(target_role), "B": target_role},
                    "result": response.parsed_json,
                    "response": response_payload(response),
                }
                if response.ok:
                    writeback_changed = apply_payload_to_item(item, payload) or writeback_changed
                record["status"] = "done" if response.ok else "failed"
                record["payload"] = payload
                record["response_ok"] = response.ok
                record["last_error"] = response.error or ""
                record["updated_at"] = now_ts()
                done_count += 1

                if not response.ok:
                    logger.warning(f"LLM failed for {source_path.name} sample_id={sample_id}: {response.error}")
                    log(f"{source_path.name}: sample_id={sample_id} FAILED {str(response.error)[:160]}")

            progress.update(len(batch))
            if writeback_changed:
                atomic_save_any_json(source_path, source_data, indent=indent)
                writeback_changed = False
            atomic_save_json(state_path, state, indent=indent)
            log(f"{source_path.name}: wrote={done_count} calls={call_count}")
    finally:
        progress.close()

    if writeback_changed and not dry_run:
        atomic_save_any_json(source_path, source_data, indent=indent)

    return done_count, call_count


def main() -> None:
    args = resolve_llm_args(default_args())
    request_model = args.model if args.llm_provider == "codex_exec" else None
    request_effort = args.effort if args.llm_provider == "codex_exec" else None
    source_files = list(iter_chat_files(args.input_dir))
    if args.limit_files is not None:
        source_files = source_files[: args.limit_files]

    if not source_files:
        raise FileNotFoundError(f"No chat JSON files found in {args.input_dir}")

    state_path = Path(args.state_path) if args.state_path else default_state_path(args.output_dir)
    state = load_state(
        state_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    state["provider"] = args.llm_provider
    state["model"] = request_model
    state["effort"] = request_effort
    state["batch_size"] = args.batch_size
    state["updated_at"] = now_ts()

    log(f"输入目录: {args.input_dir}  文件数: {len(source_files)}")
    log(f"输出目录: {args.output_dir}")
    log(f"断点文件: {state_path}")
    log(
        f"provider={args.llm_provider} model={request_model} "
        f"effort={request_effort} batch_size={args.batch_size} dry_run={args.dry_run}"
    )

    client = None
    if not args.dry_run:
        from weclone.core.inference.llm_client import build_llm_client

        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_save_json(state_path, state, indent=args.indent)
        client = build_llm_client(
            args.llm_provider,
            config_path=args.config_path,
            model=request_model,
            max_workers=args.batch_size,
            timeout=args.timeout,
            effort=args.effort,
            command=args.codex_command,
            sandbox=args.codex_sandbox,
        )

    total_done = 0
    total_calls = 0
    try:
        for source_path in source_files:
            done_count, call_count = process_file(
                source_path,
                output_dir=args.output_dir,
                target_role=args.target_role,
                provider=args.llm_provider,
                model=request_model,
                effort=request_effort,
                client=client,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                limit_records=args.limit_records,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                state=state,
                state_path=state_path,
                indent=args.indent,
            )
            total_done += done_count
            total_calls += call_count
            if not args.dry_run:
                atomic_save_json(state_path, state, indent=args.indent)
            logger.info(f"Processed {source_path.name}: wrote={done_count}, calls={call_count}")
    finally:
        if client is not None:
            client.close()

    if not args.dry_run:
        state["updated_at"] = now_ts()
        atomic_save_json(state_path, state, indent=args.indent)
    logger.info(f"Done. wrote={total_done}, llm_calls={total_calls}, dry_run={args.dry_run}")
    log(f"完成: wrote={total_done} llm_calls={total_calls} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
