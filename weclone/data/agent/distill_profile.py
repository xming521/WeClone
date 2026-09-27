import json
import threading
import time
from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import click
import pyjson5
from tqdm import tqdm

from weclone.data.agent.distill_windows import (
    ChatSample,
    generate_window_batch,
    group_samples,
    make_window_request,
    render_sample,
)
from weclone.prompts.chat_distill import build_state_extract_prompt
from weclone.utils.log import logger

DEFAULT_INPUT_DIR = Path("dataset/res_csv/agent/people")
DEFAULT_OUTPUT_DIR = Path("dataset/res_csv/agent/distill")
DEFAULT_STATE_PATH = None
DEFAULT_CONFIG_PATH = Path("settings.jsonc")
DEFAULT_TARGET_ROLE = "assistant"
DEFAULT_PROVIDER = "codex_exec"
DEFAULT_OVERWRITE = True
STATE_WRITEBACK_FIELD = "state_memories"

DEFAULT_LIMIT_FILES = None
DEFAULT_LIMIT_RECORDS = None
_print_lock = threading.Lock()


def default_args() -> SimpleNamespace:
    return SimpleNamespace(
        input_dir=DEFAULT_INPUT_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        state_path=DEFAULT_STATE_PATH,
        config_path=DEFAULT_CONFIG_PATH,
        target_role=DEFAULT_TARGET_ROLE,
        llm_provider=None,
        model=None,
        effort=None,
        batch_size=None,
        max_samples_per_window=8,
        max_content_chars=400,
        codex_command=None,
        codex_sandbox=None,
        max_tokens=None,
        timeout=None,
        limit_files=DEFAULT_LIMIT_FILES,
        limit_records=DEFAULT_LIMIT_RECORDS,
        overwrite=DEFAULT_OVERWRITE,
        dry_run=False,
        indent=2,
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


def load_agent_distill_config(config_path: Path) -> dict[str, Any]:
    config_data = pyjson5.loads(config_path.read_text(encoding="utf-8"))
    distill_config = config_data.get("agent_distill_args", {})
    if distill_config is None:
        return {}
    if not isinstance(distill_config, dict):
        raise ValueError(f"agent_distill_args must be an object in {config_path}")
    return distill_config


def normalize_llm_provider(provider: Any) -> str:
    text = str(provider or DEFAULT_PROVIDER).strip().lower().replace("-", "_")
    if text in {"codex", "codex_exec"}:
        return "codex_exec"
    if text in {"api", "openai", "deepseek", "openrouter", "openai_compatible"}:
        return "api"
    raise ValueError(f"unknown agent_distill_args.llm_provider: {provider}")


def required_config_value(config: dict[str, Any], key: str, config_path: Path) -> Any:
    value = config.get(key)
    if value is None or value == "":
        raise ValueError(f"codex_exec_args.{key} is required in {config_path}")
    return value


def required_config_str(config: dict[str, Any], key: str, config_path: Path) -> str:
    text = str(required_config_value(config, key, config_path)).strip()
    if not text:
        raise ValueError(f"codex_exec_args.{key} is required in {config_path}")
    return text


def required_config_int(config: dict[str, Any], key: str, config_path: Path) -> int:
    value = int(required_config_value(config, key, config_path))
    if value < 1:
        raise ValueError(f"codex_exec_args.{key} must be >= 1 in {config_path}")
    return value


def resolve_llm_args(args: SimpleNamespace) -> SimpleNamespace:
    distill_config = load_agent_distill_config(args.config_path)
    args.overwrite = distill_config.get("overwrite", args.overwrite)
    if type(args.overwrite) is not bool:
        raise ValueError("agent_distill_args.overwrite must be a boolean")
    args.llm_provider = normalize_llm_provider(args.llm_provider or distill_config.get("llm_provider"))
    codex_config = load_codex_exec_config(args.config_path)
    args.batch_size = required_config_int(codex_config, "batch_size", args.config_path)
    args.max_tokens = (
        required_config_int(codex_config, "max_tokens", args.config_path)
        if codex_config.get("max_tokens") is not None
        else None
    )
    args.timeout = required_config_int(codex_config, "timeout", args.config_path)
    for name in ("max_samples_per_window", "max_content_chars"):
        value = distill_config.get(name, getattr(args, name))
        if type(value) is not int or value < 1:
            raise ValueError(f"agent_distill_args.{name} must be a positive integer")
        setattr(args, name, value)

    if args.llm_provider != "codex_exec":
        args.model = None
        return args

    args.model = required_config_str(codex_config, "model", args.config_path)
    args.effort = required_config_str(codex_config, "effort", args.config_path)
    args.codex_command = required_config_str(codex_config, "command", args.config_path)
    args.codex_sandbox = required_config_str(codex_config, "sandbox", args.config_path)
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


def confirm_distillation(args: SimpleNamespace, *, task: str, file_count: int) -> bool:
    click.echo(
        f"即将执行{task}蒸馏：输入 {args.input_dir}（{file_count} 个聊天文件），输出 {args.output_dir}。"
    )
    click.echo(f"模型后端：{args.llm_provider}。输入目录中的聊天样本将交给该模型处理。")
    click.echo("聊天内容可能包含身份、联系方式和私密对话；模型服务可能记录或留存这些内容，存在隐私泄露风险。")
    if args.overwrite:
        click.echo("当前配置 overwrite=true，将重新抽取并更新同名结果和断点。")
    click.echo("请确认你有权处理这些聊天记录，并同意将聊天内容发送给该模型服务。")
    try:
        return click.confirm("继续执行？", default=False)
    except click.Abort:
        return False


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    size = max(1, int(batch_size or 1))
    for start in range(0, len(items), size):
        yield items[start : start + size]


def chat_items(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict) and isinstance(item.get("messages"), list)]


def parse_sample_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def current_state_time_window(items: list[dict[str, Any]]) -> tuple[datetime | None, datetime | None]:
    times = [parsed for item in items if (parsed := parse_sample_time(item.get("time"))) is not None]
    if not times:
        return None, None
    latest_time = max(times)
    return latest_time, latest_time - timedelta(days=14)


def allow_current_state(item: dict[str, Any], cutoff_time: datetime | None) -> bool:
    sample_time = parse_sample_time(item.get("time"))
    return cutoff_time is not None and sample_time is not None and sample_time >= cutoff_time


def default_state_path(output_dir: Path) -> Path:
    return output_dir / "distill_state_checkpoint.json"


def output_path_for(output_dir: Path, source_path: Path) -> Path:
    return output_dir / "state_people" / source_path.name


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
    return isinstance(record, dict) and str(record.get("status") or "") in {"done", "failed"}


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


def filter_current_state_payload(payload: dict[str, Any], *, include_current_state: bool) -> dict[str, Any]:
    if include_current_state:
        return payload

    memories = memories_from_payload(payload)
    if memories is None:
        return payload

    filtered_memories = [
        memory
        for memory in memories
        if not (isinstance(memory, dict) and memory.get("type") == "current_state")
    ]
    if len(filtered_memories) == len(memories):
        return payload

    filtered_payload = dict(payload)
    result = dict(payload.get("result") or {})
    result["memories"] = filtered_memories
    filtered_payload["result"] = result
    return filtered_payload


def apply_payload_to_item(item: dict[str, Any], payload: dict[str, Any]) -> bool:
    memories = memories_from_payload(payload)
    if memories is None:
        return False
    if item.get(STATE_WRITEBACK_FIELD) == memories:
        return False
    item[STATE_WRITEBACK_FIELD] = memories
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
    return render_sample(item, sample_id=sample_id, target_role=target_role, include_time=False)


def build_prompt(
    item: dict[str, Any],
    *,
    target_role: str,
    sample_id: str,
    include_current_state: bool = True,
) -> str:
    rendered_chat = render_chat(item, target_role=target_role, sample_id=sample_id)
    prompt = build_state_extract_prompt(include_current_state=include_current_state)
    return prompt.replace("{{CHAT_JSON}}", rendered_chat)


def response_payload(response: Any) -> dict[str, Any]:
    if not is_dataclass(response):
        return {"text": str(response)}
    return {field.name: getattr(response, field.name) for field in fields(response) if field.name != "raw"}


def process_file(
    source_path: Path,
    *,
    output_dir: Path,
    target_role: str,
    provider: str,
    model: str | None,
    effort: str | None,
    client: Any,
    max_tokens: int | None,
    batch_size: int,
    limit_records: int | None,
    overwrite: bool,
    dry_run: bool,
    state: dict[str, Any],
    state_path: Path,
    indent: int,
    progress_factory: Any = tqdm,
    max_samples_per_window: int = 8,
    max_content_chars: int = 400,
) -> tuple[int, int]:
    source_data = load_json(source_path)
    output_path = output_path_for(output_dir, source_path)
    all_items = chat_items(source_data)
    latest_time, current_state_cutoff_time = current_state_time_window(all_items)
    items = all_items
    if limit_records is not None:
        items = items[:limit_records]
    if not items:
        logger.warning(f"Skip {source_path}: no chat records found")
        return 0, 0

    entries = state.setdefault("entries", {})
    done_count = 0
    call_count = 0
    pending_samples = []
    skipped_count = 0
    writeback_changed = False

    for source_index, item in enumerate(items):
        sample_id = sample_id_for(item, source_index)
        include_current_state = allow_current_state(item, current_state_cutoff_time)
        key = state_key(source_path, sample_id)
        record = entries.get(key)
        if not isinstance(record, dict):
            record = make_record(source_path, item, source_index=source_index, sample_id=sample_id)
            entries[key] = record

        if is_terminal(record):
            payload = record.get("payload")
            if isinstance(payload, dict) and not dry_run:
                payload = filter_current_state_payload(payload, include_current_state=include_current_state)
                record["payload"] = payload
                writeback_changed = apply_payload_to_item(item, payload) or writeback_changed
            skipped_count += 1
            continue

        if not overwrite and isinstance(item.get(STATE_WRITEBACK_FIELD), list):
            record["status"] = "done"
            record["done_reason"] = "input_writeback"
            record["updated_at"] = now_ts()
            skipped_count += 1
            continue

        pending_samples.append(ChatSample(source_index, sample_id, item, include_current_state))

    windows = group_samples(
        pending_samples,
        max_samples=max_samples_per_window,
        max_content_chars=max_content_chars,
    )
    request_rows = [
        (
            window,
            make_window_request(
                window,
                task="state",
                source_path=source_path,
                target_role=target_role,
                provider=provider,
                model=model,
                effort=effort,
                max_tokens=max_tokens,
            ),
        )
        for window in windows
    ]
    if dry_run:
        if request_rows:
            window, request = request_rows[0]
            logger.info(f"Dry run: {source_path.name}, samples={[s.sample_id for s in window]}")
            print(request.messages[0]["content"])
            return len(window), 0
        return 0, 0

    log(
        f"{source_path.name}: total={len(items)} pending={len(pending_samples)} windows={len(request_rows)} "
        f"skipped={skipped_count} batch_size={batch_size} "
        f"latest_time={latest_time.isoformat() if latest_time else ''} "
        f"current_state_since={current_state_cutoff_time.isoformat() if current_state_cutoff_time else ''} "
        f"output={output_path}"
    )
    if not dry_run and (writeback_changed or (skipped_count and not output_path.exists())):
        atomic_save_any_json(output_path, source_data, indent=indent)
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
            outcomes = generate_window_batch(client, batch, task="state")
            for outcome in outcomes:
                call_count += len(outcome.attempts)
                window_key = f"{source_path}::" + ",".join(str(s.source_index) for s in outcome.samples)
                state.setdefault("windows", {})[window_key] = {
                    "sample_ids": [s.sample_id for s in outcome.samples],
                    "attempts": [response_payload(response) for response in outcome.attempts],
                    "last_error": outcome.error,
                }
                for sample in outcome.samples:
                    item, sample_id = sample.item, sample.sample_id
                    record = entries[state_key(source_path, sample_id)]
                    payload = {
                        "source_file": str(source_path),
                        "source_index": sample.source_index,
                        "sample_id": sample_id,
                        "sample_time": item.get("time", ""),
                        "chat_with": item.get("chat_with", ""),
                        "target_role": target_role,
                        "role_mapping": {"A": other_role(target_role), "B": target_role},
                        "include_current_state": sample.include_current_state,
                        "current_state_window_days": 14,
                        "latest_sample_time": latest_time.isoformat() if latest_time else "",
                        "current_state_since": current_state_cutoff_time.isoformat()
                        if current_state_cutoff_time
                        else "",
                        "result": outcome.results.get(sample_id),
                        "response": {
                            "ok": not outcome.error,
                            "error": outcome.error,
                            "window_key": window_key,
                        },
                    }
                    if not outcome.error:
                        writeback_changed = apply_payload_to_item(item, payload) or writeback_changed
                    record["status"] = "failed" if outcome.error else "done"
                    record["payload"] = payload
                    record["response_ok"] = not outcome.error
                    record["last_error"] = outcome.error
                    record["updated_at"] = now_ts()
                    done_count += 1
                if outcome.error:
                    logger.warning(f"LLM window failed for {source_path.name}: {outcome.error}")
                progress.update(len(outcome.samples))

            if writeback_changed:
                atomic_save_any_json(output_path, source_data, indent=indent)
                writeback_changed = False
            atomic_save_json(state_path, state, indent=indent)
            log(f"{source_path.name}: wrote={done_count} calls={call_count}")
    finally:
        progress.close()

    if writeback_changed and not dry_run:
        atomic_save_any_json(output_path, source_data, indent=indent)

    return done_count, call_count


def main(
    *, input_dir: Path | None = None, output_dir: Path | None = None, config_path: Path | None = None
) -> None:
    args = default_args()
    if input_dir is not None:
        args.input_dir = input_dir
    if output_dir is not None:
        args.output_dir = output_dir
    if config_path is not None:
        args.config_path = config_path
    args = resolve_llm_args(args)
    request_model = args.model if args.llm_provider == "codex_exec" else None
    request_effort = args.effort if args.llm_provider == "codex_exec" else None
    source_files = list(iter_chat_files(args.input_dir))
    if args.limit_files is not None:
        source_files = source_files[: args.limit_files]

    if not source_files:
        raise FileNotFoundError(f"No chat JSON files found in {args.input_dir}")
    if not confirm_distillation(args, task="用户画像", file_count=len(source_files)):
        raise SystemExit("已取消蒸馏；未写入结果或调用模型。")

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
    state["max_samples_per_window"] = args.max_samples_per_window
    state["max_content_chars"] = args.max_content_chars
    state["writeback_field"] = STATE_WRITEBACK_FIELD
    state["output_subdir"] = "state_people"
    state["updated_at"] = now_ts()

    log(f"输入目录: {args.input_dir}  文件数: {len(source_files)}")
    log(f"输出目录: {args.output_dir}")
    log(f"断点文件: {state_path}")
    log(f"画像字段: {STATE_WRITEBACK_FIELD}")
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
                max_samples_per_window=args.max_samples_per_window,
                max_content_chars=args.max_content_chars,
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
