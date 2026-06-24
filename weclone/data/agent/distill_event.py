from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tqdm import tqdm

from weclone.data.agent.distill_state import (
    atomic_save_any_json,
    atomic_save_json,
    batched,
    chat_items,
    default_args as state_default_args,
    iter_chat_files,
    load_json,
    load_state,
    log,
    make_record,
    now_ts,
    other_role,
    render_chat,
    resolve_llm_args,
    response_payload,
    sample_id_for,
    state_key,
)
from weclone.prompts.chat_distill import EVENT_EXTRACT_PROMPT
from weclone.utils.log import logger

EVENT_WRITEBACK_FIELD = "event_memories"


def default_args() -> SimpleNamespace:
    args = state_default_args()
    args.state_path = None
    return args


def default_event_state_path(output_dir: Path) -> Path:
    return output_dir / "distill_event_checkpoint.json"


def output_path_for(output_dir: Path, source_path: Path) -> Path:
    return output_dir / "event_people" / source_path.name


def event_result_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    result = payload.get("result")
    if not isinstance(result, dict):
        return None

    event_result: dict[str, Any] = {}
    saw_event_array = False
    for key in ("surface_events", "inferred_events"):
        value = result.get(key)
        if isinstance(value, list):
            saw_event_array = True
            if value:
                event_result[key] = value

    if event_result or saw_event_array or not result:
        return event_result
    return None


def apply_payload_to_item(item: dict[str, Any], payload: dict[str, Any]) -> bool:
    event_result = event_result_from_payload(payload)
    if event_result is None:
        return False
    if item.get(EVENT_WRITEBACK_FIELD) == event_result:
        return False
    item[EVENT_WRITEBACK_FIELD] = event_result
    return True


def render_event_chat(item: dict[str, Any], *, target_role: str, sample_id: str) -> str:
    rendered_chat = render_chat(item, target_role=target_role, sample_id=sample_id)
    sample_time = str(item.get("time") or "").strip()
    if not sample_time:
        return rendered_chat
    return f"sample_time: {sample_time}\n{rendered_chat}"


def build_prompt(item: dict[str, Any], *, target_role: str, sample_id: str) -> str:
    rendered_chat = render_event_chat(item, target_role=target_role, sample_id=sample_id)
    return EVENT_EXTRACT_PROMPT.replace("{{CHAT_JSON}}", rendered_chat)


def run_dry_preview(
    source_path: Path,
    item: dict[str, Any],
    *,
    target_role: str,
    sample_id: str,
) -> None:
    rendered_chat = render_event_chat(item, target_role=target_role, sample_id=sample_id)
    prompt = EVENT_EXTRACT_PROMPT.replace("{{CHAT_JSON}}", rendered_chat)
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
    output_path = output_path_for(output_dir, source_path)
    all_items = chat_items(source_data)
    items = all_items[:limit_records] if limit_records is not None else all_items
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

        if isinstance(record, dict) and str(record.get("status") or "") in {"done", "failed"}:
            payload = record.get("payload")
            if isinstance(payload, dict) and not dry_run:
                writeback_changed = apply_payload_to_item(item, payload) or writeback_changed
            skipped_count += 1
            continue

        if not overwrite and isinstance(item.get(EVENT_WRITEBACK_FIELD), dict):
            record["status"] = "done"
            record["done_reason"] = "input_writeback"
            record["updated_at"] = now_ts()
            skipped_count += 1
            continue

        if dry_run:
            run_dry_preview(
                source_path,
                item,
                target_role=target_role,
                sample_id=sample_id,
            )
            return 1, 0

        from weclone.core.inference.llm_client import LLMRequest

        request = LLMRequest.from_prompt(
            build_prompt(item, target_role=target_role, sample_id=sample_id),
            provider=provider,
            model=model,
            effort=effort,
            max_tokens=max_tokens,
            json_mode=True,
            metadata={
                "task": "event_distill",
                "source_file": str(source_path),
                "source_index": source_index,
                "sample_id": sample_id,
                "sample_time": item.get("time", ""),
            },
        )
        request_rows.append((key, source_index, item, sample_id, request))

    log(
        f"{source_path.name}: total={len(items)} pending={len(request_rows)} "
        f"skipped={skipped_count} batch_size={batch_size} output={output_path}"
    )
    if writeback_changed and not dry_run:
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
                    "writeback_field": EVENT_WRITEBACK_FIELD,
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
                atomic_save_any_json(output_path, source_data, indent=indent)
                writeback_changed = False
            atomic_save_json(state_path, state, indent=indent)
            log(f"{source_path.name}: wrote={done_count} calls={call_count}")
    finally:
        progress.close()

    if writeback_changed and not dry_run:
        atomic_save_any_json(output_path, source_data, indent=indent)

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

    state_path = Path(args.state_path) if args.state_path else default_event_state_path(args.output_dir)
    state = load_state(
        state_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    state["task"] = "event_distill"
    state["provider"] = args.llm_provider
    state["model"] = request_model
    state["effort"] = request_effort
    state["batch_size"] = args.batch_size
    state["writeback_field"] = EVENT_WRITEBACK_FIELD
    state["output_subdir"] = "event_people"
    state["updated_at"] = now_ts()

    log(f"输入目录: {args.input_dir}  文件数: {len(source_files)}")
    log(f"输出目录: {args.output_dir}")
    log(f"断点文件: {state_path}")
    log(f"事件字段: {EVENT_WRITEBACK_FIELD}")
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
