from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tqdm import tqdm

from weclone.data.agent.distill_profile import (
    allow_current_state,
    atomic_save_any_json,
    atomic_save_json,
    batched,
    chat_items,
    confirm_distillation,
    current_state_time_window,
)
from weclone.data.agent.distill_profile import default_args as state_default_args
from weclone.data.agent.distill_profile import (
    iter_chat_files,
    load_json,
    load_state,
    log,
    make_record,
    now_ts,
    other_role,
    resolve_llm_args,
    response_payload,
    sample_id_for,
    state_key,
)
from weclone.data.agent.distill_windows import (
    ChatSample,
    generate_window_batch,
    group_samples,
    make_window_request,
    render_sample,
)
from weclone.prompts.chat_distill import EVENT_EXTRACT_PROMPT
from weclone.utils import secure_storage
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
    return render_sample(item, sample_id=sample_id, target_role=target_role, include_time=True)


def build_prompt(item: dict[str, Any], *, target_role: str, sample_id: str) -> str:
    rendered_chat = render_event_chat(item, target_role=target_role, sample_id=sample_id)
    return EVENT_EXTRACT_PROMPT.replace("{{CHAT_JSON}}", rendered_chat)


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
    _, current_state_cutoff_time = current_state_time_window(all_items)
    items = all_items[:limit_records] if limit_records is not None else all_items
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
                task="event",
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
            if secure_storage.is_encrypted_mode():
                logger.info("Encrypted mode: dry-run prompt content is not printed")
            else:
                print(request.messages[0]["content"])
            return len(window), 0
        return 0, 0

    log(
        f"{source_path.name}: total={len(items)} pending={len(pending_samples)} windows={len(request_rows)} "
        f"skipped={skipped_count} batch_size={batch_size} output={output_path}"
    )
    if not dry_run and (writeback_changed or (skipped_count and not secure_storage.file_exists(output_path))):
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
            outcomes = generate_window_batch(client, batch, task="event")
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
                        "writeback_field": EVENT_WRITEBACK_FIELD,
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
                    logger.warning(f"LLM window failed for {source_path.name}; details saved in checkpoint")
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
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    config_path: Path | None = None,
    confirmed: bool = False,
) -> None:
    args = default_args()
    if input_dir is not None:
        args.input_dir = input_dir
    if output_dir is not None:
        args.output_dir = output_dir
    if config_path is not None:
        args.config_path = config_path
    secure_storage.configure(args.config_path)
    args = resolve_llm_args(args)
    request_model = args.model if args.llm_provider == "codex_exec" else None
    request_effort = args.effort if args.llm_provider == "codex_exec" else None
    source_files = list(iter_chat_files(args.input_dir))
    if args.limit_files is not None:
        source_files = source_files[: args.limit_files]

    if not source_files:
        raise FileNotFoundError(f"No chat JSON files found in {args.input_dir}")
    if not confirmed and not confirm_distillation(args, task="事件记忆", file_count=len(source_files)):
        raise SystemExit("已取消蒸馏；未写入结果或调用模型。")

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
    state["max_samples_per_window"] = args.max_samples_per_window
    state["max_content_chars"] = args.max_content_chars
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
