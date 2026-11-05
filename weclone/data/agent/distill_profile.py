"""Profile (state) and event memories from chat samples, several samples per model request.

Each sample's answer is kept in the LLM cache under its extraction rules and chat text, so a rerun extracts only
the samples that have no answer yet: those that failed, and those whose text is new or changed.
"""

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, NamedTuple

import click
import pyjson5
from tqdm import tqdm

from weclone.data.agent.distill_windows import (
    ChatSample,
    generate_window_batch,
    group_samples,
    make_window_request,
    sample_cache_key,
)
from weclone.data.agent.llm_cache import LLMCache
from weclone.utils import secure_storage
from weclone.utils.log import logger

DEFAULT_CONFIG_PATH = Path("settings.jsonc")
DEFAULT_TARGET_ROLE = "assistant"
DEFAULT_PROVIDER = "codex_exec"
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_TOKENS = None
DEFAULT_TIMEOUT = 120
STATE_WRITEBACK_FIELD = "state_memories"
# Each task writes its answers into copies of the chat files, in its own subdirectory and field.
TASK_OUTPUTS = {"state": ("state_people", STATE_WRITEBACK_FIELD), "event": ("event_people", "event_memories")}
TASK_LABELS = {"state": "用户画像", "event": "事件记忆"}
TASK = "state"

DEFAULT_LIMIT_FILES = None
DEFAULT_LIMIT_RECORDS = None
_print_lock = threading.Lock()


def default_args() -> SimpleNamespace:
    return SimpleNamespace(
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
        dry_run=False,
        indent=2,
    )


def log(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


def load_json(path: Path) -> Any:
    return secure_storage.read_json(path)


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


def required_config_value(
    config: dict[str, Any],
    key: str,
    config_path: Path,
    *,
    section: str = "agent_distill_args",
) -> Any:
    value = config.get(key)
    if value is None or value == "":
        raise ValueError(f"{section}.{key} is required in {config_path}")
    return value


def required_config_str(
    config: dict[str, Any],
    key: str,
    config_path: Path,
    *,
    section: str = "agent_distill_args",
) -> str:
    text = str(required_config_value(config, key, config_path, section=section)).strip()
    if not text:
        raise ValueError(f"{section}.{key} is required in {config_path}")
    return text


def positive_config_int(
    config: dict[str, Any],
    key: str,
    default: int,
    *,
    section: str,
    config_path: Path,
) -> int:
    value = config.get(key, default)
    if type(value) is not int or value < 1:
        raise ValueError(f"{section}.{key} must be a positive integer in {config_path}")
    return value


def optional_positive_config_int(
    config: dict[str, Any],
    key: str,
    default: int | None,
    *,
    section: str,
    config_path: Path,
) -> int | None:
    value = config.get(key, default)
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError(f"{section}.{key} must be a positive integer or null in {config_path}")
    return value


def resolve_llm_args(args: SimpleNamespace) -> SimpleNamespace:
    distill_config = load_agent_distill_config(args.config_path)
    args.llm_provider = normalize_llm_provider(args.llm_provider or distill_config.get("llm_provider"))
    for name in ("max_samples_per_window", "max_content_chars"):
        value = distill_config.get(name, getattr(args, name))
        if type(value) is not int or value < 1:
            raise ValueError(f"agent_distill_args.{name} must be a positive integer")
        setattr(args, name, value)

    # codex exec takes every request setting from codex_exec_args. The API backend reads its endpoint
    # from make_dataset_args and its concurrency, timeout and token limit from agent_distill_args.
    codex = args.llm_provider == "codex_exec"
    section = "codex_exec_args" if codex else "agent_distill_args"
    config = load_codex_exec_config(args.config_path) if codex else distill_config
    args.batch_size = positive_config_int(
        config, "batch_size", DEFAULT_BATCH_SIZE, section=section, config_path=args.config_path
    )
    args.max_tokens = optional_positive_config_int(
        config, "max_tokens", DEFAULT_MAX_TOKENS, section=section, config_path=args.config_path
    )
    args.timeout = positive_config_int(
        config, "timeout", DEFAULT_TIMEOUT, section=section, config_path=args.config_path
    )

    if not codex:
        args.model = None
        args.effort = None
        args.codex_command = None
        args.codex_sandbox = None
        return args

    args.model = required_config_str(config, "model", args.config_path, section=section)
    args.effort = required_config_str(config, "effort", args.config_path, section=section)
    args.codex_command = required_config_str(config, "command", args.config_path, section=section)
    args.codex_sandbox = required_config_str(config, "sandbox", args.config_path, section=section)
    return args


def atomic_save_any_json(path: Path, payload: Any, *, indent: int) -> None:
    secure_storage.write_json(path, payload, indent=indent, default=str)


def iter_chat_files(input_dir: Path) -> Iterable[Path]:
    for path in secure_storage.iter_files(input_dir):
        if "manifest" in path.name:
            continue
        yield secure_storage.logical_path(path)


def confirm_distillation(args: SimpleNamespace, *, task: str, file_count: int | None) -> bool:
    input_summary = f"{file_count} 个聊天文件" if file_count is not None else "聊天文件数将在数据准备后确定"
    click.echo(f"即将执行{task}蒸馏：输入 {args.input_dir}（{input_summary}），输出 {args.output_dir}。")
    click.echo(f"模型后端：{args.llm_provider}。输入目录中的聊天样本将交给该模型处理。")
    click.echo("聊天内容可能包含身份、联系方式和私密对话；模型服务可能记录或留存这些内容，存在隐私泄露风险。")
    click.echo("已经抽取过的聊天样本直接读取缓存结果，不会再次发送。")
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


def output_path_for(output_dir: Path, source_path: Path, task: str) -> Path:
    return output_dir / TASK_OUTPUTS[task][0] / source_path.name


def sample_id_for(item: dict[str, Any], source_index: int) -> str:
    for key in ("id", "local_id"):
        value = item.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return str(source_index)


def written_answer(task: str, result: dict[str, Any]) -> Any:
    """What a validated sample result writes into the sample: its memories, or its non-empty event arrays."""
    if task == "state":
        return result["memories"]
    return {kind: events for kind, events in result.items() if events}


class FileStats(NamedTuple):
    cached: int = 0  # samples answered from the cache
    pending: int = 0  # samples sent to the model
    windows: int = 0  # requests the pending samples were packed into
    failed: int = 0  # pending samples still without an answer after retries
    calls: int = 0  # model calls made, retries included


def process_task_file(
    source_path: Path,
    *,
    task: str,
    output_dir: Path,
    target_role: str,
    provider: str,
    model: str | None,
    effort: str | None,
    client: Any,
    cache: LLMCache,
    max_tokens: int | None,
    batch_size: int,
    limit_records: int | None,
    dry_run: bool,
    indent: int,
    progress_factory: Any = tqdm,
    max_samples_per_window: int = 8,
    max_content_chars: int = 400,
) -> FileStats:
    field = TASK_OUTPUTS[task][1]
    output_path = output_path_for(output_dir, source_path, task)
    source_data = load_json(source_path)
    all_items = chat_items(source_data)
    _, current_state_since = current_state_time_window(all_items)
    items = all_items[:limit_records] if limit_records is not None else all_items
    if not items:
        logger.warning(f"Skip {source_path}: no chat records found")
        return FileStats()

    samples = [
        ChatSample(index, sample_id_for(item, index), item, allow_current_state(item, current_state_since))
        for index, item in enumerate(items)
    ]
    keys = {
        sample.source_index: sample_cache_key(sample, task=task, target_role=target_role)
        for sample in samples
    }
    answers = cache.get_many(keys.values())
    pending = []
    for sample in samples:
        if keys[sample.source_index] in answers:
            sample.item[field] = answers[keys[sample.source_index]]
        else:
            pending.append(sample)
    cached = len(samples) - len(pending)
    windows = group_samples(pending, max_samples=max_samples_per_window, max_content_chars=max_content_chars)
    request_rows = [
        (
            window,
            make_window_request(
                window,
                task=task,
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
    since = current_state_since.isoformat() if task == "state" and current_state_since else ""
    log(
        f"{source_path.name}: total={len(items)} cached={cached} pending={len(pending)} windows={len(request_rows)} "
        + (f"current_state_since={since} " if since else "")
        + f"output={output_path}"
    )
    if dry_run:
        if request_rows:
            window, request = request_rows[0]
            logger.info(f"Dry run: {source_path.name}, samples={[s.sample_id for s in window]}")
            if secure_storage.is_encrypted_mode():
                logger.info("Encrypted mode: dry-run prompt content is not printed")
            else:
                print(request.messages[0]["content"])
        return FileStats(cached, len(pending), len(request_rows))

    failed = calls = 0
    progress = progress_factory(total=len(items), initial=cached, desc=source_path.name, unit="sample")
    try:
        for batch in batched(request_rows, batch_size):
            answered = []
            for outcome in generate_window_batch(client, batch, task=task):
                calls += len(outcome.attempts)
                if outcome.error:
                    failed += len(outcome.samples)
                    logger.warning(f"LLM window failed for {source_path.name}: {outcome.error}")
                else:
                    for sample in outcome.samples:
                        answer = written_answer(task, outcome.results[sample.sample_id])
                        sample.item[field] = answer
                        answered.append(
                            (keys[sample.source_index], f"{task}_distill", answer, outcome.attempts[-1].model)
                        )
                progress.update(len(outcome.samples))
            cache.put_many(answered)
    finally:
        progress.close()

    if failed:
        # An output file holds an answer for every sample or does not exist, so no later stage reads a partial one.
        secure_storage.resolve_path(output_path).unlink(missing_ok=True)
    else:
        atomic_save_any_json(output_path, source_data, indent=indent)
    log(f"{source_path.name}: extracted={len(pending) - failed} failed={failed} calls={calls}")
    return FileStats(cached, len(pending), len(request_rows), failed, calls)


def process_file(source_path: Path, **options: Any) -> FileStats:
    return process_task_file(source_path, task=TASK, **options)


def run_task(
    task: str,
    *,
    input_dir: Path,
    output_dir: Path,
    config_path: Path | None = None,
    confirmed: bool = False,
    dry_run: bool = False,
) -> FileStats:
    args = default_args()
    args.input_dir, args.output_dir = input_dir, output_dir
    if config_path is not None:
        args.config_path = config_path
    args.dry_run = dry_run
    secure_storage.configure(args.config_path)
    args = resolve_llm_args(args)
    request_model = args.model if args.llm_provider == "codex_exec" else None
    request_effort = args.effort if args.llm_provider == "codex_exec" else None
    source_files = list(iter_chat_files(args.input_dir))
    if args.limit_files is not None:
        source_files = source_files[: args.limit_files]

    if not source_files:
        raise FileNotFoundError(f"No chat JSON files found in {args.input_dir}")
    label = TASK_LABELS[task]
    if not confirmed and not confirm_distillation(args, task=label, file_count=len(source_files)):
        raise SystemExit("已取消蒸馏；未写入结果或调用模型。")

    cache = LLMCache()
    log(f"输入目录: {args.input_dir}  文件数: {len(source_files)}")
    log(f"输出目录: {args.output_dir}")
    log(f"结果缓存: {cache.path}")
    log(f"写回字段: {TASK_OUTPUTS[task][1]}")
    log(
        f"provider={args.llm_provider} model={request_model} "
        f"effort={request_effort} batch_size={args.batch_size} dry_run={args.dry_run}"
    )

    client = None
    if not args.dry_run:
        from weclone.core.inference.llm_client import build_llm_client

        args.output_dir.mkdir(parents=True, exist_ok=True)
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

    stats = []
    try:
        for source_path in source_files:
            stats.append(
                process_task_file(
                    source_path,
                    task=task,
                    output_dir=args.output_dir,
                    target_role=args.target_role,
                    provider=args.llm_provider,
                    model=request_model,
                    effort=request_effort,
                    client=client,
                    cache=cache,
                    max_tokens=args.max_tokens,
                    batch_size=args.batch_size,
                    max_samples_per_window=args.max_samples_per_window,
                    max_content_chars=args.max_content_chars,
                    limit_records=args.limit_records,
                    dry_run=args.dry_run,
                    indent=args.indent,
                )
            )
    finally:
        if client is not None:
            client.close()

    total = FileStats(*(sum(values) for values in zip(*stats))) if stats else FileStats()
    log(
        f"完成: cached={total.cached} pending={total.pending} windows={total.windows} "
        f"failed={total.failed} llm_calls={total.calls} dry_run={args.dry_run}"
    )
    if total.failed:
        raise RuntimeError(
            f"{label}抽取有 {total.failed} 个样本失败。重新运行 build-profile 只会重抽这些样本，其余从缓存读取：{cache.path}"
        )
    return total


def main(
    *,
    input_dir: Path,
    output_dir: Path,
    config_path: Path | None = None,
    confirmed: bool = False,
    dry_run: bool = False,
) -> FileStats:
    return run_task(
        TASK,
        input_dir=input_dir,
        output_dir=output_dir,
        config_path=config_path,
        confirmed=confirmed,
        dry_run=dry_run,
    )
