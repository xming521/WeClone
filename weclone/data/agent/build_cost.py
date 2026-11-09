"""What a build will cost: its requests, counted from the chats without calling the model, and, once a probe of a
few real requests has measured what one request uses, its tokens, time and money.

A probe sends windows the build would send anyway, the same way, and caches their answers, so the build does not pay
for them again. Its result is kept per model service and request settings, next to the LLM cache."""

import contextvars
import json
import math
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyjson5

from weclone.core.inference._common import CODEX_REQUEST_INTERVAL_SECONDS
from weclone.core.inference.model_prices import Price, lookup
from weclone.core.inference.usage import UsageLimitReached, read_usage
from weclone.data.agent import distill_profile
from weclone.data.agent.distill_windows import (
    ChatSample,
    generate_window_batch,
    group_samples,
    make_window_request,
    sample_cache_key,
    window_prompt,
)
from weclone.data.agent.llm_cache import LLMCache, cache_key
from weclone.utils.log import logger

TASKS = {"state": "distill_profile", "event": "distill_event"}
PROBE_WINDOWS = 5  # per task
PROBES_NAME = "usage_probes.json"
# Organize, hierarchy and summaries against extraction, measured on two builds of the same chats: their requests and
# prompt characters per extraction request and prompt character, the answer tokens of one of their requests, and its
# time per extraction request's.
LATER_REQUESTS = 0.11
LATER_CHARS = 0.24
LATER_ANSWER_TOKENS = 950
LATER_SECONDS = 2.0
# Student's t at 0.95 for 1 to 9 degrees of freedom: the mean of n probes ± t·sd/√n is a 90% interval.
# How finely a build's time is split into the price periods it runs in.
PERIOD_STEP = timedelta(minutes=5)
T95 = (6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833)


def llm_settings(config_path: Path) -> SimpleNamespace:
    """The request settings extraction uses, with the endpoint of the API backend and the seconds Codex waits between
    two launches."""
    args = distill_profile.default_args()
    args.config_path = config_path
    args = distill_profile.resolve_llm_args(args)
    args.base_url = args.model_name = None
    args.launch_interval = 0.0
    config = pyjson5.loads(Path(config_path).read_text(encoding="utf-8"))
    if args.llm_provider == "api":
        endpoint = config.get("make_dataset_args") or {}
        args.base_url = str(endpoint.get("base_url") or "").strip().rstrip("/") or None
        args.model_name = str(endpoint.get("model_name") or "").strip() or None
    else:
        pacing = config.get("codex_exec_args") or {}
        args.launch_interval = float(pacing.get("request_interval_seconds", CODEX_REQUEST_INTERVAL_SECONDS))
    return args


def price_of(args: SimpleNamespace) -> Price | None:
    return lookup(args.base_url, args.model_name) if args.llm_provider == "api" else None


def settings_key(args: SimpleNamespace) -> str:
    """What a probe's usage depends on: the service, the model, how it is asked to think, and how much each request
    sends and may answer."""
    shape = (args.max_samples_per_window, args.max_content_chars, args.max_tokens)
    if args.llm_provider == "api":
        return cache_key(
            "api", args.base_url, args.model_name, args.reasoning_effort, args.extra_body, *shape
        )
    return cache_key("codex_exec", args.model, args.effort, *shape)


def probes_path() -> Path:
    return LLMCache().path.with_name(PROBES_NAME)


def saved_probe(args: SimpleNamespace) -> dict | None:
    try:
        probes = json.loads(probes_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    found = probes.get(settings_key(args)) if isinstance(probes, dict) else None
    return found if isinstance(found, dict) else None


def save_probe(args: SimpleNamespace, probe: dict) -> None:
    """Plaintext in either storage mode, like the build status: numbers and a time, never chat content."""
    path = probes_path()
    try:
        probes = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        probes = {}
    probes[settings_key(args)] = probe
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(probes, indent=2), encoding="utf-8")
    temporary.replace(path)


def chats_of_config(config_path: Path, chats_folder: Path | None) -> list[list[dict]]:
    """Each chat's items as the build's snapshot would write them, cut in memory."""
    from weclone.data.agent import chat_snapshot
    from weclone.data.agent.profile_pipeline import load_chat_args

    contacts = chat_snapshot.contact_sessions(load_chat_args(config_path, chats_folder))
    return [chat_snapshot.session_items(kept, chat_with, None) for chat_with, kept, _ in contacts.values()]


def chats_of_snapshot(people_dir: Path) -> list[list[dict]]:
    return [
        distill_profile.chat_items(distill_profile.load_json(path))
        for path in distill_profile.iter_chat_files(people_dir)
    ]


def style_requests(config_path: Path, sessions: int) -> int:
    """One request per batch of the sessions style samples, for each style group that asks the model; the groups
    batched jointly share their requests."""
    from weclone.data.agent.distill_style import run as distill_style
    from weclone.data.agent.distill_style.batching import JOINT_LABELS
    from weclone.data.agent.distill_style.taxonomy import extractors_for, selected_categories

    style = distill_style.parse_args(["--input-dir", ".", "--output", ".", "--settings", str(config_path)])
    if not style.llm:
        return 0
    asking = {
        m for m in extractors_for(selected_categories(style.chapter)) if m not in distill_style.LOCAL_ONLY
    }
    streams = len(asking - set(JOINT_LABELS)) + bool(asking & set(JOINT_LABELS))
    return streams * math.ceil(min(style.llm_samples, sessions) / style.llm_batch_size)


def count(
    chats: list[list[dict]], args: SimpleNamespace, purpose: str
) -> tuple[dict, dict[str, list[list[ChatSample]]]]:
    """The requests of each stage, and the extraction windows still to send, by task. Windows answered already are
    not requests."""
    cache = LLMCache()
    plan: dict[str, Any] = {}
    pending: dict[str, list[list[ChatSample]]] = {}
    all_requests = all_chars = 0
    for task, stage in TASKS.items():
        pending[task] = []
        for items in chats:
            _, since = distill_profile.current_state_time_window(items)
            found = distill_profile.plan_samples(
                items,
                since,
                task=task,
                target_role=args.target_role,
                cache=cache,
                max_samples_per_window=args.max_samples_per_window,
                max_content_chars=args.max_content_chars,
            )
            pending[task] += found.windows
            every = group_samples(
                found.samples,
                max_samples=args.max_samples_per_window,
                max_content_chars=args.max_content_chars,
            )
            all_requests += len(every)
            all_chars += sum(len(window_prompt(w, task=task, target_role=args.target_role)) for w in every)
        plan[stage] = len(pending[task])
    plan["later"] = round(all_requests * LATER_REQUESTS)
    plan["later_chars"] = round(all_chars * LATER_CHARS)
    sessions = sum(len(items) for items in chats)
    plan["style"] = style_requests(args.config_path, sessions) if purpose == "clone" else 0
    return plan, pending


def _interval(values: list[float]) -> tuple[float, float]:
    """The mean and the half width of its 90% interval, relative to the mean."""
    mean = statistics.fmean(values)
    if len(values) < 2 or mean <= 0:
        return mean, 0.0
    return mean, T95[min(len(values), 10) - 2] * statistics.stdev(values) / math.sqrt(len(values)) / mean


def off_peak_share(price: Price, start: datetime, seconds: float) -> float:
    """The share of a build's time from `start` on that falls in the price's off-peak hours."""
    steps = max(1, math.ceil(seconds / PERIOD_STEP.total_seconds()))
    return sum(price.off_peak_at(start + PERIOD_STEP * (step + 0.5)) for step in range(steps)) / steps


def describe(plan: dict, probe: dict | None, args: SimpleNamespace, start: datetime | None = None) -> dict:
    """`probe_requests` is how many requests a probe would send now: none when every extraction window has an
    answer. The cost is for a build that starts at `start`, now unless given, at the price of each period its time
    falls in."""
    extraction = plan["distill_profile"] + plan["distill_event"] + plan["style"]
    result: dict[str, Any] = {
        "backend": args.llm_provider,
        "requests": extraction + plan["later"],
        "probe_requests": sum(min(PROBE_WINDOWS, plan[stage]) for stage in TASKS.values()),
        "plan": {key: plan[key] for key in ("distill_profile", "distill_event", "later", "style")},
        "probe": None,
        "unmetered": False,
        "input_tokens": None,
        "output_tokens": None,
        "cost": None,
        "seconds": None,
    }
    windows = (probe or {}).get("windows") or []
    if not windows:
        return result
    seconds, rel_seconds = _interval([w["seconds"] for w in windows])

    def wall(per_request: float) -> float:
        """Concurrent requests share their time, but none starts sooner than the launch interval after the last."""
        each = [(extraction, per_request), (plan["later"], per_request * LATER_SECONDS)]
        return sum(n * max(s / max(1, args.batch_size), args.launch_interval) for n, s in each)

    result["seconds"] = {
        "low": round(wall(seconds * max(0.0, 1 - rel_seconds))),
        "high": round(wall(seconds * (1 + rel_seconds))),
    }
    metered = [w for w in windows if w["metered"]]
    means = {
        key: statistics.fmean(w[key] for w in metered) if metered else 0.0
        for key in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")
    }
    result["probe"] = {"requests": len(windows), "at": probe["at"], **means, "seconds": seconds}
    if not metered:
        result["unmetered"] = True
        return result
    chars = sum(w["chars"] for w in metered)
    later_input = plan["later_chars"] * (sum(w["input_tokens"] for w in metered) / chars if chars else 0.0)
    later_output = plan["later"] * (means["reasoning_tokens"] + LATER_ANSWER_TOKENS)
    result["input_tokens"] = round(extraction * means["input_tokens"] + later_input)
    result["output_tokens"] = round(extraction * means["output_tokens"] + later_output)
    price = price_of(args)
    if price is None:
        return result

    def costs(off_peak: bool) -> tuple[list[float], float]:
        return [
            price.cost(w["input_tokens"], w["cached_tokens"], w["output_tokens"], off_peak=off_peak)
            for w in metered
        ], price.cost(round(later_input), 0, round(later_output), off_peak=off_peak)

    (peak, later_peak), (off, later_off) = costs(False), costs(True)

    def total(share: float) -> tuple[float, float, float]:
        """The mean cost of a request, its relative half width, and the build's, with `share` of it off-peak."""
        mean, rel = _interval([share * o + (1 - share) * p for o, p in zip(off, peak)])
        return mean, rel, extraction * mean + share * later_off + (1 - share) * later_peak

    start = start or datetime.now(timezone.utc)
    # A longer or a shorter build reaches into other periods, so each end of the range takes the period mix that
    # moves it furthest.
    ends = [total(off_peak_share(price, start, result["seconds"][end])) for end in ("low", "high")]
    low, high = min(ends, key=lambda end: end[2]), max(ends, key=lambda end: end[2])
    full_peak, full_off = total(0.0)[2], total(1.0)[2]
    result["cost"] = {
        "low": low[2] * max(0.0, 1 - low[1]),
        "high": high[2] * (1 + high[1]),
        "per_request": {"low": low[0] * max(0.0, 1 - low[1]), "high": high[0] * (1 + high[1])},
        "off_peak_share": off_peak_share(price, start, wall(seconds)) if price.off_peak is not None else None,
        "off_peak_saving": 1 - full_off / full_peak if price.off_peak is not None and full_peak else None,
        "source": price.source,
        "vendor": price.vendor,
        "prices_date": price.date,
    }
    return result


def estimate(config_path: Path, purpose: str, chats_folder: Path | None = None) -> dict:
    """The estimate of a build of the configured or uploaded chats, before its snapshot exists."""
    args = llm_settings(config_path)
    plan, _ = count(chats_of_config(config_path, chats_folder), args, purpose)
    return describe(plan, saved_probe(args), args)


def _window(task: str, request, outcome) -> dict:
    """What one probed window used, attempts that were sent again included, by the client's retries too."""
    usages = [
        read_usage(usage)
        for response in outcome.attempts
        for usage in response.metadata.get("attempt_usages", [response.metadata.get("usage")])
    ]
    found = [usage for usage in usages if usage is not None]
    return {
        "task": task,
        "chars": len(request.messages[0]["content"]),
        "metered": bool(found),
        "input_tokens": sum(u.input_tokens for u in found),
        "cached_tokens": sum(u.cached_tokens for u in found),
        "output_tokens": sum(u.output_tokens for u in found),
        "reasoning_tokens": sum(u.reasoning_tokens for u in found),
        "seconds": sum(response.elapsed_s or 0.0 for response in outcome.attempts),
    }


def probe(config_path: Path, purpose: str, chats_folder: Path | None = None) -> dict:
    """Sends PROBE_WINDOWS random windows of each task that have no answer yet, as the build sends them, and returns
    the estimate with what they used. Raises RuntimeError when no window got an answer."""
    from weclone.core.inference.llm_client import RetryPolicy, build_llm_client

    args = llm_settings(config_path)
    plan, pending = count(chats_of_config(config_path, chats_folder), args, purpose)
    chosen = {
        task: random.sample(windows, min(PROBE_WINDOWS, len(windows))) for task, windows in pending.items()
    }
    if not any(chosen.values()):
        return describe(plan, saved_probe(args), args)
    codex = args.llm_provider == "codex_exec"
    batches = {
        task: [
            (
                window,
                make_window_request(
                    window,
                    task=task,
                    source_path=Path("probe"),
                    target_role=args.target_role,
                    provider=args.llm_provider,
                    model=args.model if codex else None,
                    effort=args.effort if codex else None,
                    max_tokens=args.max_tokens,
                ),
            )
            for window in windows
        ]
        for task, windows in chosen.items()
        if windows
    }
    client = build_llm_client(
        args.llm_provider,
        config_path=args.config_path,
        model=args.model if codex else None,
        max_workers=args.batch_size,
        timeout=args.timeout,
        effort=args.effort,
        reasoning_effort=args.reasoning_effort,
        extra_body=args.extra_body,
        structured_output=args.structured_output,
        command=args.codex_command,
        sandbox=args.codex_sandbox,
        # A service that keeps failing says so now rather than after the build's long retries.
        retry_policy=RetryPolicy(max_retries=2),
    )
    try:
        # Both tasks at once, so every window goes out together as in a build.
        with ThreadPoolExecutor(max_workers=len(batches)) as tasks:
            futures = {
                task: tasks.submit(
                    contextvars.copy_context().run, generate_window_batch, client, rows, task=task
                )
                for task, rows in batches.items()
            }
            outcomes = {}
            for task, future in futures.items():
                try:
                    outcomes[task] = future.result()
                except UsageLimitReached:
                    # The task started after the other met the limit, whose answers are still kept below.
                    outcomes[task] = []
    finally:
        client.close()

    windows, answered, errors = [], [], []
    for task, rows in batches.items():
        for (samples, request), outcome in zip(rows, outcomes[task]):
            measured = _window(task, request, outcome)
            if outcome.error:
                errors.append(outcome.error)
                if not measured["metered"]:
                    continue
            else:
                # Answered windows are cached now and leave the count.
                plan[TASKS[task]] -= 1
                for sample in samples:
                    answer = distill_profile.written_answer(task, outcome.results[sample.sample_id])
                    answered.append(
                        (
                            sample_cache_key(sample, task=task, target_role=args.target_role),
                            f"{task}_distill",
                            answer,
                            outcome.attempts[-1].model,
                        )
                    )
            windows.append(measured)
    LLMCache().put_many(answered)
    if client.limit is not None:
        raise client.limit
    if not windows:
        raise RuntimeError(errors[0] if errors else "试算没有得到回答")
    logger.info(
        "Usage probe: {counts[windows]} windows, {counts[failed]} failed",
        counts={"windows": len(windows), "failed": len(errors)},
    )
    result = {"at": datetime.now(timezone.utc).isoformat(), "windows": windows}
    save_probe(args, result)
    return describe(plan, result, args)
