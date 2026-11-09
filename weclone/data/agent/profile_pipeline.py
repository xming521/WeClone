"""Build a profile from chat data, keeping each run's intermediate results separate.

One build runs at a time, under a lock that the command line and the web server share. Each build writes its stages
into a status file in its result directory, so the web shows the progress of a build whoever started it.
"""

import json
import os
import shutil
import signal
import subprocess
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from time import monotonic, perf_counter

import click
import pyjson5
from filelock import FileLock, Timeout

from weclone.core.inference import usage
from weclone.core.inference.usage import METER, UsageLimitReached
from weclone.utils import secure_storage
from weclone.utils.log import logger

RUNS_DIR = secure_storage.ROOT / "dataset/res_csv/agent/profile_runs"
LOCK_NAME = ".build.lock"
# Holds the result directory of the latest build, written when it starts.
LAST_BUILD_NAME = ".last_build"
STATUS_FILE = "build_status.json"
LOG_FILE = "build.log"
# A server writes these into the result directory before it starts the build there.
BUILD_FILES = {STATUS_FILE, LOG_FILE}
TELEGRAM_DIR = secure_storage.ROOT / "dataset/telegram"
# Chats uploaded from the web, kept apart from csv_folder and read as generic CSV.
UPLOAD_DIR = secure_storage.ROOT / "dataset/uploaded"
# Step, key and label of each stage, the [n/N] of the command line. Only the clone's build has the last one.
STAGES = [
    (1, "snapshot", "准备聊天数据"),
    (2, "distill_profile", "抽取画像记忆"),
    (3, "distill_event", "抽取事件记忆"),
    (4, "organize", "划分板块、分类和汇总记忆"),
    (5, "hierarchy", "把事实分成主题，生成画像层级"),
    (6, "summaries", "生成画像总结"),
    (7, "style", "抽取语言风格"),
]
PROGRESS_SECONDS = 1.0


class BuildCancelled(BaseException):
    """Raised by the SIGTERM handler, so the stages unwind as they do on Ctrl+C."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _stage(label: str, **counts):
    """Logs the start and end of a stage. The counts, such as its step, stay in an encrypted log too."""
    started = perf_counter()
    logger.info("{label}：开始", label=label, counts=counts)
    try:
        yield
    except BaseException as exc:
        logger.error(
            "{label}：失败（{error}），耗时 {counts[seconds]} 秒",
            label=label,
            error=type(exc).__name__,
            counts={**counts, "seconds": round(perf_counter() - started, 1)},
        )
        raise
    else:
        logger.info(
            "{label}：完成，耗时 {counts[seconds]} 秒",
            label=label,
            counts={**counts, "seconds": round(perf_counter() - started, 1)},
        )


def build_lock() -> FileLock:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return FileLock(RUNS_DIR / LOCK_NAME)


def build_running() -> bool:
    lock = build_lock()
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return True
    lock.release()
    return False


def last_build() -> Path | None:
    try:
        return Path((RUNS_DIR / LAST_BUILD_NAME).read_text(encoding="utf-8").strip())
    except OSError:
        return None


def record_build(run_dir: Path) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / LAST_BUILD_NAME).write_text(str(run_dir.resolve()), encoding="utf-8")


def read_status(run_dir: Path) -> dict | None:
    try:
        return json.loads((run_dir / STATUS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_status(run_dir: Path, status: dict) -> None:
    """Plaintext in either storage mode: stages, counts and times, never chat content."""
    path = run_dir / STATUS_FILE
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def stage_list(purpose: str) -> list[tuple[int, str, str]]:
    return [stage for stage in STAGES if purpose == "clone" or stage[1] != "style"]


def new_status(purpose: str, pid: int | None = None, pgid: int | None = None) -> dict:
    """A build that has started and has no stage running yet. `usage` is what its model calls used so far, in all
    and in each stage, with the cost in US dollars, or none when the model has no price. `estimate` is
    build_cost.describe() of the snapshot."""
    return {
        "purpose": purpose,
        "state": "running",
        "pid": pid,
        "pgid": pgid,
        "started_at": _now(),
        "finished_at": None,
        "chats": None,
        "usage": None,
        "estimate": None,
        "stages": [
            {
                "step": step,
                "key": key,
                "state": "pending",
                "started_at": None,
                "finished_at": None,
                # Requests done and in all, for the stages that count them.
                "done": None,
                "total": None,
                "unit": None,
                "usage": None,
            }
            for step, key, _ in stage_list(purpose)
        ],
        "error": None,
    }


class Build:
    """The status file of one build, rewritten at every change, and every second while the model calls of the build
    use more."""

    def __init__(self, run_dir: Path, purpose: str):
        self.run_dir = run_dir
        self.stages = stage_list(purpose)
        self.steps = self.stages[-1][0]
        self.status = new_status(purpose, os.getpid(), os.getpgrp() if hasattr(os, "getpgrp") else None)
        self.current: dict | None = None
        self.lock = threading.RLock()
        self.started = self.stage_started = self.saved = METER.totals()
        METER.limit = None
        self.saved_at = 0.0
        self.save()

    def save(self) -> None:
        with self.lock:
            self.saved = METER.totals()
            self.status["usage"] = usage.subtract(self.saved, self.started)
            if self.current is not None:
                self.current["usage"] = usage.subtract(self.saved, self.stage_started)
            write_status(self.run_dir, self.status)
            self.saved_at = monotonic()

    @contextmanager
    def stage(self, key: str):
        index, (step, _, label) = next((i, stage) for i, stage in enumerate(self.stages) if stage[1] == key)
        with self.lock:
            self.current = self.status["stages"][index]
            self.current.update(state="running", started_at=_now())
            self.stage_started = METER.totals()
            self.save()
        with _stage(f"[{step}/{self.steps}] {label}", step=step, steps=self.steps):
            # A stage that met the usage limit ends the build, also when it goes on without the answers or fails
            # over them.
            try:
                yield
            except Exception as exc:
                if METER.limit is not None and not isinstance(exc, UsageLimitReached):
                    raise UsageLimitReached(METER.limit) from exc
                raise
            if METER.limit is not None:
                raise UsageLimitReached(METER.limit)
        with self.lock:
            self.current.update(state="succeeded", finished_at=_now())
            self.save()
            self.current = None

    def progress(self, done: int, total: int) -> None:
        """Requests done and in all, in the running stage."""
        with self.lock:
            if self.current is None:
                return
            self.current.update(done=done, total=total, unit="requests")
            if done in (0, total) or monotonic() - self.saved_at >= PROGRESS_SECONDS:
                self.save()

    def _tick(self, stop: threading.Event) -> None:
        while not stop.wait(PROGRESS_SECONDS):
            if METER.totals() != self.saved:
                self.save()

    @contextmanager
    def running(self):
        """Records how the build ends. SIGTERM, Ctrl+C and an aborted prompt end it as cancelled."""

        def cancel(signum, frame):
            raise BuildCancelled()

        previous = signal.signal(signal.SIGTERM, cancel)
        stop = threading.Event()
        threading.Thread(target=self._tick, args=(stop,), name="build-status", daemon=True).start()
        try:
            yield
        except (BuildCancelled, KeyboardInterrupt, click.Abort):
            self.finish("cancelled")
            raise
        except BaseException as exc:
            self.finish("failed", exc)
            raise
        else:
            self.finish("succeeded")
        finally:
            stop.set()
            signal.signal(signal.SIGTERM, previous)

    def finish(self, state: str, exc: BaseException | None = None) -> None:
        with self.lock:
            if self.current is not None:
                self.current.update(state=state, finished_at=_now())
            if exc is not None:
                step, key = (self.current["step"], self.current["key"]) if self.current else (None, None)
                self.status["error"] = {
                    "step": step,
                    "key": key,
                    "type": type(exc).__name__,
                    # Persistent logs keep only the exception class in encrypted mode, and so does this file.
                    "message": "" if secure_storage.is_encrypted_mode() else str(exc)[:1000],
                }
            self.status.update(state=state, finished_at=_now())
            self.save()


def _problem(code: str, message: str, **params) -> dict:
    return {"code": code, "params": params, "message": message}


def csv_folder(chat_args) -> Path:
    folder = Path(chat_args.csv_folder)
    return folder if folder.is_absolute() else secure_storage.ROOT / folder


def load_chat_args(config_path: Path, chats_folder: Path | None = None):
    """The chats a build reads: those of the config, or the generic CSV chats of `chats_folder` when given."""
    from weclone.data.agent import chat_snapshot
    from weclone.utils.config_models import PlatformType

    chat_args = chat_snapshot.load_chat_args(config_path)
    if chats_folder is None:
        return chat_args
    return chat_args.model_copy(update={"platform": PlatformType.CHAT, "csv_folder": str(chats_folder)})


def count_csv_chats(folder: Path) -> int:
    """Chat folders holding CSV chunks, counted by name without reading them."""
    if not folder.is_dir():
        return 0
    return sum(
        next(secure_storage.iter_files(chat, "*.csv"), None) is not None
        for chat in folder.iterdir()
        if chat.is_dir()
    )


def _count_chats(chat_args) -> int:
    """Chat folders stage 1 would read, counted by name without reading them."""
    if str(chat_args.platform) == "telegram":
        if not TELEGRAM_DIR.is_dir():
            return 0
        return sum(
            secure_storage.file_exists(folder / "result.json")
            for folder in TELEGRAM_DIR.iterdir()
            if folder.is_dir()
        )
    return count_csv_chats(csv_folder(chat_args))


def chat_problems(config_path: Path, chats_folder: Path | None = None) -> tuple[dict, list[dict]]:
    """The chats a build would read, as in `load_chat_args`, and what keeps it from reading them."""
    try:
        chat_args = load_chat_args(config_path, chats_folder)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"platform": None, "folder": None, "count": 0}, [
            _problem("config_invalid", f"配置文件读取失败：{exc}", detail=str(exc))
        ]
    platform = str(chat_args.platform)
    problems = []
    if platform == "telegram":
        folder = "dataset/telegram"
        if not chat_args.telegram_args or not chat_args.telegram_args.my_id:
            problems.append(
                _problem(
                    "telegram_id_missing", "settings.jsonc 里没有填 make_dataset_args.telegram_args.my_id"
                )
            )
    else:
        folder = str(chat_args.csv_folder)
    count = _count_chats(chat_args)
    if not count:
        problems.append(
            _problem("chats_missing", f"{folder} 里没有聊天记录", platform=platform, folder=folder)
        )
    return {"platform": platform, "folder": folder, "count": count}, problems


def codex_problems(command: str) -> list[dict]:
    """Whether the Codex CLI is there and signed in."""
    if shutil.which(command) is None:
        return [_problem("codex_missing", f"找不到 Codex CLI：{command}", command=command)]
    try:
        logged_in = (
            subprocess.run(
                [command, "login", "status"], capture_output=True, timeout=30, check=False
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        logged_in = False
    if not logged_in:
        return [
            _problem("codex_logged_out", f"Codex CLI 没有登录，请先执行 {command} login", command=command)
        ]
    return []


def _backend_problems(config_path: Path, max_context_tokens: int | None) -> tuple[dict, list[dict]]:
    from weclone.data.agent import distill_profile, organize_memories

    args = distill_profile.default_args()
    args.config_path = config_path
    try:
        args = distill_profile.resolve_llm_args(args)
    except (OSError, ValueError) as exc:
        return {"provider": None, "model": None, "max_context_tokens": max_context_tokens}, [
            _problem("config_invalid", f"配置文件有误：{exc}", detail=str(exc))
        ]
    backend = {"provider": args.llm_provider, "model": args.model, "max_context_tokens": max_context_tokens}
    if args.llm_provider == "api":
        # The endpoint build_llm_client reads.
        config = pyjson5.loads(config_path.read_text(encoding="utf-8")).get("make_dataset_args") or {}
        backend["model"] = str(config.get("model_name") or "").strip() or None
        missing = [
            key for key in ("base_url", "llm_api_key", "model_name") if not str(config.get(key) or "").strip()
        ]
        problems = []
        if missing:
            problems.append(
                _problem(
                    "api_config_missing",
                    f"settings.jsonc 的 make_dataset_args 里没有填 {'、'.join(missing)}",
                    keys=missing,
                )
            )
        if max_context_tokens is None:
            args.max_context_tokens = None
            try:
                backend["max_context_tokens"] = organize_memories.resolve_context_window(args)
            except ValueError as exc:
                problems.append(_problem("config_invalid", f"配置文件有误：{exc}", detail=str(exc)))
        return backend, problems

    problems = codex_problems(args.codex_command)
    if problems:
        return backend, problems
    if max_context_tokens is None:
        args.max_context_tokens = None
        backend["max_context_tokens"] = organize_memories.resolve_context_window(args)
    return backend, []


def _style_problems(config_path: Path) -> list[dict]:
    from weclone.data.agent.distill_style import run as distill_style

    ltp_model = distill_style.parse_args(
        ["--input-dir", ".", "--output", ".", "--settings", str(config_path)]
    ).ltp_model
    if find_spec("ltp") is not None and ltp_model.is_dir():
        return []
    return [
        _problem(
            "style_deps_missing",
            "AI 分身要抽取语言风格，请先执行 uv pip install --group style --torch-backend cpu -e .，"
            f"并把 LTP 模型放在 {ltp_model}",
            model_dir=str(ltp_model),
        )
    ]


def check(
    config_path: Path, purpose: str, max_context_tokens: int | None = None, chats_folder: Path | None = None
) -> dict:
    """What would stop a build before its first model call, found without calling the model."""
    secure_storage.configure(config_path)
    chats, problems = chat_problems(config_path, chats_folder)
    backend, backend_problems = _backend_problems(config_path, max_context_tokens)
    # Marked for the web page, where unsaved changes to the model service make them out of date.
    problems += [{**problem, "backend": True} for problem in backend_problems]
    if purpose == "clone":
        problems += _style_problems(config_path)
    if build_running():
        problems.append(_problem("build_running", "已经有一次构建在运行"))
    return {"backend": backend, "chats": chats, "problems": problems}


def new_run_dir() -> Path:
    return RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def run(
    *,
    with_style: bool,
    output_dir: Path | None,
    config_path: Path,
    max_context_tokens: int | None = None,
    confirmed: bool = False,
    chats_folder: Path | None = None,
) -> Path:
    from weclone.data.agent import build_cost, distill_profile
    from weclone.data.agent.distill_style import run as distill_style

    secure_storage.configure(config_path)
    purpose = "clone" if with_style else "assistant"
    found = check(config_path, purpose, max_context_tokens, chats_folder)
    if found["problems"]:
        raise ValueError(
            "\n".join(["运行前检查没有通过："] + [f"- {item['message']}" for item in found["problems"]])
        )
    max_context_tokens = found["backend"]["max_context_tokens"]
    args = distill_profile.default_args()
    args.config_path = config_path
    args = distill_profile.resolve_llm_args(args)
    if output_dir is None:
        output_dir = new_run_dir()
    if output_dir.exists() and any(path.name not in BUILD_FILES for path in output_dir.iterdir()):
        raise ValueError(f"结果目录非空，请指定新的目录：{output_dir}")
    people_dir = output_dir / "people"
    distill_dir = output_dir / "distill"
    organization_dir = output_dir / "memory_organization"
    hierarchy_dir = output_dir / "profile_hierarchy"
    args.input_dir, args.output_dir = people_dir, distill_dir
    style_dir = output_dir / "style"
    style_argv = ["--input-dir", str(people_dir), "--output", str(style_dir), "--settings", str(config_path)]
    click.echo(f"本次结果目录：{output_dir}")
    click.echo(f"抽取内容：{'画像、事件和语言风格' if with_style else '画像和事件'}")
    click.echo(f"输入 Token 上限：{max_context_tokens}")
    task = "用户画像、事件记忆和语言风格" if with_style else "用户画像和事件记忆"
    if not confirmed and not distill_profile.confirm_distillation(args, task=task, file_count=None):
        raise click.Abort()

    lock = build_lock()
    try:
        # A server polling the lock holds it for a moment.
        lock.acquire(timeout=5)
    except Timeout:
        raise RuntimeError("已经有一次构建在运行") from None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        record_build(output_dir)
        settings = build_cost.llm_settings(config_path)
        METER.price = build_cost.price_of(settings)
        build = Build(output_dir, purpose)
        with build.running():
            with _stage("启动：加载抽取与整理模块"):
                from weclone.data.agent import distill_event, llm_profile_hierarchy, organize_memories

            with build.stage("snapshot"):
                from weclone.data.agent import chat_snapshot

                chat_snapshot.build_snapshot(people_dir, load_chat_args(config_path, chats_folder))

            source_files = list(distill_profile.iter_chat_files(people_dir))
            if not source_files:
                raise FileNotFoundError(f"没有生成聊天样本：{people_dir}")
            logger.info("聊天样本准备完成：{counts[chats]} 个聊天文件。", counts={"chats": len(source_files)})
            try:
                plan, _ = build_cost.count(build_cost.chats_of_snapshot(people_dir), settings, purpose)
                estimate = build_cost.describe(plan, build_cost.saved_probe(settings), settings)
            except Exception as exc:
                # The estimate only informs, so a build goes on without it.
                logger.warning("构建前的估算失败：{error}", error=exc)
                estimate = None
            with build.lock:
                build.status["chats"] = len(source_files)
                build.status["estimate"] = estimate

            with build.stage("distill_profile"):
                # Raises when samples failed. Rerunning build-profile into a new directory asks only for those.
                distill_profile.main(
                    input_dir=people_dir,
                    output_dir=distill_dir,
                    config_path=config_path,
                    confirmed=True,
                    on_progress=build.progress,
                )
            with build.stage("distill_event"):
                distill_event.main(
                    input_dir=people_dir,
                    output_dir=distill_dir,
                    config_path=config_path,
                    confirmed=True,
                    on_progress=build.progress,
                )

            with build.stage("organize"):
                organize_memories.main(
                    [
                        "--stage",
                        "all",
                        "--input-dir",
                        str(distill_dir),
                        "--output-dir",
                        str(organization_dir),
                        "--config-path",
                        str(config_path),
                        "--max-context-tokens",
                        str(max_context_tokens),
                    ]
                )

            with build.stage("hierarchy"):
                llm_profile_hierarchy.main(
                    [
                        "--input-path",
                        str(organization_dir / "organized_memories.json"),
                        "--output-dir",
                        str(hierarchy_dir),
                        "--config-path",
                        str(config_path),
                        "--max-context-tokens",
                        str(max_context_tokens),
                    ]
                )
                result = secure_storage.resolve_path(hierarchy_dir / "profile_hierarchy.json")
                if not result.is_file():
                    raise FileNotFoundError(f"没有生成画像文件：{result}")

            # The profile is done by now, so domains left without a summary only get a note.
            with build.stage("summaries"):
                from weclone.server.profile_summary import build_summaries

                missing = build_summaries(hierarchy_dir)
            if missing:
                click.echo(f"画像总结：{missing}")

            if with_style:
                with build.stage("style"):
                    distill_style.main(style_argv)
        # Chats uploaded from the web serve one build, so the next one starts from its own upload.
        if chats_folder is not None and chats_folder.resolve() == UPLOAD_DIR.resolve():
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    finally:
        lock.release()
    return result
