"""Building a profile from the web: the checks before a build, the build in a background process, its progress,
and the runs the server can switch between."""

import os
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from weclone.data.agent import profile_pipeline as pipeline
from weclone.server import style_view
from weclone.server.errors import ApiError
from weclone.utils import secure_storage as secure
from weclone.utils.log import REDACT_ENV

CLI_SCRIPT = Path(__file__).resolve().parents[1] / "cli.py"


def run_complete(run: Path) -> bool:
    """Whether the run has a profile to show, including one whose source was removed but whose snapshot is kept."""
    hierarchy = run / "profile_hierarchy"
    return secure.file_exists(hierarchy / "profile_hierarchy.json") or secure.file_exists(
        hierarchy / "profile_review.sqlite3"
    )


def reviewed_at(database: Path) -> str | None:
    """When the owner last changed a fact, read without creating the review database."""
    if not secure.file_exists(database):
        return None
    with secure.encrypted_sqlite(database) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='history'").fetchone():
            return None
        return db.execute("SELECT MAX(at) FROM history").fetchone()[0]


def generated_at(run: Path) -> str | None:
    source = secure.resolve_path(run / "profile_hierarchy" / "profile_hierarchy.json")
    if not source.is_file():
        return None
    return datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat()


def same(first: Path | None, second: Path | None) -> bool:
    return first is not None and second is not None and first.resolve() == second.resolve()


def config_path() -> Path:
    return Path(os.environ.get("WECLONE_CONFIG_PATH", secure.ROOT / "settings.jsonc"))


# upload: the chats uploaded from the web. config: the platform and csv_folder of the config.
ChatSource = Literal["upload", "config"]


def chats_folder(source: ChatSource | None) -> Path | None:
    """The folder a build reads for this source, None for the config's chats. Without a source, the uploaded chats
    if there are any."""
    if source is None:
        source = "upload" if pipeline.count_csv_chats(pipeline.UPLOAD_DIR) else "config"
    return pipeline.UPLOAD_DIR if source == "upload" else None


class BuildStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    purpose: Literal["assistant", "clone"]
    consent: bool
    max_context_tokens: int | None = Field(default=None, ge=1)
    chats: ChatSource | None = None


class RunChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)


class Builds:
    """Builds started here run in their own process group and outlive the server. Builds started on the command
    line show up too: both write the same status file under the same lock."""

    def __init__(self, runs, *, demo: bool, script: Path | None):
        self.runs = runs
        self.demo = demo
        self.script = script
        self.process: subprocess.Popen | None = None
        self.cancelled: Path | None = None
        self.lock = threading.Lock()
        # A server started on a run keeps showing it over a build that ended before it started.
        if runs.chosen and not demo:
            run = pipeline.last_build()
            status = pipeline.read_status(run) if run is not None else None
            if status is not None and status["state"] != "running":
                runs.settings.save_setting("settled_build", str(run))

    def started_here(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def running(self) -> bool:
        return self.started_here() or pipeline.build_running()

    def state(self, run: Path, status: dict) -> str:
        # The build running now, if any, is the latest one.
        if status["state"] != "running" or (self.running() and same(run, pipeline.last_build())):
            return status["state"]
        # The process ended without recording how, as when it was killed before its SIGTERM handler was in place.
        return "cancelled" if same(run, self.cancelled) else "interrupted"

    def latest(self) -> dict | None:
        run = pipeline.last_build()
        status = pipeline.read_status(run) if run is not None and not self.demo else None
        if status is None:
            return None
        shown = {key: value for key, value in status.items() if key not in ("pid", "pgid")}
        return {
            **shown,
            "state": self.state(run, status),
            "run": run.name,
            "current": same(run, self.runs.run_dir()),
            "log": str(run / pipeline.LOG_FILE),
        }

    def settle(self):
        """Opens the latest build once after it succeeds, unless the owner has reviewed the run shown, whose
        reviews stay with that run."""
        run = pipeline.last_build()
        if self.demo or run is None or self.runs.settings.setting("settled_build") == str(run):
            return
        status = pipeline.read_status(run)
        if status is None or status["state"] != "succeeded":
            return
        self.runs.settings.save_setting("settled_build", str(run))
        if run_complete(run) and not self.runs.reviewed():
            self.runs.switch(run)

    def check(self, purpose: str, max_context_tokens: int | None, source: ChatSource | None) -> dict:
        if self.demo:
            raise ApiError(409, "demo_build_disabled", "演示模式不能构建画像")
        folder = chats_folder(source)
        found = pipeline.check(config_path(), purpose, max_context_tokens, folder)
        config = pipeline.chat_problems(config_path())[0] if folder else found["chats"]
        found["chats"] = {
            "source": "upload" if folder else "config",
            "upload": {"count": pipeline.count_csv_chats(pipeline.UPLOAD_DIR)},
            "config": config,
        }
        return found

    def start(self, request: BuildStart) -> dict | None:
        if self.demo:
            raise ApiError(409, "demo_build_disabled", "演示模式不能构建画像")
        if not request.consent:
            raise ApiError(422, "consent_required", "请先确认把聊天内容发送给模型服务")
        with self.lock:
            if self.running():
                raise ApiError(409, "build_running", "已经有一次构建在运行")
            folder = chats_folder(request.chats)
            found = pipeline.check(config_path(), request.purpose, request.max_context_tokens, folder)
            if found["problems"]:
                raise ApiError(422, "build_check_failed", "运行前检查没有通过", problems=found["problems"])
            run = pipeline.new_run_dir()
            run.mkdir(parents=True)
            # Shown until the build process writes its own status.
            pipeline.write_status(run, pipeline.new_status(request.purpose))
            pipeline.record_build(run)
            args = ["build-profile", "--purpose", request.purpose, "--output-dir", str(run), "--yes"]
            if request.max_context_tokens is not None:
                args += ["--max-context-tokens", str(request.max_context_tokens)]
            if folder is not None:
                args += ["--csv-folder", str(folder)]
            self.process = self.spawn(run, args)
            self.cancelled = None
        return self.latest()

    def spawn(self, run: Path, args: list[str]) -> subprocess.Popen:
        encrypted = secure.is_encrypted_mode()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if encrypted:
            env[REDACT_ENV] = "1"
        with open(run / pipeline.LOG_FILE, "ab") as log:
            options = {
                "cwd": secure.ROOT,
                "stdin": subprocess.DEVNULL,
                "stdout": log,
                "stderr": subprocess.STDOUT,
            }
            options["env"] = env
            if os.name == "nt":
                options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                options["start_new_session"] = True
            if encrypted:
                # The key of the session that asked for the build, which the request runs with.
                return secure.popen_python(self.script or CLI_SCRIPT, args, **options)
            command = [str(self.script)] if self.script else ["-m", "weclone.cli"]
            return subprocess.Popen([sys.executable, *command, *args], **options)

    def cancel(self):
        run = pipeline.last_build()
        status = pipeline.read_status(run) if run is not None else None
        if self.demo or status is None or status["state"] != "running" or not self.running():
            raise ApiError(409, "build_not_running", "没有正在运行的构建")
        if os.name == "nt":
            raise ApiError(409, "build_cancel_unsupported", "Windows 上暂时不能从网页取消构建")
        pid, pgid = status.get("pid"), status.get("pgid")
        if pgid is None and self.started_here():
            # Started here in a new session, before the build wrote its own status.
            pid = pgid = self.process.pid
        if pgid is None:
            raise ApiError(409, "build_not_running", "没有正在运行的构建")
        try:
            # A build sharing the server's process group, as one started by the same script can, is stopped alone.
            if pgid == os.getpgrp():
                os.kill(pid, signal.SIGTERM)
            else:
                os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        self.cancelled = run

    def versions(self) -> list[dict]:
        current = self.runs.run_dir()
        found = []
        if not self.demo and pipeline.RUNS_DIR.is_dir():
            found = sorted(
                (
                    path
                    for path in pipeline.RUNS_DIR.iterdir()
                    if path.is_dir() and not path.name.startswith(".")
                ),
                reverse=True,
            )
        if current is not None and not any(same(current, run) for run in found):
            found.insert(0, current)
        listed = []
        for run in found:
            status = pipeline.read_status(run)
            complete = run_complete(run)
            # Directories that are neither a profile nor a build, such as ones copied by hand, are left out.
            if status is None and not complete:
                continue
            listed.append(
                {
                    "name": run.name,
                    "purpose": status["purpose"] if status else None,
                    "state": self.state(run, status) if status else "succeeded",
                    "complete": complete,
                    "current": same(run, current),
                    "started_at": status["started_at"] if status else None,
                    "generated_at": generated_at(run),
                    "reviewed_at": (
                        reviewed_at(run / "profile_hierarchy" / "profile_review.sqlite3")
                        if complete
                        else None
                    ),
                    "style": style_view.complete(run / "style"),
                }
            )
        return listed

    def choose(self, name: str):
        if self.demo:
            raise ApiError(409, "demo_runs_fixed", "演示模式只有一份画像")
        latest = pipeline.last_build()
        run = pipeline.RUNS_DIR / name
        if Path(name).name != name or name.startswith("."):
            raise ApiError(404, "run_not_found", "没有这个版本")
        if not run.is_dir():
            if latest is None or latest.name != name:
                raise ApiError(404, "run_not_found", "没有这个版本")
            run = latest
        self.runs.switch(run)

    def delete(self, name: str):
        if self.demo:
            raise ApiError(409, "demo_runs_fixed", "演示模式只有一份画像")
        run = pipeline.RUNS_DIR / name
        if Path(name).name != name or name.startswith(".") or not run.is_dir():
            raise ApiError(404, "run_not_found", "没有这个版本")
        with self.lock:
            if self.running() and same(run, pipeline.last_build()):
                raise ApiError(409, "run_building", "这个版本还在构建")
            self.runs.remove(run)


def install_build(app: FastAPI, runs, *, demo: bool, script: Path | None = None) -> Builds:
    """`script` replaces weclone/cli.py as the program a web build runs, for tests."""
    builds = Builds(runs, demo=demo, script=script)

    @app.get("/api/build/check")
    def build_check(
        purpose: Literal["assistant", "clone"] = "assistant",
        max_context_tokens: int | None = None,
        chats: ChatSource | None = None,
    ):
        return builds.check(purpose, max_context_tokens, chats)

    @app.get("/api/build")
    def build_status():
        builds.settle()
        return builds.latest()

    @app.post("/api/build", status_code=202)
    def build_start(request: BuildStart):
        return builds.start(request)

    @app.delete("/api/build")
    def build_cancel():
        builds.cancel()
        return {"ok": True}

    @app.get("/api/runs")
    def run_list():
        builds.settle()
        return builds.versions()

    @app.put("/api/run")
    def run_choose(request: RunChoice):
        builds.choose(request.name)
        return {"name": request.name}

    @app.delete("/api/runs/{name}")
    def run_delete(name: str):
        builds.delete(name)
        return {"ok": True}

    return builds
