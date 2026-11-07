"""Local profile review API and persistent review store."""

import hashlib
import json
import shutil
import sqlite3
import threading
import tomllib
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from weclone.data.agent import profile_pipeline
from weclone.server import style_view
from weclone.server.assistant import install_assistant
from weclone.server.auth import AuthStore, install_auth
from weclone.server.bots import install_bots
from weclone.server.chat_upload import install_upload
from weclone.server.errors import ApiError, error_response, install_errors, security_response
from weclone.server.profile_build import install_build, reviewed_at, run_complete, same
from weclone.server.profile_intro import profile_intro
from weclone.server.profile_summary import (
    GENERATING,
    Level,
    SummaryEdit,
    SummaryGenerate,
    SummaryReview,
    SummaryStore,
)
from weclone.server.source_chat import source_chat
from weclone.utils import secure_storage as secure

Status = Literal["pending", "approved", "rejected"]
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "dist"
AUTH_FILE = "web.sqlite3"


def newest_run() -> Path | None:
    runs = profile_pipeline.RUNS_DIR
    return next((run for run in sorted(runs.glob("*/"), reverse=True) if run_complete(run)), None)


def review_paths(database: Path | None = None, source: Path | None = None) -> tuple[Path, Path]:
    if source is None:
        run = newest_run()
        source = (run / "profile_hierarchy" if run else profile_pipeline.RUNS_DIR) / "profile_hierarchy.json"
    return database or source.parent / "profile_review.sqlite3", source


def weclone_version():
    # A checkout is installed editable, whose package metadata keeps the version of the first install.
    try:
        return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        pass
    try:
        return metadata.version("WeClone")
    except metadata.PackageNotFoundError:
        return None


def encode(value):
    return json.dumps(value, ensure_ascii=False)


class FactWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str
    attr: str
    group_id: str
    approve: bool = False

    @field_validator("value", "attr")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("内容和属性名称不能为空")
        return value.strip()


class FactEdit(FactWrite):
    version: int = Field(ge=1)


class ReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    version: int = Field(ge=1)


class ReviewBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[ReviewItem] = Field(min_length=1)
    status: Status


class ReviewStore:
    def __init__(self, database: Path, source: Path):
        self.database = database
        self.source = source

    def initialize(self, db):
        # Sensitive sources are read only inside an authenticated request.
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='snapshot'").fetchone()
        if exists and db.execute("SELECT 1 FROM snapshot").fetchone():
            return
        if not exists:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS snapshot (id INTEGER PRIMARY KEY CHECK(id=1), data TEXT NOT NULL, sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY, parent_id TEXT REFERENCES nodes(id), kind TEXT NOT NULL,
                    name TEXT NOT NULL, dim INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY, node_id TEXT NOT NULL REFERENCES nodes(id),
                    value TEXT NOT NULL, original TEXT NOT NULL, origin TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','approved','rejected')),
                    version INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS history (
                    seq INTEGER PRIMARY KEY, fact_id TEXT NOT NULL REFERENCES facts(id),
                    at TEXT NOT NULL, action TEXT NOT NULL, before_json TEXT, after_json TEXT NOT NULL
                );
            """)
        db.execute("BEGIN IMMEDIATE")
        if not db.execute("SELECT 1 FROM snapshot").fetchone():
            raw = secure.read_bytes(self.source)
            data = json.loads(raw)
            seen = set()

            def add(items, parent_id, dim):
                for item in items:
                    node_id = str(uuid4())
                    attribute = "attr" in item
                    db.execute(
                        "INSERT INTO nodes VALUES (?,?,?,?,?)",
                        (
                            node_id,
                            parent_id,
                            "attribute" if attribute else "topic",
                            item.get("attr", item.get("name")),
                            dim,
                        ),
                    )
                    if attribute:
                        for index in item["fact_indices"]:
                            if index in seen:
                                raise ValueError("Duplicate fact reference in input hierarchy")
                            seen.add(index)
                            fact = data["facts"][index]
                            db.execute(
                                "INSERT INTO facts VALUES (?,?,?,?,?,?,1)",
                                (
                                    str(uuid4()),
                                    node_id,
                                    fact["value"],
                                    encode(fact),
                                    "extracted",
                                    "pending",
                                ),
                            )
                    else:
                        add(item["items"], node_id, dim)

            for dimension in data["dimensions"]:
                dim = dimension["dim"]
                node_id = f"dimension:{dim}"
                db.execute(
                    "INSERT INTO nodes VALUES (?,NULL,'dimension',?,?)", (node_id, dimension["name"], dim)
                )
                add(dimension["groups"], node_id, dim)
            if seen != set(range(len(data["facts"]))):
                raise ValueError("Input hierarchy must reference every fact exactly once")
            db.execute("INSERT INTO snapshot VALUES (1,?,?)", (encode(data), hashlib.sha256(raw).hexdigest()))

    @contextmanager
    def connect(self):
        if not secure.file_exists(self.database) and not secure.file_exists(self.source):
            raise FileNotFoundError(self.source)
        with secure.encrypted_sqlite(self.database) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            self.initialize(db)
            # Initialization and the requested edit have separate transactions.
            db.commit()
            yield db

    def fact(self, db, fact_id):
        row = db.execute(
            """SELECT f.*, n.name AS attr, n.dim, n.parent_id AS group_id
                            FROM facts f JOIN nodes n ON f.node_id=n.id WHERE f.id=?""",
            (fact_id,),
        ).fetchone()
        if row is None:
            raise ApiError(404, "fact_not_found", "画像记录不存在")
        original = json.loads(row["original"])
        return {
            **original,
            **{
                key: row[key]
                for key in (
                    "id",
                    "node_id",
                    "value",
                    "attr",
                    "dim",
                    "group_id",
                    "origin",
                    "status",
                    "version",
                )
            },
            "original": original,
            "source_ids": original.get("source_ids", []),
        }

    def record(self, db, fact_id, action, before):
        after = self.fact(db, fact_id)
        db.execute(
            "INSERT INTO history(fact_id,at,action,before_json,after_json) VALUES (?,?,?,?,?)",
            (
                fact_id,
                datetime.now(timezone.utc).isoformat(),
                action,
                encode(before) if before else None,
                encode(after),
            ),
        )
        return after

    def attribute(self, db, request):
        group = db.execute(
            "SELECT * FROM nodes WHERE id=? AND kind IN ('dimension','topic')", (request.group_id,)
        ).fetchone()
        if group is None:
            raise ApiError(422, "location_invalid", "请选择已有维度或主题")
        row = db.execute(
            "SELECT id FROM nodes WHERE parent_id=? AND kind='attribute' AND name=?",
            (request.group_id, request.attr),
        ).fetchone()
        if row:
            return row["id"]
        node_id = str(uuid4())
        db.execute(
            "INSERT INTO nodes VALUES (?,?,'attribute',?,?)",
            (node_id, request.group_id, request.attr, group["dim"]),
        )
        return node_id

    def create(self, request):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            node_id = self.attribute(db, request)
            fact_id = str(uuid4())
            original = {"value": request.value, "attr": request.attr, "source_ids": []}
            db.execute(
                "INSERT INTO facts VALUES (?,?,?,?,?,?,1)",
                (
                    fact_id,
                    node_id,
                    request.value,
                    encode(original),
                    "manual",
                    "approved" if request.approve else "pending",
                ),
            )
            return self.record(db, fact_id, "create", None)

    def edit(self, fact_id, request):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            before = self.fact(db, fact_id)
            if before["version"] != request.version:
                raise ApiError(409, "fact_changed", "记录已被修改，请刷新数据后核对并重试")
            changed = any(before[key] != getattr(request, key) for key in ("value", "attr", "group_id"))
            status = "approved" if request.approve else "pending" if changed else before["status"]
            if not changed and status == before["status"]:
                return before
            node_id = self.attribute(db, request)
            db.execute(
                "UPDATE facts SET node_id=?,value=?,status=?,version=version+1 WHERE id=?",
                (
                    node_id,
                    request.value,
                    status,
                    fact_id,
                ),
            )
            return self.record(db, fact_id, "edit", before)

    def review(self, request):
        if len({item.id for item in request.items}) != len(request.items):
            raise ApiError(422, "duplicate_facts", "批量操作不能包含重复记录")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            before = [self.fact(db, item.id) for item in request.items]
            if any(fact["version"] != item.version for fact, item in zip(before, request.items)):
                raise ApiError(409, "batch_changed", "记录已被修改，整批操作未保存，请刷新后重试")
            for fact in before:
                if fact["status"] != request.status:
                    db.execute(
                        "UPDATE facts SET status=?,version=version+1 WHERE id=?", (request.status, fact["id"])
                    )
                    self.record(db, fact["id"], "review", fact)
            return {"updated": len(before)}

    def history(self, fact_id):
        with self.connect() as db:
            self.fact(db, fact_id)
            return [
                {
                    "at": row["at"],
                    "action": row["action"],
                    "before": json.loads(row["before_json"]) if row["before_json"] else None,
                    "after": json.loads(row["after_json"]),
                }
                for row in db.execute("SELECT * FROM history WHERE fact_id=? ORDER BY seq DESC", (fact_id,))
            ]

    def chat(self, source_id):
        with self.connect() as db:
            data = json.loads(db.execute("SELECT data FROM snapshot WHERE id=1").fetchone()["data"])
            source = data.get("sources", {}).get(source_id)
        if not isinstance(source, dict):
            raise ApiError(404, "source_not_found", "来源记录不存在")
        return source_chat(source_id, source)

    def run_info(self):
        """Which build-profile run the server shows and when it was made and last reviewed."""
        run = self.source.parent.parent
        source = secure.resolve_path(self.source)
        try:
            with self.connect() as db:
                reviewed = db.execute("SELECT MAX(at) FROM history").fetchone()[0]
        except FileNotFoundError:
            reviewed = None
        return {
            "name": run.name,
            "path": str(run),
            "generated_at": (
                datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat()
                if source.is_file()
                else None
            ),
            "reviewed_at": reviewed,
            "style": style_view.complete(run / "style"),
        }

    def intro(self):
        profile = self.profile()
        with self.connect() as db:
            snapshot = json.loads(db.execute("SELECT data FROM snapshot WHERE id=1").fetchone()["data"])
        # Demo profiles carry their message counts beside the facts.
        return profile_intro({**profile, "_demo": snapshot.get("_demo")})

    def profile(self, approved_only=False):
        with self.connect() as db:
            db.execute("BEGIN")
            snapshot = db.execute("SELECT * FROM snapshot").fetchone()
            source_data = json.loads(snapshot["data"])
            nodes = [dict(row) for row in db.execute("SELECT * FROM nodes ORDER BY rowid")]
            facts = [
                self.fact(db, row["id"])
                for row in db.execute(
                    "SELECT id FROM facts"
                    + (" WHERE status='approved'" if approved_only else "")
                    + " ORDER BY rowid"
                )
            ]
            indices = {}
            for index, fact in enumerate(facts):
                indices.setdefault(fact["node_id"], []).append(index)
            children = {}
            for node in nodes:
                children.setdefault(node["parent_id"], []).append(node)

            def expand(node):
                if node["kind"] == "attribute":
                    refs = indices.get(node["id"], [])
                    return {"id": node["id"], "attr": node["name"], "fact_indices": refs} if refs else None
                items = [
                    result for child in children.get(node["id"], []) if (result := expand(child)) is not None
                ]
                if approved_only and not items:
                    return None
                return {"id": node["id"], "name": node["name"], "items": items}

            dimensions = []
            for node in children.get(None, []):
                group = expand(node)
                if group:
                    # Direct attributes are represented as a group for the existing hierarchy format.
                    items = group["items"]
                    direct = [item for item in items if "attr" in item]
                    groups = [item for item in items if "attr" not in item]
                    if direct:
                        groups.append({"id": f"{node['id']}:direct", "name": "直属属性", "items": direct})
                    dimensions.append({"dim": node["dim"], "name": node["name"], "groups": groups})
            used = {sid for fact in facts for sid in fact["source_ids"]}
            sources = {sid: source_data["sources"][sid] for sid in used if sid in source_data["sources"]}
            if approved_only:
                facts = [
                    {key: fact[key] for key in ("id", "dim", "attr", "value", "source_ids", "origin")}
                    for fact in facts
                ]
            result = {"dimensions": dimensions, "facts": facts, "sources": sources}
            if not approved_only:
                result["locations"] = [node for node in nodes if node["kind"] != "attribute"]
                result["snapshot_sha256"] = snapshot["sha256"]
                # Each dimension is grouped by domain, then topic. The domain view of the review UI
                # regroups that tree, taking order and descriptions from here.
                result["domains"] = [
                    {
                        "name": domain["name"],
                        "description": domain["description"],
                        "topics": [topic["name"] for topic in domain["topics"]],
                    }
                    for domain in source_data["domains"]
                ]
            return result


class Runs:
    """The build-profile run the server shows. Switching replaces its review and summary stores as a pair, and a
    request takes the pair once, so one that is running during a switch finishes on the run it started with."""

    def __init__(self, settings: AuthStore, database: Path | None = None, source: Path | None = None):
        self.settings = settings
        self.lock = threading.Lock()
        self.pairs: dict[Path, tuple[ReviewStore, SummaryStore]] = {}
        self.pair: tuple[ReviewStore, SummaryStore] | None = None
        # Whether the server was started on this run, as by `server --run`.
        self.chosen = source is not None
        if source is None:
            saved = settings.setting("current_run")
            run = Path(saved) if saved and run_complete(Path(saved)) else newest_run()
            if run is not None:
                database, source = review_paths(None, run / "profile_hierarchy" / "profile_hierarchy.json")
        if source is not None:
            self.pair = self.stores(database or source.parent / "profile_review.sqlite3", source)

    def stores(self, database: Path, source: Path) -> tuple[ReviewStore, SummaryStore]:
        if source not in self.pairs:
            store = ReviewStore(database, source)
            self.pairs[source] = (store, SummaryStore(store))
        return self.pairs[source]

    def current(self) -> tuple[ReviewStore, SummaryStore]:
        pair = self.pair
        if pair is None:
            raise FileNotFoundError(profile_pipeline.RUNS_DIR)
        return pair

    def store(self) -> ReviewStore:
        return self.current()[0]

    def summaries(self) -> SummaryStore:
        return self.current()[1]

    def run_dir(self) -> Path | None:
        pair = self.pair
        return pair[0].source.parent.parent if pair else None

    def reviewed(self) -> bool:
        pair = self.pair
        return pair is not None and reviewed_at(pair[0].database) is not None

    def switch(self, run: Path):
        if not run_complete(run):
            raise ApiError(409, "run_incomplete", "这个版本没有生成完整的画像")
        with self.lock:
            self.pair = self.stores(*review_paths(None, run / "profile_hierarchy" / "profile_hierarchy.json"))
            self.settings.save_setting("current_run", str(run.resolve()))

    def remove(self, run: Path):
        """Removes a run's folder, or only the link when it is linked from elsewhere, with the stores opened on it.
        Not the run shown, nor one that may be writing its summaries."""
        with self.lock:
            if same(self.run_dir(), run):
                raise ApiError(409, "run_current", "正在显示的版本不能删除")
            opened = [
                source for source, (store, _) in self.pairs.items() if same(store.source.parent.parent, run)
            ]
            if opened and GENERATING.locked():
                raise ApiError(409, "summary_generating", "正在生成板块总结，请稍候")
            for source in opened:
                del self.pairs[source]
            if run.is_symlink():
                run.unlink()
            else:
                shutil.rmtree(run)


def create_app(
    database: Path | None = None,
    source: Path | None = None,
    static_dir: Path | None = None,
    *,
    inference_router: APIRouter | None = None,
    demo_password: str | None = None,
    build_script: Path | None = None,
):
    """Serves the run of `source`, or else the run last opened on the web, or else the newest one.
    `build_script` replaces weclone/cli.py as the program a web build runs, for tests."""
    app = FastAPI(title="WeClone")
    install_errors(app)
    # Sessions and settings outlive the run being shown.
    auth = AuthStore(secure.configure()["state_dir"] / AUTH_FILE)
    install_auth(app, auth, demo_password)
    install_bots(app, inference=inference_router is not None)
    runs = Runs(auth, database, source)
    builds = install_build(app, runs, demo=demo_password is not None, script=build_script)
    install_upload(app, builds, demo=demo_password is not None)

    @app.exception_handler(secure.SecurityError)
    async def security_error(request, exc):
        return security_response(exc, 401 if isinstance(exc, secure.StorageLocked) else 409)

    @app.exception_handler(FileNotFoundError)
    async def missing_profile(request, exc):
        return error_response(404, "profile_missing", "画像数据尚未生成，请先完成抽取和整理")

    if inference_router is not None:
        app.include_router(inference_router)

    @app.get("/api/profile")
    def profile():
        builds.settle()
        return runs.store().profile()

    @app.get("/api/run")
    def run_info():
        builds.settle()
        demo = demo_password is not None
        try:
            info = runs.store().run_info()
        except FileNotFoundError:
            info = {"name": None, "path": None, "generated_at": None, "reviewed_at": None, "style": False}
        # The demo is open to anyone with its published password: the server's directories stay out of it.
        if demo:
            info["path"] = None
        return {**info, "demo": demo, "version": weclone_version()}

    @app.get("/api/profile/intro")
    def intro():
        return runs.store().intro()

    @app.get("/api/avatar-profile")
    def avatar(download: bool = False):
        headers = {"Content-Disposition": 'attachment; filename="avatar_profile.json"'} if download else {}
        return JSONResponse(runs.store().profile(approved_only=True), headers=headers)

    @app.get("/api/facts/{fact_id}/history")
    def history(fact_id: str):
        return runs.store().history(fact_id)

    @app.get("/api/sources/{source_id}/chat")
    def chat(source_id: str):
        return runs.store().chat(source_id)

    @app.post("/api/facts", status_code=201)
    def create(request: FactWrite):
        return runs.store().create(request)

    @app.patch("/api/facts/{fact_id}")
    def edit(fact_id: str, request: FactEdit):
        return runs.store().edit(fact_id, request)

    @app.post("/api/reviews/batch")
    def review(request: ReviewBatch):
        return runs.store().review(request)

    @app.get("/api/summaries")
    def summary_view(level: Level = "full"):
        return runs.summaries().view(level)

    @app.post("/api/summaries/generate")
    def summary_generate(request: SummaryGenerate):
        return runs.summaries().generate(request.level)

    @app.patch("/api/summaries")
    def summary_edit(request: SummaryEdit):
        return runs.summaries().edit(request)

    @app.post("/api/summaries/review")
    def summary_review(request: SummaryReview):
        return runs.summaries().review(request)

    install_assistant(app, runs)

    @app.get("/api/style")
    def style():
        # The style of the run the profile comes from. Read per request: the files are read with the session key,
        # and a style stage that finishes after the server started shows up without a restart.
        run = runs.store().source.parent.parent / "style"
        if not style_view.complete(run):
            raise ApiError(
                404, "style_missing", "这个版本没有抽取语言风格，要用 AI 分身，请在构建画像时选择 AI 分身"
            )
        return style_view.build_view(run)

    @app.get("/api/{path:path}", include_in_schema=False)
    def unknown_api(path: str):
        raise ApiError(404, "api_not_found", "接口不存在")

    static_dir = static_dir or DEFAULT_STATIC_DIR
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")
    return app
