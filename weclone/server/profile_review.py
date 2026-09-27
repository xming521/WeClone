"""Local profile review API and persistent review store."""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from weclone.server.auth import install_auth

Status = Literal["pending", "approved", "rejected"]
ROOT = Path(__file__).resolve().parents[2]


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
        database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
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
            if db.execute("SELECT 1 FROM snapshot").fetchone():
                return
            raw = source.read_bytes()
            data = json.loads(raw)
            seen = set()

            def add(items, parent_id, dim):
                for item in items:
                    node_id = str(uuid4())
                    attribute = "attr" in item
                    db.execute("INSERT INTO nodes VALUES (?,?,?,?,?)", (
                        node_id, parent_id, "attribute" if attribute else "topic", item.get("attr", item.get("name")), dim,
                    ))
                    if attribute:
                        for index in item["fact_indices"]:
                            if index in seen:
                                raise ValueError("Duplicate fact reference in input hierarchy")
                            seen.add(index)
                            fact = data["facts"][index]
                            db.execute("INSERT INTO facts VALUES (?,?,?,?,?,?,1)", (
                                str(uuid4()), node_id, fact["value"], encode(fact), "extracted", "pending",
                            ))
                    else:
                        add(item["items"], node_id, dim)

            for dimension in data["dimensions"]:
                dim = dimension["dim"]
                node_id = f"dimension:{dim}"
                db.execute("INSERT INTO nodes VALUES (?,NULL,'dimension',?,?)", (node_id, dimension["name"], dim))
                add(dimension["groups"], node_id, dim)
            if seen != set(range(len(data["facts"]))):
                raise ValueError("Input hierarchy must reference every fact exactly once")
            db.execute("INSERT INTO snapshot VALUES (1,?,?)", (encode(data), hashlib.sha256(raw).hexdigest()))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def fact(self, db, fact_id):
        row = db.execute("""SELECT f.*, n.name AS attr, n.dim, n.parent_id AS group_id
                            FROM facts f JOIN nodes n ON f.node_id=n.id WHERE f.id=?""", (fact_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "画像记录不存在")
        original = json.loads(row["original"])
        return {
            **original, **{key: row[key] for key in ("id", "node_id", "value", "attr", "dim", "group_id", "origin", "status", "version")},
            "original": original, "source_ids": original.get("source_ids", []),
        }

    def record(self, db, fact_id, action, before):
        after = self.fact(db, fact_id)
        db.execute("INSERT INTO history(fact_id,at,action,before_json,after_json) VALUES (?,?,?,?,?)", (
            fact_id, datetime.now(timezone.utc).isoformat(), action, encode(before) if before else None, encode(after),
        ))
        return after

    def attribute(self, db, request):
        group = db.execute("SELECT * FROM nodes WHERE id=? AND kind IN ('dimension','topic')", (request.group_id,)).fetchone()
        if group is None:
            raise HTTPException(422, "请选择已有维度或主题")
        row = db.execute("SELECT id FROM nodes WHERE parent_id=? AND kind='attribute' AND name=?", (request.group_id, request.attr)).fetchone()
        if row:
            return row["id"]
        node_id = str(uuid4())
        db.execute("INSERT INTO nodes VALUES (?,?,'attribute',?,?)", (node_id, request.group_id, request.attr, group["dim"]))
        return node_id

    def create(self, request):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            node_id = self.attribute(db, request)
            fact_id = str(uuid4())
            original = {"value": request.value, "attr": request.attr, "source_ids": []}
            db.execute("INSERT INTO facts VALUES (?,?,?,?,?,?,1)", (
                fact_id, node_id, request.value, encode(original), "manual", "approved" if request.approve else "pending",
            ))
            return self.record(db, fact_id, "create", None)

    def edit(self, fact_id, request):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            before = self.fact(db, fact_id)
            if before["version"] != request.version:
                raise HTTPException(409, "记录已被修改，请刷新数据后核对并重试")
            changed = any(before[key] != getattr(request, key) for key in ("value", "attr", "group_id"))
            status = "approved" if request.approve else "pending" if changed else before["status"]
            if not changed and status == before["status"]:
                return before
            node_id = self.attribute(db, request)
            db.execute("UPDATE facts SET node_id=?,value=?,status=?,version=version+1 WHERE id=?", (
                node_id, request.value, status, fact_id,
            ))
            return self.record(db, fact_id, "edit", before)

    def review(self, request):
        if len({item.id for item in request.items}) != len(request.items):
            raise HTTPException(422, "批量操作不能包含重复记录")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            before = [self.fact(db, item.id) for item in request.items]
            if any(fact["version"] != item.version for fact, item in zip(before, request.items)):
                raise HTTPException(409, "记录已被修改，整批操作未保存，请刷新后重试")
            for fact in before:
                if fact["status"] != request.status:
                    db.execute("UPDATE facts SET status=?,version=version+1 WHERE id=?", (request.status, fact["id"]))
                    self.record(db, fact["id"], "review", fact)
            return {"updated": len(before)}

    def history(self, fact_id):
        with self.connect() as db:
            self.fact(db, fact_id)
            return [{"at": row["at"], "action": row["action"], "before": json.loads(row["before_json"]) if row["before_json"] else None,
                     "after": json.loads(row["after_json"])} for row in db.execute(
                         "SELECT * FROM history WHERE fact_id=? ORDER BY seq DESC", (fact_id,))]

    def profile(self, approved_only=False):
        with self.connect() as db:
            db.execute("BEGIN")
            snapshot = db.execute("SELECT * FROM snapshot").fetchone()
            source_data = json.loads(snapshot["data"])
            nodes = [dict(row) for row in db.execute("SELECT * FROM nodes ORDER BY rowid")]
            facts = [self.fact(db, row["id"]) for row in db.execute(
                "SELECT id FROM facts" + (" WHERE status='approved'" if approved_only else "") + " ORDER BY rowid")]
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
                items = [result for child in children.get(node["id"], []) if (result := expand(child)) is not None]
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
                facts = [{key: fact[key] for key in ("id", "dim", "attr", "value", "source_ids", "origin")} for fact in facts]
            result = {"dimensions": dimensions, "facts": facts, "sources": sources}
            if not approved_only:
                result["locations"] = [node for node in nodes if node["kind"] != "attribute"]
                result["snapshot_sha256"] = snapshot["sha256"]
            return result


def create_app(database: Path | None = None, source: Path | None = None, static_dir: Path | None = None,
               *, inference_router: APIRouter | None = None):
    directory = ROOT / "dataset/res_csv/agent/memory_organization"
    store = ReviewStore(database or directory / "profile_review.sqlite3", source or directory / "profile_hierarchy.json")
    app = FastAPI(title="WeClone")
    install_auth(app, store.database)
    if inference_router is not None:
        app.include_router(inference_router)

    @app.get("/api/profile")
    def profile():
        return store.profile()

    @app.get("/api/avatar-profile")
    def avatar(download: bool = False):
        headers = {"Content-Disposition": 'attachment; filename="avatar_profile.json"'} if download else {}
        return JSONResponse(store.profile(approved_only=True), headers=headers)

    @app.get("/api/facts/{fact_id}/history")
    def history(fact_id: str):
        return store.history(fact_id)

    @app.post("/api/facts", status_code=201)
    def create(request: FactWrite):
        return store.create(request)

    @app.patch("/api/facts/{fact_id}")
    def edit(fact_id: str, request: FactEdit):
        return store.edit(fact_id, request)

    @app.post("/api/reviews/batch")
    def review(request: ReviewBatch):
        return store.review(request)

    @app.get("/api/{path:path}", include_in_schema=False)
    def unknown_api(path: str):
        raise HTTPException(404, "接口不存在")

    static_dir = static_dir or ROOT / "web/dist"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")
    return app
