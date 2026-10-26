"""Domain summaries of the reviewed profile, one per life domain and level, written by the LLM from the domain's facts.

The full level is B's own summary of every fact still true, written by build-profile as its last stage. The public
and normal levels are shorter versions for a personal assistant to keep in context, written from identity, goals,
conditions and preferences at or below that privacy level. A summary keeps the hash of the fact lines it was written
from. Once those facts change it is stale and leaves the export until it is regenerated and approved again. Each
summary is stored as soon as it comes back, so a model that fails on some domains still leaves the others done.
"""

import argparse
import contextvars
import hashlib
import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from weclone.prompts.memory_organization import dimension_name
from weclone.prompts.profile_summary import SUMMARY_CHARS, SUMMARY_SCHEMA, summary_prompt
from weclone.utils import secure_storage as secure
from weclone.utils.log import logger

Level = Literal["full", "normal", "public"]
READERS = {"full": "owner", "normal": "assistant", "public": "assistant"}
# Privacy levels an assistant's summary draws on. Private facts never go into its always-on text.
ASSISTANT_LEVELS = {"normal": ("public", "normal"), "public": ("public",)}
# Identity, goals, conditions and preferences. Plans and commitments date too fast for an assistant's always-on text.
ASSISTANT_DIMS = (1, 3, 4, 8)
# Facts of one domain the model reads, most supported first: a few sentences cannot use more.
SUMMARY_FACTS = 60
# Bounds one prompt of SUMMARY_FACTS facts. A guard against runaway input, not the model's context window.
INPUT_TOKENS = 16_000
TITLE_CHARS = 12
# The model counts Chinese characters loosely, so only clear overruns of the length it is asked for are rejected.
SUMMARY_MAX_CHARS = {level: SUMMARY_CHARS[reader] * 3 // 2 for level, reader in READERS.items()}
# Facts the export can list under each summary.
EXTRA_FACTS = 10
EXPORT_HEADER = (
    "以下是我的背景资料，由 WeClone 根据我的聊天记录整理，可能已经过时，涉及重要决定时请先跟我确认。"
)
TABLE = """CREATE TABLE IF NOT EXISTS domain_summaries (
    level TEXT NOT NULL CHECK(level IN ('full','normal','public')), domain TEXT NOT NULL,
    title TEXT NOT NULL, summary TEXT NOT NULL, input_sha256 TEXT NOT NULL, fact_ids TEXT NOT NULL,
    origin TEXT NOT NULL CHECK(origin IN ('generated','edited')),
    status TEXT NOT NULL CHECK(status IN ('pending','approved','rejected')),
    version INTEGER NOT NULL, at TEXT NOT NULL, PRIMARY KEY (level, domain)
)"""
# One generation at a time, held until its summaries are stored. Its model calls run outside any database connection.
GENERATING = threading.Lock()


def clean(text: str) -> str:
    return " ".join(text.split())


def month_span(fact: dict, sources: dict) -> tuple[str, str]:
    """First and last month the fact came up in chat, empty without source times."""
    months = sorted(
        str(sources[sid]["sample_time"])[:7]
        for sid in fact.get("source_ids", [])
        if sources.get(sid, {}).get("sample_time")
    )
    return (months[0], months[-1]) if months else ("", "")


def in_scope(fact: dict, level: str) -> bool:
    if fact["status"] == "rejected" or fact.get("time_scope") == "superseded":
        return False
    return level == "full" or (
        fact["dim"] in ASSISTANT_DIMS and fact.get("sensitivity") in ASSISTANT_LEVELS[level]
    )


def domain_inputs(profile: dict, level: str) -> list[dict]:
    """For each domain of the review store's profile, in order: the facts a summary at `level` is written from,
    as prompt lines with their hash and as export lines, and the last month any of them came up."""
    members: dict[str, list[int]] = {domain["name"]: [] for domain in profile["domains"]}

    def collect(items: list[dict], domain: str):
        for item in items:
            if "attr" in item:
                members.setdefault(domain, []).extend(item["fact_indices"])
            else:
                collect(item["items"], domain)

    for dimension in profile["dimensions"]:
        for group in dimension["groups"]:
            # Attributes placed directly on a dimension belong to no domain.
            if not group["id"].endswith(":direct"):
                collect(group["items"], group["name"])
    sources = profile.get("sources", {})
    inputs = []
    for domain, indices in members.items():
        rows = []
        for index in indices:
            fact = profile["facts"][index]
            if not in_scope(fact, level):
                continue
            first, last = month_span(fact, sources)
            rows.append(
                {"fact": fact, "support": fact.get("support_count") or 1, "first": first, "last": last}
            )
        # Most supported first, then the most recent.
        rows.sort(key=lambda row: row["last"], reverse=True)
        rows.sort(key=lambda row: row["support"], reverse=True)
        used = rows[:SUMMARY_FACTS]
        lines, extra = [], []
        for row in used:
            fact = row["fact"]
            text = f"{fact['attr']}：{clean(fact['value'])}"
            span = row["first"] if row["first"] == row["last"] else f"{row['first']}~{row['last']}"
            lines.append(
                f"{text}（{dimension_name(fact['dim'])}，{row['support']} 次{'，' + span if span else ''}）"
            )
            extra.append(f"{text}（{span}）" if span else text)
        inputs.append(
            {
                "domain": domain,
                "facts": len(rows),
                "fact_ids": [row["fact"]["id"] for row in used],
                "lines": lines,
                "extra": extra,
                "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
                "as_of": max((row["last"] for row in used), default=""),
            }
        )
    return inputs


def due(item: dict, rows: dict[str, dict]) -> bool:
    """Generation writes domains that have facts and no summary, or a summary of facts that have since changed.
    Domains left out of export stay as they are."""
    row = rows.get(item["domain"])
    if not item["lines"]:
        return False
    return row is None or (row["status"] != "rejected" and row["input_sha256"] != item["sha256"])


def validate_summary(payload, level: str) -> dict:
    from weclone.data.agent.llm_profile_hierarchy import PEER_ID

    if not isinstance(payload, dict) or set(payload) != {"title", "summary"}:
        raise ValueError("Summary requires title and summary")
    if not all(isinstance(value, str) for value in payload.values()):
        raise ValueError("Title and summary must be strings")
    title, summary = clean(payload["title"]), clean(payload["summary"])
    if not 0 < len(title) <= TITLE_CHARS or not 0 < len(summary) <= SUMMARY_MAX_CHARS[level]:
        raise ValueError("Title or summary is empty or too long")
    # Peer ids only mean something next to the chats they number.
    if PEER_ID.search(title) or PEER_ID.search(summary):
        raise ValueError("Summary must not contain chat peer ids")
    return {"title": title, "summary": summary}


def export_text(chosen: list[tuple[dict, dict]], extra: int) -> str:
    """Approved, current summaries as text to paste into an assistant, each with up to `extra` of its facts."""
    lines = []
    for item, row in chosen:
        as_of = f"（截至 {item['as_of']}）" if item["as_of"] else ""
        lines.append(f"- {row['title']}{as_of}：{row['summary']}")
        lines += [f"  - {text}" for text in item["extra"][:extra]]
    return f"{EXPORT_HEADER}\n\n" + "\n".join(lines) if lines else ""


def summary_runner(output_dir: Path):
    """build-profile's LLM task runner with the configured model, checkpointed in output_dir."""
    from weclone.data.agent.organize_memories import LLMTasks

    return LLMTasks(
        argparse.Namespace(
            config_path=Path(os.environ.get("WECLONE_CONFIG_PATH", "settings.jsonc")),
            llm_provider=None,
            max_context_tokens=INPUT_TOKENS,
            token_encoding="o200k_base",
            tokenizer_file=None,
            output_dir=output_dir,
        )
    )


class SummaryGenerate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: Level


class SummaryEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: Level
    domain: str
    title: str
    summary: str
    version: int = Field(ge=1)
    approve: bool = False

    @field_validator("title", "summary")
    @classmethod
    def cleaned(cls, value):
        return clean(value)

    @model_validator(mode="after")
    def bounded(self):
        limit = SUMMARY_MAX_CHARS[self.level]
        if not 0 < len(self.title) <= TITLE_CHARS or not 0 < len(self.summary) <= limit:
            raise ValueError(f"标题和总结不能为空，标题不超过 {TITLE_CHARS} 字，总结不超过 {limit} 字")
        return self


class SummaryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain: str
    version: int = Field(ge=1)


class SummaryReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: Level
    items: list[SummaryItem] = Field(min_length=1)
    status: Literal["pending", "approved", "rejected"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def public_row(row: dict, stale: bool) -> dict:
    return {**{key: row[key] for key in ("title", "summary", "origin", "status", "version")}, "stale": stale}


class SummaryStore:
    """Summary rows kept in the review store's database, next to the facts they are written from."""

    def __init__(self, store):
        self.store = store
        # Why the last generation of each level left domains without a summary.
        self.failures: dict[str, str] = {}

    @contextmanager
    def connect(self):
        with self.store.connect() as db:
            db.execute(TABLE)
            yield db

    def state(self, level: str) -> tuple[list[dict], dict[str, dict]]:
        inputs = domain_inputs(self.store.profile(), level)
        with self.connect() as db:
            rows = {
                row["domain"]: dict(row)
                for row in db.execute("SELECT * FROM domain_summaries WHERE level=?", (level,))
            }
        return inputs, rows

    def view(self, level: str, extra: int = 0) -> dict:
        inputs, rows = self.state(level)
        domains, chosen = [], []
        for item in inputs:
            row = rows.get(item["domain"])
            stale = row is not None and row["input_sha256"] != item["sha256"]
            if row is not None and row["status"] == "approved" and not stale:
                chosen.append((item, row))
            domains.append(
                {
                    "domain": item["domain"],
                    "facts": item["facts"],
                    "as_of": item["as_of"],
                    "summary": public_row(row, stale) if row is not None else None,
                }
            )
        return {
            "level": level,
            "generating": GENERATING.locked(),
            # Shown while the domains it is about still wait for a summary.
            "error": self.failures.get(level, "") if any(due(item, rows) for item in inputs) else "",
            "limits": {"title": TITLE_CHARS, "summary": SUMMARY_MAX_CHARS[level], "facts": EXTRA_FACTS},
            "domains": domains,
            "text": export_text(chosen, extra),
        }

    def generate(self, level: str, *, wait: bool = False) -> dict:
        """Generate the domains that are due. The server runs the model calls in a background thread and the view
        shows when they end. build-profile waits for them."""
        if not GENERATING.acquire(blocking=False):
            raise HTTPException(409, "正在生成领域总结，请稍候")
        started = False
        try:
            inputs, rows = self.state(level)
            todo = [item for item in inputs if due(item, rows)]
            if todo:
                self.failures.pop(level, None)
                if wait:
                    started = True
                    self.run(level, todo, rows)
                else:
                    # The thread reads and writes storage with this request's key.
                    context = contextvars.copy_context()
                    threading.Thread(
                        target=context.run, args=(self.run, level, todo, rows), daemon=True
                    ).start()
                    started = True
            return {"generating": len(todo)}
        finally:
            if not started:
                GENERATING.release()

    def run(self, level: str, todo: list[dict], rows: dict[str, dict]):
        """One generation. Releases GENERATING when it ends."""
        try:
            tasks = [
                {
                    "stage": "domain_summary",
                    "schema": SUMMARY_SCHEMA,
                    "prompt": summary_prompt(READERS[level], item["lines"]),
                }
                for item in todo
            ]
            items = {id(task): item for task, item in zip(tasks, todo)}
            stored: dict[int, dict] = {}

            def accept(payload, task):
                # Called for each answer as it comes back, and again for every task once all have.
                if id(task) not in stored:
                    result = validate_summary(payload, level)
                    self.store_result(level, rows, items[id(task)], result)
                    stored[id(task)] = result
                return stored[id(task)]

            try:
                # Stored summaries are the checkpoint. The runner's own lasts one generation, so it never holds a
                # summary the store lacks.
                with secure.runtime_directory("weclone-summary-") as directory:
                    runner = summary_runner(directory)
                    try:
                        runner.run(tasks, accept)
                    finally:
                        runner.close()
            except Exception as exc:
                logger.warning(f"领域总结生成失败：{type(exc).__name__}: {exc}")
                failure = f"{len(todo) - len(stored)} 个领域没有生成成功，下次生成只补这些领域。"
                # A bare RuntimeError is LLMTasks giving up after retries, pointing to a checkpoint that is gone.
                if type(exc) is not RuntimeError:
                    failure += f"原因：{str(exc) or type(exc).__name__}"
                self.failures[level] = failure
        finally:
            GENERATING.release()

    def store_result(self, level: str, rows: dict[str, dict], item: dict, result: dict):
        values = (result["title"], result["summary"], item["sha256"], json.dumps(item["fact_ids"]), now())
        row = rows.get(item["domain"])
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # A summary the owner changed while the model ran keeps the owner's change.
            if row is None:
                db.execute(
                    "INSERT OR IGNORE INTO domain_summaries(level,domain,title,summary,input_sha256,fact_ids,"
                    "at,origin,status,version) VALUES (?,?,?,?,?,?,?,'generated','pending',1)",
                    (level, item["domain"], *values),
                )
            else:
                db.execute(
                    "UPDATE domain_summaries SET title=?,summary=?,input_sha256=?,fact_ids=?,at=?,"
                    "origin='generated',status='pending',version=version+1 "
                    "WHERE level=? AND domain=? AND version=?",
                    (*values, level, item["domain"], row["version"]),
                )

    def edit(self, request: SummaryEdit) -> dict:
        inputs, rows = self.state(request.level)
        current = {item["domain"]: item["sha256"] for item in inputs}
        row = rows.get(request.domain)
        if row is None:
            raise HTTPException(404, "领域总结不存在")
        if row["version"] != request.version:
            raise HTTPException(409, "总结已被修改，请刷新后核对并重试")
        if row["input_sha256"] != current.get(request.domain):
            raise HTTPException(409, "这个领域的画像有变动，请先重新生成总结")
        changed = (request.title, request.summary) != (row["title"], row["summary"])
        status = "approved" if request.approve else "pending" if changed else row["status"]
        if not changed and status == row["status"]:
            return public_row(row, False)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            updated = db.execute(
                "UPDATE domain_summaries SET title=?,summary=?,origin=?,status=?,at=?,version=version+1 "
                "WHERE level=? AND domain=? AND version=?",
                (
                    request.title,
                    request.summary,
                    "edited" if changed else row["origin"],
                    status,
                    now(),
                    request.level,
                    request.domain,
                    request.version,
                ),
            ).rowcount
            if updated != 1:
                raise HTTPException(409, "总结已被修改，请刷新后核对并重试")
            after = db.execute(
                "SELECT * FROM domain_summaries WHERE level=? AND domain=?", (request.level, request.domain)
            ).fetchone()
            return public_row(dict(after), False)

    def review(self, request: SummaryReview) -> dict:
        if len({item.domain for item in request.items}) != len(request.items):
            raise HTTPException(422, "批量操作不能包含重复领域")
        inputs, _ = self.state(request.level)
        current = {item["domain"]: item["sha256"] for item in inputs}
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = []
            for item in request.items:
                row = db.execute(
                    "SELECT * FROM domain_summaries WHERE level=? AND domain=?", (request.level, item.domain)
                ).fetchone()
                if row is None:
                    raise HTTPException(404, "领域总结不存在")
                if row["version"] != item.version:
                    raise HTTPException(409, "总结已被修改，整批操作未保存，请刷新后重试")
                if request.status == "approved" and row["input_sha256"] != current.get(item.domain):
                    raise HTTPException(409, "有领域的画像已变动，请先重新生成总结再确认")
                rows.append(row)
            for row in rows:
                if row["status"] != request.status:
                    db.execute(
                        "UPDATE domain_summaries SET status=?,at=?,version=version+1 WHERE level=? AND domain=?",
                        (request.status, now(), request.level, row["domain"]),
                    )
            return {"updated": len(rows)}


def build_summaries(directory: Path) -> str:
    """build-profile's last stage: B's full summary of each domain of the profile in `directory`, stored in the
    review database next to it. Returns why some domains have none, empty when all do."""
    from weclone.server.profile_review import ReviewStore

    try:
        store = SummaryStore(
            ReviewStore(directory / "profile_review.sqlite3", directory / "profile_hierarchy.json")
        )
        store.generate("full", wait=True)
    except Exception as exc:
        # The profile is already written, so any failure here only leaves the summaries for a later run.
        logger.warning(f"领域总结生成失败：{type(exc).__name__}: {exc}")
        return f"没有生成成功。原因：{str(exc) or type(exc).__name__}"
    return store.failures.get("full", "")
