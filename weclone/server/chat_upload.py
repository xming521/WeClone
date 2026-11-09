"""Chats uploaded from the web, as files of folders or as zips, kept in their own folder apart from csv_folder.

An upload collects files in its own staging directory, written as the storage mode writes everything, and replaces
the uploaded chats with the chats it finds when it is done. Staging belongs to the server process and is cleared when
it starts.
"""

import secrets
import shutil
import stat
import zipfile
from pathlib import Path

from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool

from weclone.data import chat_csv
from weclone.data.agent import profile_pipeline as pipeline
from weclone.server.errors import ApiError
from weclone.utils import secure_storage as secure

STAGING_DIR = secure.ROOT / "dataset" / ".uploads"


def upload_parts(path: str) -> tuple[str, ...] | None:
    """The segments of an uploaded file's relative path, or None for a file the upload leaves out: one hidden by the
    system or an archiver, or not read from a chat folder."""
    path = path.replace("\\", "/")
    parts = tuple(part for part in path.split("/") if part)
    # A colon in any segment: on Windows a drive there starts the path again from that drive.
    if not parts or path.startswith("/") or any(p in (".", "..") or "\0" in p or ":" in p for p in parts):
        raise ApiError(422, "upload_path_invalid", "文件路径必须是相对路径")
    if any(part.startswith(".") or part == "__MACOSX" for part in parts):
        return None
    name = parts[-1]
    return parts if name.lower().endswith(".zip") or chat_csv.is_chat_file(name) else None


def extract(archive: Path, staging: Path, parts: tuple[str, ...]) -> int:
    """Writes the chat files of a zip under its own name, as a folder of that name would be uploaded."""
    base = staging.joinpath(*parts[:-1], Path(parts[-1]).stem)
    try:
        with zipfile.ZipFile(archive) as zip_file:
            members = []
            for info in zip_file.infolist():
                if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                    continue
                member = upload_parts(info.filename)
                if member is not None and not member[-1].lower().endswith(".zip"):
                    members.append((info, member))
            # The sizes a zip lists bound what reading it yields.
            if sum(info.file_size for info, _ in members) > shutil.disk_usage(STAGING_DIR).free:
                raise ApiError(422, "upload_too_large", "解压后的文件超过磁盘剩余空间")
            for info, member in members:
                with zip_file.open(info) as source, secure.open_write(base.joinpath(*member)) as output:
                    while chunk := source.read(secure.CHUNK_SIZE):
                        output.write(chunk)
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        # RuntimeError: a member needs a password. NotImplementedError: a compression zipfile cannot read.
        raise ApiError(422, "upload_zip_invalid", f"无法读取 zip：{exc}") from exc
    return len(members)


def find_chats(staging: Path) -> dict[str, Path]:
    """Every folder holding CSV chunks is a chat, named by the folder, whose name is the contact."""
    chats: dict[str, Path] = {}
    for folder in sorted(path for path in (staging, *staging.rglob("*")) if path.is_dir()):
        if next(secure.iter_files(folder, "*.csv"), None) is None:
            continue
        if folder == staging:
            raise ApiError(422, "upload_no_chats", "CSV 要放在以聊天对象命名的文件夹里")
        if folder.name in chats:
            raise ApiError(422, "upload_duplicate_chat", f"有两个聊天都叫 {folder.name}", name=folder.name)
        chats[folder.name] = folder
    if not chats:
        raise ApiError(422, "upload_no_chats", "没有找到聊天记录")
    return chats


class Uploads:
    def __init__(self, builds, *, demo: bool):
        self.builds = builds
        self.demo = demo
        self.active: set[str] = set()
        shutil.rmtree(STAGING_DIR, ignore_errors=True)

    def start(self) -> str:
        if self.demo:
            raise ApiError(409, "demo_data_fixed", "演示模式不能上传聊天记录")
        if self.builds.running():
            raise ApiError(409, "build_running", "已经有一次构建在运行")
        upload_id = secrets.token_hex(8)
        (STAGING_DIR / upload_id).mkdir(parents=True)
        self.active.add(upload_id)
        return upload_id

    def staging(self, upload_id: str) -> Path:
        if upload_id not in self.active:
            raise ApiError(404, "upload_not_found", "这次上传已经结束")
        return STAGING_DIR / upload_id

    def finish(self, upload_id: str) -> dict:
        staging = self.staging(upload_id)
        chats = find_chats(staging)
        # Hidden names, which no uploaded path has.
        placed, replaced = staging / ".placed", staging / ".replaced"
        # A build starting now would read a folder half replaced.
        with self.builds.lock:
            # Before anything moves, so a refused upload is finished as it was once the build ends.
            if self.builds.running():
                raise ApiError(409, "build_running", "已经有一次构建在运行")
            for name, source in chats.items():
                # Only the chat's own files: a chat folder can hold another one.
                chat = placed / name
                chat.mkdir(parents=True)
                for path in source.iterdir():
                    if path.is_file():
                        path.rename(chat / path.name)
            if pipeline.UPLOAD_DIR.exists():
                pipeline.UPLOAD_DIR.rename(replaced)
            pipeline.UPLOAD_DIR.parent.mkdir(parents=True, exist_ok=True)
            placed.rename(pipeline.UPLOAD_DIR)
        self.discard(upload_id)
        return {"chats": len(chats)}

    def clear(self):
        """Removes the uploaded chats, as a new build starts from its own upload."""
        if self.demo:
            raise ApiError(409, "demo_data_fixed", "演示模式不能上传聊天记录")
        with self.builds.lock:
            if self.builds.running():
                raise ApiError(409, "build_running", "已经有一次构建在运行")
            shutil.rmtree(pipeline.UPLOAD_DIR, ignore_errors=True)

    def discard(self, upload_id: str):
        self.active.discard(upload_id)
        shutil.rmtree(STAGING_DIR / upload_id, ignore_errors=True)


def install_upload(app: FastAPI, builds, *, demo: bool) -> Uploads:
    uploads = Uploads(builds, demo=demo)

    @app.post("/api/data/uploads", status_code=201)
    def upload_start():
        return {"id": uploads.start()}

    @app.put("/api/data/uploads/{upload_id}/files")
    async def upload_file(upload_id: str, path: str, request: Request):
        staging = uploads.staging(upload_id)
        parts = upload_parts(path)
        if parts is None:
            return {"kept": 0}
        if not parts[-1].lower().endswith(".zip"):
            with secure.open_write(staging.joinpath(*parts)) as output:
                async for chunk in request.stream():
                    output.write(chunk)
            return {"kept": 1}
        # The zip itself is plaintext, so it waits where the storage mode keeps plaintext for a moment.
        with secure.runtime_directory("weclone-upload-") as directory:
            archive = directory / "upload.zip"
            with archive.open("wb") as output:
                async for chunk in request.stream():
                    output.write(chunk)
            kept = await run_in_threadpool(extract, archive, staging, parts)
        return {"kept": kept}

    @app.post("/api/data/uploads/{upload_id}/done")
    def upload_done(upload_id: str):
        return uploads.finish(upload_id)

    @app.delete("/api/data/uploads/{upload_id}")
    def upload_discard(upload_id: str):
        uploads.staging(upload_id)
        uploads.discard(upload_id)
        return {"ok": True}

    @app.delete("/api/data/uploaded")
    def uploaded_clear():
        uploads.clear()
        return {"ok": True}

    return uploads
