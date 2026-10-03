"""Password protected storage shared by chat processing, training and the web UI.

Logical filenames stay stable; encrypted files have an authenticated binary header.
Passwords and unwrapped keys never go into configuration, argv, or environment.
"""

from __future__ import annotations

import base64
import contextlib
import contextvars
import copy
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import click
import pyjson5
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
MAGIC = b"WCENC\x01\r\n"
ENCRYPTED_SUFFIX = ".enc"
CHUNK_SIZE = 1024 * 1024
HEADER_SIZE = len(MAGIC) + 16 + 16
_settings: dict[str, Any] | None = None
_settings_stamp: tuple[str, int] | None = None
_run_key: tuple[str, str, bytes] | None = None
_request_key: contextvars.ContextVar[tuple[str, str, bytes] | None] = contextvars.ContextVar(
    "weclone_storage_key", default=None
)
_local_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


class SecurityError(RuntimeError):
    """A storage policy, integrity, or authentication check failed."""


class InvalidPassword(SecurityError):
    pass


class StorageLocked(SecurityError):
    pass


class UnlockedKey(bytes):
    """Key material with the immutable workspace identity it authenticated."""

    def __new__(cls, value: bytes, state_dir: str, generation: str):
        key = super().__new__(cls, value)
        key.state_dir = state_dir
        key.generation = generation
        return key


def configure(config_path: str | Path | None = None) -> dict[str, Any]:
    global _settings, _settings_stamp
    if config_path is not None:
        os.environ["WECLONE_CONFIG_PATH"] = str(config_path)
    path = Path(os.environ.get("WECLONE_CONFIG_PATH", ROOT / "settings.jsonc")).resolve()
    stamp = (str(path), path.stat().st_mtime_ns if path.exists() else -1)
    if _settings_stamp != stamp:
        try:
            document = pyjson5.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError) as exc:
            raise SecurityError("无法读取安全配置，请检查配置文件格式。") from exc
        args = document.get("security_args", {})
        if not isinstance(args, dict):
            raise SecurityError("security_args 必须是配置对象")
        if set(args) - {"storage_mode", "state_dir", "runtime_dir"}:
            raise SecurityError("security_args 仅支持 storage_mode/state_dir/runtime_dir，密码不能写入配置。")
        mode = args.get("storage_mode", "plaintext")
        if mode not in {"encrypted", "plaintext"}:
            raise SecurityError("storage_mode 只能是 encrypted 或 plaintext")
        state_dir = Path(args.get("state_dir", ".weclone"))
        if not state_dir.is_absolute():
            state_dir = ROOT / state_dir
        runtime_dir = Path(args.get("runtime_dir") or tempfile.gettempdir())
        _settings = {
            "storage_mode": mode,
            "state_dir": state_dir.resolve(),
            "runtime_dir": runtime_dir.resolve(),
            "explicit": "security_args" in document,
        }
        _settings_stamp = stamp
    assert _settings is not None
    return _settings


def _manifest_path() -> Path:
    return configure()["state_dir"] / "manifest.json"


def _manifest(*, required: bool = True, check_mode: bool = True) -> dict[str, Any] | None:
    path = _manifest_path()
    if not path.exists():
        if required:
            raise StorageLocked("尚未初始化。请先运行 weclone-cli security-init 设置密码。")
        return None
    try:
        value = json.loads(path.read_bytes())
        if value["version"] != 1 or value["storage_mode"] not in {"encrypted", "plaintext"}:
            raise ValueError
        if len(bytes.fromhex(value["generation"])) != 16:
            raise ValueError
    except (KeyError, ValueError, TypeError) as exc:
        raise SecurityError("安全状态文件无效，禁止继续读写") from exc
    if check_mode and value["storage_mode"] != configure()["storage_mode"]:
        raise SecurityError("配置模式与初始化模式不一致。切换模式必须重新初始化并重新处理全部聊天数据。")
    return value


def is_encrypted_mode() -> bool:
    _manifest(required=False)
    return configure()["storage_mode"] == "encrypted"


def is_initialized() -> bool:
    return _manifest(required=False) is not None


def get_generation() -> str:
    value = _manifest()
    assert value is not None
    return value["generation"]


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise SecurityError("安全存储目录不能是符号链接")
    path.chmod(0o700)


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    _private_dir(path.parent)
    if path.is_symlink():
        raise SecurityError("安全锁文件不能是符号链接")
    with _locks_guard:
        local_lock = _local_locks.setdefault(str(path.resolve()), threading.RLock())
    with local_lock, FileLock(str(path), mode=0o600):
        yield


@contextlib.contextmanager
def _atomic_writer(path: Path, *, exclusive: bool = False, guarded: bool = False) -> Iterator[BinaryIO]:
    _private_dir(path.parent)
    if path.is_symlink():
        raise SecurityError("安全输出文件不能是符号链接")
    fd, temporary = tempfile.mkstemp(prefix=".wc-write-", dir=path.parent)
    generation = get_generation() if guarded else None
    try:
        with os.fdopen(fd, "wb") as output:
            if hasattr(os, "fchmod"):
                os.fchmod(output.fileno(), 0o600)
            yield output
            output.flush()
            os.fsync(output.fileno())
        guard = (
            _file_lock(configure()["state_dir"] / "manifest.lock") if guarded else contextlib.nullcontext()
        )
        with guard:
            if guarded and generation != get_generation():
                raise SecurityError("密码重置或模式更换发生在写入期间，已取消旧代次提交。")
            if exclusive:
                # link is atomic and fails if another writer created the target.
                os.link(temporary, path)
            else:
                os.replace(temporary, path)
        if os.name != "nt":
            parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _password_key(password: str, salt: bytes) -> bytes:
    return Scrypt(salt=salt, length=32, n=2**17, r=8, p=1).derive(password.encode("utf-8"))


def _new_manifest(password: str) -> tuple[dict[str, Any], bytes]:
    if not password:
        raise SecurityError("密码不能为空。")
    generation = secrets.token_hex(16)
    salt, nonce, key = secrets.token_bytes(16), secrets.token_bytes(12), AESGCM.generate_key(bit_length=256)
    mode = configure()["storage_mode"]
    wrapped = AESGCM(_password_key(password, salt)).encrypt(
        nonce, key, f"weclone:{generation}:{mode}".encode()
    )

    def encode(value: bytes) -> str:
        return base64.b64encode(value).decode("ascii")

    return {
        "version": 1,
        "storage_mode": mode,
        "generation": generation,
        "auth_revision": secrets.token_hex(16),
        "kdf": "scrypt-n131072-r8-p1",
        "salt": encode(salt),
        "nonce": encode(nonce),
        "wrapped_key": encode(wrapped),
    }, key


def _save_manifest(value: dict[str, Any]) -> None:
    with _atomic_writer(_manifest_path()) as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2).encode())


def ensure_initialized(password: str | None = None) -> None:
    if password is None and is_initialized():
        return
    if password is None:
        password = click.prompt("设置网页与数据读写共用密码", hide_input=True, confirmation_prompt=True)
    with _file_lock(configure()["state_dir"] / "manifest.lock"):
        if _manifest(required=False) is not None:
            raise SecurityError("已经设置密码，不能再次初始化；改密或重置请使用对应命令。")
        value, key = _new_manifest(password)
        _save_manifest(value)
        _set_run_key(key)


def _unwrap(password: str) -> UnlockedKey:
    state_dir = str(configure()["state_dir"])
    value = _manifest()
    assert value is not None
    try:
        if value["kdf"] != "scrypt-n131072-r8-p1":
            raise SecurityError("不支持的密码派生格式")

        def decode(field: str) -> bytes:
            return base64.b64decode(value[field], validate=True)

        key = AESGCM(_password_key(password, decode("salt"))).decrypt(
            decode("nonce"),
            decode("wrapped_key"),
            f"weclone:{value['generation']}:{value['storage_mode']}".encode(),
        )
        return UnlockedKey(key, state_dir, value["generation"])
    except InvalidTag as exc:
        raise InvalidPassword("密码错误") from exc
    except (KeyError, ValueError, TypeError) as exc:
        raise SecurityError("安全状态文件无效") from exc


def check_password(password: str) -> bool:
    try:
        _unwrap(password)
        return True
    except InvalidPassword:
        return False


def unlock_key(password: str) -> bytes:
    """Authenticate without changing a process-wide or request-local context."""
    return _unwrap(password)


def get_auth_generation() -> str:
    value = _manifest()
    assert value is not None
    return value.get("auth_revision", value["generation"])


def _set_run_key(key: bytes) -> None:
    global _run_key
    state_dir, generation = str(configure()["state_dir"]), get_generation()
    if isinstance(key, UnlockedKey):
        if (key.state_dir, key.generation) != (state_dir, generation):
            raise StorageLocked("解锁期间数据代次已更换，请重新输入密码。")
    else:
        key = UnlockedKey(key, state_dir, generation)
    _run_key = (state_dir, generation, key)


def unlock(password: str) -> None:
    _set_run_key(_unwrap(password))


def lock() -> None:
    global _run_key
    _run_key = None


def get_unlocked_key() -> bytes:
    identity = (str(configure()["state_dir"]), get_generation())
    current = _request_key.get() or _run_key
    if current is None or current[:2] != identity:
        raise StorageLocked("数据尚未解锁，请输入同一访问密码。")
    return current[2]


@contextlib.contextmanager
def use_key(key: bytes) -> Iterator[None]:
    identity = (str(configure()["state_dir"]), get_generation())
    if not isinstance(key, UnlockedKey) or (key.state_dir, key.generation) != identity:
        raise StorageLocked("会话密钥所属数据代次已失效，请重新登录。")
    token = _request_key.set((*identity, key))
    try:
        yield
    finally:
        _request_key.reset(token)


@contextlib.contextmanager
def storage_context(session_key: bytes | None = None) -> Iterator[None]:
    if session_key is None:
        yield
    else:
        with use_key(session_key):
            yield


def ensure_unlocked() -> None:
    if not is_encrypted_mode():
        return
    ensure_initialized()
    try:
        get_unlocked_key()
    except StorageLocked:
        unlock(click.prompt("输入网页与数据读写共用密码", hide_input=True))


def change_password(old_password: str, new_password: str) -> None:
    with _file_lock(configure()["state_dir"] / "manifest.lock"):
        key = _unwrap(old_password)
        old = _manifest()
        assert old is not None
        updated, _ = _new_manifest(new_password)
        updated["generation"] = old["generation"]
        nonce = base64.b64decode(updated["nonce"])
        salt = base64.b64decode(updated["salt"])
        updated["wrapped_key"] = base64.b64encode(
            AESGCM(_password_key(new_password, salt)).encrypt(
                nonce, key, f"weclone:{old['generation']}:{old['storage_mode']}".encode()
            )
        ).decode()
        _save_manifest(updated)
        lock()


def reset_password(password: str) -> None:
    """Start a new generation; prior encrypted files are never converted or deleted."""
    with _file_lock(configure()["state_dir"] / "manifest.lock"):
        old = _manifest(required=False, check_mode=False)
        if old is not None:
            backup = configure()["state_dir"] / f"previous-{old['generation']}.json"
            if not backup.exists():
                with _atomic_writer(backup) as output:
                    output.write(json.dumps(old).encode())
        value, key = _new_manifest(password)
        _save_manifest(value)
        _set_run_key(key)


def logical_path(path: str | Path) -> Path:
    """Remove the storage suffix while keeping the underlying format suffix."""
    return Path(str(path).removesuffix(ENCRYPTED_SUFFIX))


def encrypted_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix == ENCRYPTED_SUFFIX else Path(str(path) + ENCRYPTED_SUFFIX)


def storage_path(path: str | Path) -> Path:
    """Return the filename used for new persisted data in the current mode."""
    return encrypted_path(path) if is_encrypted_mode() else Path(path)


def resolve_path(path: str | Path) -> Path:
    """Resolve suffixed ciphertext while allowing original plaintext inputs."""
    path = Path(path)
    candidate = encrypted_path(path)
    if candidate.is_file() and (is_encrypted_mode() or not path.exists()):
        return candidate
    if path.is_file() and path.suffix != ENCRYPTED_SUFFIX and is_encrypted_file(path):
        raise SecurityError("加密文件必须使用 .enc 后缀，请重新生成。")
    return path


def file_exists(path: str | Path) -> bool:
    return resolve_path(path).is_file()


def iter_files(directory: str | Path, pattern: str = "*.json", *, recursive: bool = False) -> Iterator[Path]:
    """Scan plaintext inputs and .enc files, excluding unsuffixed ciphertext."""
    directory = Path(directory)
    scan = directory.rglob if recursive else directory.glob
    files: dict[Path, Path] = {}
    for path in (*scan(pattern), *scan(pattern + ENCRYPTED_SUFFIX)):
        if path.is_file():
            if path.suffix != ENCRYPTED_SUFFIX and is_encrypted_file(path):
                continue
            name = logical_path(path)
            if name not in files or path.suffix == ENCRYPTED_SUFFIX:
                files[name] = path
    yield from (files[name] for name in sorted(files))


def _check_output_mode(path: Path, encrypted: bool) -> None:
    for candidate in {path, logical_path(path), encrypted_path(path)}:
        if candidate.exists() and is_encrypted_file(candidate) != encrypted:
            raise SecurityError(
                f"输出文件已有不同存储模式：{candidate}。请使用新目录重新生成，不能原地转换。"
            )


def is_encrypted_file(path: str | Path) -> bool:
    with Path(path).open("rb") as source:
        return source.read(len(MAGIC)) == MAGIC


def _file_cipher(salt: bytes, generation: bytes) -> AESGCM:
    return AESGCM(
        HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=MAGIC + generation).derive(
            get_unlocked_key()
        )
    )


def _encrypt_stream(source: BinaryIO, output: BinaryIO) -> None:
    generation, salt = bytes.fromhex(get_generation()), secrets.token_bytes(16)
    header = MAGIC + generation + salt
    cipher = _file_cipher(salt, generation)
    output.write(header)
    index = 0
    while chunk := source.read(CHUNK_SIZE):
        counter = struct.pack(">Q", index)
        ciphertext = cipher.encrypt(b"\0" * 4 + counter, chunk, header + counter + b"D")
        output.write(struct.pack(">I", len(ciphertext)))
        output.write(ciphertext)
        index += 1
    counter = struct.pack(">Q", index)
    ciphertext = cipher.encrypt(b"\0" * 4 + counter, b"", header + counter + b"F")
    output.write(struct.pack(">I", len(ciphertext)))
    output.write(ciphertext)


def _decrypt_stream(source: BinaryIO, output: BinaryIO) -> None:
    header = source.read(HEADER_SIZE)
    if len(header) != HEADER_SIZE or not header.startswith(MAGIC):
        raise SecurityError("文件不是受支持的加密文件")
    generation = header[len(MAGIC) : len(MAGIC) + 16]
    if generation.hex() != get_generation():
        raise SecurityError("文件属于旧密码或旧数据代次，请从原始聊天重新处理。")
    cipher = _file_cipher(header[-16:], generation)
    index = 0
    try:
        while True:
            size_bytes = source.read(4)
            if len(size_bytes) != 4:
                raise SecurityError("加密文件被截断")
            size = struct.unpack(">I", size_bytes)[0]
            if size < 16 or size > CHUNK_SIZE + 16:
                raise SecurityError("加密文件帧长度无效")
            ciphertext = source.read(size)
            if len(ciphertext) != size:
                raise SecurityError("加密文件被截断")
            final = size == 16
            counter = struct.pack(">Q", index)
            chunk = cipher.decrypt(
                b"\0" * 4 + counter, ciphertext, header + counter + (b"F" if final else b"D")
            )
            if final:
                if source.read(1):
                    raise SecurityError("加密文件末尾存在未认证数据")
                return
            output.write(chunk)
            index += 1
    except InvalidTag as exc:
        raise SecurityError("加密文件认证失败，文件可能被修改或密钥错误") from exc


def import_file(path: str | Path) -> Path:
    source_path = resolve_path(path).resolve()
    if not is_encrypted_mode() or is_encrypted_file(source_path):
        return source_path
    ensure_unlocked()
    stat = source_path.stat()
    identity = f"{source_path}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    name = hmac.new(get_unlocked_key(), identity, hashlib.sha256).hexdigest()
    destination = encrypted_path(
        configure()["state_dir"] / "imports" / get_generation() / f"{name}{source_path.suffix}"
    )
    if not destination.exists():
        with source_path.open("rb") as source, _atomic_writer(destination, guarded=True) as output:
            _encrypt_stream(source, output)
    return destination


def read_bytes(path: str | Path, *, import_plaintext: bool = False) -> bytes:
    path = resolve_path(path)
    if is_encrypted_mode():
        ensure_unlocked()
        if not is_encrypted_file(path):
            if not import_plaintext:
                raise SecurityError("安全模式拒绝读取内部明文文件；请从原始聊天重新生成加密产物。")
            path = import_file(path)
    with path.open("rb") as source:
        prefix = source.read(len(MAGIC))
        source.seek(0)
        if prefix != MAGIC:
            if is_encrypted_mode():
                raise SecurityError("安全模式拒绝读取明文文件，禁止自动降级")
            return source.read()
        if not is_encrypted_mode():
            raise SecurityError("明文模式不能读取加密文件，禁止自动降级")
        output = io.BytesIO()
        _decrypt_stream(source, output)
        return output.getvalue()


def read_text(path: str | Path, *, encoding: str = "utf-8", import_plaintext: bool = False) -> str:
    return read_bytes(path, import_plaintext=import_plaintext).decode(encoding)


def read_json(path: str | Path, *, import_plaintext: bool = False) -> Any:
    return json.loads(read_bytes(path, import_plaintext=import_plaintext))


def write_bytes(path: str | Path, data: bytes) -> Path:
    path = Path(path)
    encrypted = is_encrypted_mode()
    if encrypted:
        ensure_unlocked()
    _check_output_mode(path, encrypted)
    path = encrypted_path(path) if encrypted else path
    with _atomic_writer(path, guarded=encrypted) as output:
        if encrypted:
            _encrypt_stream(io.BytesIO(data), output)
        else:
            output.write(data)
    return path


def write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    return write_bytes(path, text.encode(encoding))


def write_json(path: str | Path, payload: Any, *, indent: int | None = 2, default: Any = None) -> Path:
    return write_text(path, json.dumps(payload, ensure_ascii=False, indent=indent, default=default) + "\n")


def read_jsonl(path: str | Path) -> list[Any]:
    return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Any) -> Path:
    return write_text(path, "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows))


def decrypt_file(source: str | Path, destination: str | Path) -> None:
    ensure_unlocked()
    source, destination = resolve_path(source), Path(destination)
    if not is_encrypted_mode():
        raise SecurityError("解密导出只能用于已初始化的安全模式")
    if source.resolve() == destination.resolve() or destination.exists():
        raise SecurityError("解密必须输出到新的文件，禁止覆盖输入或已有文件")
    with source.open("rb") as input_file, _atomic_writer(destination, exclusive=True, guarded=True) as output:
        _decrypt_stream(input_file, output)


@contextlib.contextmanager
def encrypted_sqlite(path: str | Path) -> Iterator[sqlite3.Connection]:
    path = Path(path)
    encrypted = is_encrypted_mode()
    if not encrypted:
        _check_output_mode(path, False)
        _private_dir(path.parent)
        db = sqlite3.connect(path, timeout=10)
        path.chmod(0o600)
        try:
            with db:
                yield db
        finally:
            db.close()
        return
    ensure_unlocked()
    lock_path = logical_path(path)
    with _file_lock(lock_path.with_name(lock_path.name + ".lock")):
        db = sqlite3.connect(":memory:", timeout=10)
        try:
            if file_exists(path):
                db.deserialize(read_bytes(path))
            db.execute("PRAGMA temp_store=MEMORY")
            db.execute("PRAGMA journal_mode=MEMORY")
            before = db.serialize() if file_exists(path) else None
            with db:
                yield db
            after = db.serialize()
            if after != before:
                write_bytes(path, after)
        finally:
            db.close()


@contextlib.contextmanager
def runtime_directory(prefix: str = "weclone-") -> Iterator[Path]:
    encrypted = is_encrypted_mode()
    root = configure()["runtime_dir"] if encrypted else None
    if encrypted:
        ensure_unlocked()
        root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=prefix, dir=root) as directory:
        path = Path(directory)
        path.chmod(0o700)
        yield path


def _materialize_file(source: Path, destination: Path, *, allow_plaintext: bool = False) -> None:
    _private_dir(destination.parent)
    with source.open("rb") as input_file, _atomic_writer(destination) as output:
        if is_encrypted_file(source):
            if source.suffix != ENCRYPTED_SUFFIX:
                raise SecurityError("加密文件必须使用 .enc 后缀，请重新生成。")
            _decrypt_stream(input_file, output)
        elif allow_plaintext:
            shutil.copyfileobj(input_file, output, CHUNK_SIZE)
        else:
            raise SecurityError("第三方数据目录含旧明文产物，请重新生成。")


def materialize_tree(source: str | Path, destination: str | Path) -> None:
    source, destination = resolve_path(source), Path(destination)
    if source.is_file():
        _materialize_file(source, destination)
        return
    _private_dir(destination)
    if not source.exists():
        return
    for path in source.rglob("*"):
        if path.is_symlink():
            raise SecurityError("敏感数据目录不能包含符号链接")
        if path.is_file() and not path.name.endswith(".lock"):
            # dataset_info contains schema and paths, never chat content.
            _materialize_file(
                path,
                destination / logical_path(path.relative_to(source)),
                allow_plaintext=path.name == "dataset_info.json",
            )


def persist_tree(source: str | Path, destination: str | Path) -> None:
    source, destination = Path(source), Path(destination)
    checkpoint_roots: list[Path] = []
    if source.is_dir():
        candidates = [source, *source.rglob("checkpoint-*")]
        checkpoint_roots = [
            path for path in candidates if path.is_dir() and re.fullmatch(r"checkpoint-\d+", path.name)
        ]
        for checkpoint in checkpoint_roots:
            try:
                state = json.loads((checkpoint / "trainer_state.json").read_bytes())
            except (OSError, ValueError):
                continue
            if not isinstance(state, dict) or state.get("global_step") != int(
                checkpoint.name.removeprefix("checkpoint-")
            ):
                continue
            target = destination / checkpoint.relative_to(source)
            _private_dir(target.parent)
            if target.exists():
                # Published checkpoints are immutable; a later save must use a new step.
                if not file_exists(target / "trainer_state.json"):
                    raise SecurityError("输出目录存在不完整 checkpoint，请使用新输出目录。")
                read_bytes(target / "trainer_state.json")
                continue
            stage = Path(tempfile.mkdtemp(prefix=".wc-checkpoint-", dir=target.parent))
            generation = get_generation()
            try:
                for path in checkpoint.rglob("*"):
                    if path.is_symlink():
                        raise SecurityError("checkpoint 不能包含符号链接")
                    if path.is_file():
                        with (
                            path.open("rb") as input_file,
                            _atomic_writer(
                                encrypted_path(stage / path.relative_to(checkpoint)), guarded=True
                            ) as output,
                        ):
                            _encrypt_stream(input_file, output)
                with _file_lock(configure()["state_dir"] / "manifest.lock"):
                    if generation != get_generation():
                        raise SecurityError("checkpoint 保存期间密码已重置，旧代次提交已取消。")
                    os.rename(stage, target)
            finally:
                shutil.rmtree(stage, ignore_errors=True)
    if source.is_file():
        pairs = [(source, destination)]
    else:
        if not source.exists():
            return
        pairs = [
            (path, destination / path.relative_to(source)) for path in source.rglob("*") if path.is_file()
        ]
    for path, target in pairs:
        if any(checkpoint == path.parent or checkpoint in path.parents for checkpoint in checkpoint_roots):
            continue
        if path.is_symlink():
            raise SecurityError("运行输出不能包含符号链接")
        _check_output_mode(target, True)
        target = encrypted_path(target)
        with path.open("rb") as input_file, _atomic_writer(target, guarded=True) as output:
            _encrypt_stream(input_file, output)


def _has_encrypted_files(path: Path) -> bool:
    path = resolve_path(path)
    if path.is_file():
        return is_encrypted_file(path)
    return path.exists() and any(is_encrypted_file(item) for item in path.rglob("*") if item.is_file())


def _materialize_media(directory: Path, runtime: Path, media_root: Path | None = None) -> None:
    """Resolve encrypted attachments referenced by SFT JSON in the temporary copy only."""
    attachments: dict[str, str] = {}
    for path in directory.rglob("*.json"):
        if path.name == "dataset_info.json":
            continue
        try:
            data = json.loads(path.read_bytes())
        except (ValueError, UnicodeError):
            continue
        changed = False

        def replace_media(value: Any) -> None:
            nonlocal changed
            if isinstance(value, list):
                for entry in value:
                    replace_media(entry)
            elif isinstance(value, dict):
                for field in ("images", "videos", "audios"):
                    if not isinstance(value.get(field), list):
                        continue
                    for index, filename in enumerate(value[field]):
                        if not isinstance(filename, str):
                            continue
                        original = Path(filename)
                        if not original.is_absolute():
                            original = (media_root or ROOT) / original
                        if str(original) not in attachments:
                            imported = import_file(original)
                            target = (
                                runtime
                                / "attachments"
                                / (
                                    hashlib.sha256(str(original).encode()).hexdigest()
                                    + logical_path(original).suffix
                                )
                            )
                            _materialize_file(imported, target)
                            attachments[str(original)] = str(target)
                        value[field][index] = attachments[str(original)]
                        changed = True
                for child in value.values():
                    replace_media(child)

        replace_media(data)
        if changed:
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


@contextlib.contextmanager
def sensitive_runtime(
    config_dict: dict[str, Any],
    *,
    dataset_keys: tuple[str, ...] = ("dataset_dir",),
    model_keys: tuple[str, ...] = ("adapter_name_or_path",),
    output_keys: tuple[str, ...] = ("output_dir",),
) -> Iterator[dict[str, Any]]:
    if not is_encrypted_mode():
        yield dict(config_dict)
        return
    ensure_unlocked()
    with runtime_directory("weclone-runtime-") as runtime:
        config = dict(config_dict)
        outputs: list[tuple[Path, Path]] = []
        transformed: dict[str, Path] = {}
        cache = runtime / "cache"
        _private_dir(cache)
        variables = {
            "TMPDIR": str(cache),
            "TMP": str(cache),
            "TEMP": str(cache),
            "HF_DATASETS_CACHE": str(cache / "datasets"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torch"),
            "VLLM_CACHE_ROOT": str(cache / "vllm"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
        }
        # tempfile may have cached /tmp before environment variables were set.
        old_tempdir = tempfile.tempdir
        old_env = {key: os.environ.get(key) for key in variables}
        dataset_config = sys.modules.get("datasets.config")
        dataset_cache_attributes: dict[str, Any] = {}
        try:
            os.environ.update(variables)
            tempfile.tempdir = str(cache)
            if dataset_config is not None:
                for attribute in ("HF_DATASETS_CACHE", "DOWNLOADED_DATASETS_PATH", "EXTRACTED_DATASETS_PATH"):
                    if hasattr(dataset_config, attribute):
                        dataset_cache_attributes[attribute] = getattr(dataset_config, attribute)
                        location = cache / "datasets" / attribute.lower()
                        _private_dir(location)
                        setattr(dataset_config, attribute, location)
            if "cache_dir" in config:
                config["cache_dir"] = str(cache / "model")
            for field in dict.fromkeys((*dataset_keys, *model_keys, *output_keys)):
                value = config.get(field)
                if not isinstance(value, str) or not value:
                    continue
                if field == "media_dir":
                    # Copy only referenced attachments, not the user's original media tree.
                    attachments = runtime / "attachments"
                    _private_dir(attachments)
                    config[field] = str(attachments)
                    continue
                # Comma-separated adapters are independent model paths.
                values = value.split(",") if field == "adapter_name_or_path" else [value]
                resolved = []
                for item in values:
                    original = Path(item)
                    if (
                        field in model_keys
                        and field not in output_keys
                        and not _has_encrypted_files(original)
                    ):
                        if field != "model_name_or_path" and original.exists():
                            raise SecurityError(
                                "个人 adapter 或 checkpoint 必须是加密产物，请重新训练；仅公共基础模型允许明文。"
                            )
                        resolved.append(item)
                        continue
                    identity = str(original.resolve())
                    target = transformed.get(identity)
                    if target is None:
                        name = (
                            logical_path(original).name if resolve_path(original).is_file() else original.name
                        )
                        target = runtime / field / name
                        materialize_tree(original, target)
                        transformed[identity] = target
                        if field in dataset_keys and target.is_dir() and field != "media_dir":
                            _materialize_media(target, runtime, Path(config_dict.get("media_dir", ROOT)))
                    resolved.append(str(target))
                    if field in output_keys and (target, original) not in outputs:
                        outputs.append((target, original))
                config[field] = ",".join(resolved)
            yield config
        finally:
            try:
                for source, target in outputs:
                    persist_tree(source, target)
            finally:
                tempfile.tempdir = old_tempdir
                for attribute, value in dataset_cache_attributes.items():
                    setattr(dataset_config, attribute, value)
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


def run_python(script_path: str | Path, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    script_path = str(Path(script_path).resolve())
    if not is_encrypted_mode():
        return subprocess.run([sys.executable, script_path, *args], **kwargs)
    ensure_unlocked()
    read_fd, write_fd = os.pipe()
    payload = json.dumps(
        {
            "config": str(Path(os.environ.get("WECLONE_CONFIG_PATH", ROOT / "settings.jsonc")).resolve()),
            "generation": get_generation(),
            "key": base64.b64encode(get_unlocked_key()).decode(),
        }
    ).encode()
    restore = "_restore_from_handle" if os.name == "nt" else "_restore_from_fd"
    bootstrap = (
        f"import sys,runpy;from weclone.utils.secure_storage import {restore};"
        f"{restore}(int(sys.argv[1]));sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    try:
        os.write(write_fd, payload)
        os.close(write_fd)
        write_fd = -1
        extra_fds = tuple(kwargs.pop("pass_fds", ()))
        if os.name == "nt":
            import msvcrt

            handle = msvcrt.get_osfhandle(read_fd)
            startup = copy.copy(kwargs.pop("startupinfo", None)) or subprocess.STARTUPINFO()
            attributes = dict(startup.lpAttributeList or {})
            handles = list(attributes.get("handle_list", ()))
            handles.extend(msvcrt.get_osfhandle(fd) for fd in (*extra_fds, read_fd))
            attributes["handle_list"] = list(dict.fromkeys(handles))
            startup.lpAttributeList = attributes
            inherited = {item: os.get_handle_inheritable(item) for item in attributes["handle_list"]}
            try:
                for item in inherited:
                    os.set_handle_inheritable(item, True)
                kwargs["close_fds"] = True
                return subprocess.run(
                    [sys.executable, "-c", bootstrap, str(handle), script_path, *args],
                    startupinfo=startup,
                    **kwargs,
                )
            finally:
                for item, previous in inherited.items():
                    os.set_handle_inheritable(item, previous)
        return subprocess.run(
            [sys.executable, "-c", bootstrap, str(read_fd), script_path, *args],
            pass_fds=(*extra_fds, read_fd),
            **kwargs,
        )
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def _restore_from_fd(fd: int) -> None:
    with os.fdopen(fd, "rb") as stream:
        payload = json.loads(stream.read())
    configure(payload["config"])
    if payload["generation"] != get_generation():
        raise SecurityError("子进程的运行密钥已失效")
    _set_run_key(
        UnlockedKey(base64.b64decode(payload["key"]), str(configure()["state_dir"]), payload["generation"])
    )


def _restore_from_handle(handle: int) -> None:
    import msvcrt

    _restore_from_fd(msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY))
