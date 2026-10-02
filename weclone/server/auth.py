"""Single-owner authentication and in-memory data keys for the web UI."""

import asyncio
import hashlib
import os
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from weclone.utils import secure_storage as secure

SESSION_SECONDS = 2 * 60 * 60
COOKIE = "weclone_session"
DEFAULT_DATABASE = (
    Path(__file__).resolve().parents[2] / "dataset/res_csv/agent/memory_organization/profile_review.sqlite3"
)


class AuthStore:
    def __init__(self, database: Path):
        self.path = database.with_suffix(".auth.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        self.path.chmod(0o600)
        self._keys: dict[str, tuple[float, bytes]] = {}
        self._key_lock = threading.RLock()
        self._generation = ""
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS auth_state (
                    id INTEGER PRIMARY KEY CHECK(id=1), generation TEXT NOT NULL,
                    failures INTEGER NOT NULL DEFAULT 0, blocked_until REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS sessions (hash TEXT PRIMARY KEY, expires REAL NOT NULL);
            """)
            db.execute("BEGIN IMMEDIATE")
            # Old generated passwords and 30-day cookies are no longer valid.
            if db.execute("PRAGMA user_version").fetchone()[0] < 2:
                db.execute("DELETE FROM sessions")
                db.execute("DROP TABLE IF EXISTS credentials")
                db.execute("PRAGMA user_version=2")
            self._sync_generation(db)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _sync_generation(self, db):
        generation = secure.get_auth_generation() if secure.is_initialized() else ""
        row = db.execute("SELECT generation FROM auth_state WHERE id=1").fetchone()
        if row is None or row[0] != generation:
            db.execute("DELETE FROM sessions")
            db.execute("INSERT OR REPLACE INTO auth_state VALUES (1,?,0,0)", (generation,))
            with self._key_lock:
                self._keys.clear()
        self._generation = generation

    def setup(self, password: str, confirmation: str):
        if password != confirmation:
            raise HTTPException(422, "两次输入的密码不一致")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if secure.is_initialized():
                raise HTTPException(409, "已设置密码，请使用现有密码登录")
            try:
                secure.ensure_initialized(password)
            except secure.SecurityError as exc:
                raise HTTPException(422, str(exc)) from None
            finally:
                secure.lock()
            self._sync_generation(db)
        return self.login(password)

    def login(self, password: str):
        if not secure.is_initialized():
            raise HTTPException(409, "请先设置密码")
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._sync_generation(db)
            failures, blocked_until = db.execute(
                "SELECT failures, blocked_until FROM auth_state WHERE id=1"
            ).fetchone()
            if blocked_until > now:
                raise HTTPException(429, "尝试次数过多，请一分钟后再试")
            try:
                verified_key = secure.unlock_key(password)
            except secure.InvalidPassword:
                failures = (0 if blocked_until else failures) + 1
                db.execute(
                    "UPDATE auth_state SET failures=?, blocked_until=? WHERE id=1",
                    (failures, now + 60 if failures >= 5 else 0),
                )
                db.commit()
                raise HTTPException(401, "密码错误")
            key = None
            if secure.is_encrypted_mode():
                key = verified_key
            db.execute("UPDATE auth_state SET failures=0, blocked_until=0 WHERE id=1")
            token = secrets.token_urlsafe(32)
            expires = now + SESSION_SECONDS
            self.prune_keys(now)
            db.execute("DELETE FROM sessions WHERE expires<=?", (now,))
            digest = self.digest(token)
            db.execute("INSERT INTO sessions VALUES (?,?)", (digest, expires))
            if key is not None:
                with self._key_lock:
                    self._keys[digest] = (expires, key)
            return token, expires

    @staticmethod
    def digest(token: str):
        return hashlib.sha256(token.encode()).hexdigest()

    def prune_keys(self, now=None):
        now = time.time() if now is None else now
        with self._key_lock:
            for digest, (expires, _) in list(self._keys.items()):
                if expires <= now:
                    del self._keys[digest]

    def expiry(self, token: str):
        self.prune_keys()
        if not token:
            return None
        digest = self.digest(token)
        with self.connect() as db:
            self._sync_generation(db)
            row = db.execute(
                "SELECT expires FROM sessions WHERE hash=? AND expires>?", (digest, time.time())
            ).fetchone()
        if row and secure.is_encrypted_mode():
            # A cookie cannot recover a key lost at service restart.
            with self._key_lock:
                if digest not in self._keys:
                    return None
        return row[0] if row else None

    def key(self, token: str):
        with self._key_lock:
            entry = self._keys.get(self.digest(token))
            return entry[1] if entry else None

    def logout(self, token: str):
        digest = self.digest(token)
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE hash=?", (digest,))
        with self._key_lock:
            self._keys.pop(digest, None)


class Login(BaseModel):
    password: str = Field(min_length=1)


class Setup(Login):
    confirmation: str = Field(min_length=1)


def install_auth(app, database: Path):
    store = AuthStore(database)
    app.state.auth_store = store
    public = {"/api/auth/status", "/api/auth/setup", "/api/auth/login", "/api/auth/session"}

    async def authorize(request: Request, call_next):
        path = request.url.path
        if (
            request.method not in {"GET", "HEAD", "OPTIONS"}
            and request.headers.get("X-WeClone-Request") != "1"
        ):
            response = JSONResponse({"detail": "请求验证失败"}, status_code=403)
        elif path in public:
            response = await call_next(request)
        else:
            token = request.cookies.get(COOKIE, "")
            if not store.expiry(token):
                response = JSONResponse({"detail": "请先输入访问密码"}, status_code=401)
            else:
                key = store.key(token)
                if secure.is_encrypted_mode() and key is None:
                    response = JSONResponse({"detail": "请重新输入密码解锁"}, status_code=401)
                elif key is not None:
                    with secure.use_key(key):
                        response = await call_next(request)
                else:
                    response = await call_next(request)
        return response

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        path = request.url.path
        private = path == "/api" or path.startswith(("/api/", "/v1/"))
        if not private:
            return await call_next(request)
        try:
            response = await authorize(request, call_next)
        except secure.SecurityError as exc:
            response = JSONResponse({"detail": str(exc)}, status_code=409)
        response.headers["Cache-Control"] = "no-store"
        return response

    def session_response(token, expires, request):
        response = JSONResponse({"expires_at": expires})
        response.set_cookie(
            COOKIE,
            token,
            max_age=SESSION_SECONDS,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="strict",
            path="/",
        )
        return response

    @app.get("/api/auth/status")
    def status():
        return {
            "initialized": secure.is_initialized(),
            "storage_mode": "encrypted" if secure.is_encrypted_mode() else "plaintext",
        }

    @app.post("/api/auth/setup")
    def setup(body: Setup, request: Request):
        return session_response(*store.setup(body.password, body.confirmation), request)

    @app.post("/api/auth/login")
    def login(body: Login, request: Request):
        return session_response(*store.login(body.password), request)

    @app.get("/api/auth/session")
    def session(request: Request):
        return {"expires_at": store.expiry(request.cookies.get(COOKIE, ""))}

    @app.post("/api/auth/logout")
    def logout(request: Request):
        store.logout(request.cookies.get(COOKIE, ""))
        response = JSONResponse({"ok": True})
        response.delete_cookie(
            COOKIE, path="/", httponly=True, secure=request.url.scheme == "https", samesite="strict"
        )
        return response

    async def clear_expired_keys():
        while True:
            await asyncio.sleep(1)
            store.prune_keys()
            try:
                generation = secure.get_auth_generation() if secure.is_initialized() else ""
                if generation != store._generation:
                    with store.connect() as db:
                        store._sync_generation(db)
            except secure.SecurityError:
                with store._key_lock:
                    store._keys.clear()

    async def start_key_cleanup():
        app.state.key_cleanup_task = asyncio.create_task(clear_expired_keys())

    async def stop_key_cleanup():
        task = getattr(app.state, "key_cleanup_task", None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        with store._key_lock:
            store._keys.clear()

    app.add_event_handler("startup", start_key_cleanup)
    app.add_event_handler("shutdown", stop_key_cleanup)
