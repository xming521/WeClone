"""Single-owner authentication for the profile web UI."""

import hashlib
import hmac
import os
import secrets
import sqlite3
import string
import time
from contextlib import contextmanager
from pathlib import Path

import click
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

SESSION_SECONDS = 30 * 24 * 60 * 60
COOKIE = "weclone_session"
DEFAULT_DATABASE = Path(__file__).resolve().parents[2] / "dataset/res_csv/agent/memory_organization/profile_review.sqlite3"


def password_hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 600_000).hex()


class AuthStore:
    def __init__(self, database: Path, *, reset: bool = False):
        self.path = database.with_suffix(".auth.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        self.path.chmod(0o600)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS credentials (
                    id INTEGER PRIMARY KEY CHECK(id=1), salt TEXT NOT NULL, hash TEXT NOT NULL,
                    failures INTEGER NOT NULL DEFAULT 0, blocked_until REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS sessions (hash TEXT PRIMARY KEY, expires REAL NOT NULL);
            """)
            db.execute("BEGIN IMMEDIATE")
            if reset or not db.execute("SELECT 1 FROM credentials").fetchone():
                self._replace_password(db)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _replace_password(self, db):
        password = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))
        salt = secrets.token_hex(16)
        db.execute("INSERT OR REPLACE INTO credentials VALUES (1,?,?,0,0)", (salt, password_hash(password, salt)))
        db.execute("DELETE FROM sessions")
        click.echo(f"WeClone 网页访问密码（仅显示一次，请保存）：{password}")

    def login(self, password: str):
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            salt, expected, failures, blocked_until = db.execute(
                "SELECT salt, hash, failures, blocked_until FROM credentials WHERE id=1"
            ).fetchone()
            if blocked_until > now:
                raise HTTPException(429, "尝试次数过多，请一分钟后再试")
            if not hmac.compare_digest(password_hash(password, salt), expected):
                failures = (0 if blocked_until else failures) + 1
                db.execute("UPDATE credentials SET failures=?, blocked_until=? WHERE id=1",
                           (failures, now + 60 if failures >= 5 else 0))
                db.commit()
                raise HTTPException(401, "密码错误")
            db.execute("UPDATE credentials SET failures=0, blocked_until=0 WHERE id=1")
            token = secrets.token_urlsafe(32)
            expires = now + SESSION_SECONDS
            db.execute("DELETE FROM sessions WHERE expires<=?", (now,))
            db.execute("INSERT INTO sessions VALUES (?,?)", (self.digest(token), expires))
            return token, expires

    @staticmethod
    def digest(token: str):
        return hashlib.sha256(token.encode()).hexdigest()

    def expiry(self, token: str):
        with self.connect() as db:
            row = db.execute("SELECT expires FROM sessions WHERE hash=? AND expires>?",
                             (self.digest(token), time.time())).fetchone()
            return row[0] if row else None

    def logout(self, token: str):
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE hash=?", (self.digest(token),))


class Login(BaseModel):
    password: str = Field(min_length=1, max_length=256)


def install_auth(app, database: Path):
    store = AuthStore(database)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        path = request.url.path
        if path == "/api" or path.startswith("/api/"):
            if request.method not in {"GET", "HEAD", "OPTIONS"} and request.headers.get("X-WeClone-Request") != "1":
                response = JSONResponse({"detail": "请求验证失败"}, status_code=403)
            elif path != "/api/auth/login" and not store.expiry(request.cookies.get(COOKIE, "")):
                response = JSONResponse({"detail": "请先输入访问密码"}, status_code=401)
            else:
                response = await call_next(request)
            response.headers["Cache-Control"] = "no-store"
            return response
        return await call_next(request)

    @app.post("/api/auth/login")
    def login(body: Login, request: Request):
        token, expires = store.login(body.password)
        response = JSONResponse({"expires_at": expires})
        response.set_cookie(COOKIE, token, max_age=SESSION_SECONDS, httponly=True,
                            secure=request.url.scheme == "https", samesite="strict", path="/")
        return response

    @app.get("/api/auth/session")
    def session(request: Request):
        return {"expires_at": store.expiry(request.cookies.get(COOKIE, ""))}

    @app.post("/api/auth/logout")
    def logout(request: Request):
        store.logout(request.cookies.get(COOKIE, ""))
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE, path="/", httponly=True,
                               secure=request.url.scheme == "https", samesite="strict")
        return response
