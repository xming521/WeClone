"""How programs other than the owner's browser connect: one bearer token per kind of client, stored only as a hash in
the state directory so it outlives profile rebuilds.
"""

import hashlib
import os
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from fastapi import Request

from weclone.server.profile_summary import now


def bearer(request: Request) -> str:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def local_url(request: Request, path: str) -> str:
    """`path` on this server as programs on this machine reach it, whatever address the browser used."""
    host, port = request.scope.get("server") or (request.url.hostname, request.url.port)
    if host in (None, "0.0.0.0", "::"):
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    address = host if port in (None, 80) else f"{host}:{port}"
    return f"http://{address}{request.scope.get('root_path', '').rstrip('/')}/{path}"


class TokenStore:
    """The one token a kind of client connects with, in the single row of the `access` table of a 0600 file.
    Subclasses add their own columns to that row by overriding `create`."""

    PREFIX = ""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.fchmod(fd, 0o600)
        os.close(fd)
        with self.connect() as db:
            self.create(db)

    def create(self, db):
        db.execute(
            "CREATE TABLE IF NOT EXISTS access (id INTEGER PRIMARY KEY CHECK(id=1), "
            "token_sha256 TEXT, created TEXT, last_used TEXT)"
        )
        db.execute("INSERT OR IGNORE INTO access(id) VALUES (1)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def digest(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def row(self) -> sqlite3.Row:
        with self.connect() as db:
            return db.execute("SELECT * FROM access").fetchone()

    @staticmethod
    def token_state(row) -> dict | None:
        """When the token was made and last used, None without a token. Never the token itself."""
        return {"created": row["created"], "last_used": row["last_used"]} if row["token_sha256"] else None

    def issue(self) -> dict:
        """A new token in place of the old one. The token itself is returned only here."""
        token = self.PREFIX + secrets.token_urlsafe(32)
        created = now()
        with self.connect() as db:
            db.execute(
                "UPDATE access SET token_sha256=?,created=?,last_used=NULL", (self.digest(token), created)
            )
        return {"token": token, "created": created}

    def withdraw(self):
        with self.connect() as db:
            db.execute("UPDATE access SET token_sha256=NULL,created=NULL,last_used=NULL")

    def match(self, token: str) -> sqlite3.Row | None:
        """The row when `token` is the current one, None for anything else."""
        if not token.startswith(self.PREFIX):
            return None
        with self.connect() as db:
            return db.execute("SELECT * FROM access WHERE token_sha256=?", (self.digest(token),)).fetchone()

    def used(self):
        with self.connect() as db:
            db.execute("UPDATE access SET last_used=?", (now(),))
