"""Chat bot platforms (AstrBot, LangBot and others) calling the fine-tuned clone over the OpenAI-compatible /v1 API.

They send no session cookie, only the API key they were configured with, so all of them share one token made on the
web page and stored only as a hash in the state directory. In encrypted mode the data key, sealed under the token,
is stored with it: a bot's request opens it with the token it presents, so the model loads and replies are generated
with nobody logged in and across restarts, until the token is generated again or withdrawn. The owner's own tools,
such as test-model, keep calling /v1 with their logged-in session.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from weclone.server.client_access import TokenStore, bearer, local_url
from weclone.utils import secure_storage as secure

ACCESS_FILE = "bots.sqlite3"
STALE = "WeClone 的数据重新设置过密码，这个令牌已失效。请在 WeClone 网页 {url} 重新生成令牌。"


class BotAccess(TokenStore):
    """The one token chat bots call /v1 with, and in encrypted mode the data key sealed under it."""

    PREFIX = "wcbot_"

    def create(self, db):
        db.execute(
            "CREATE TABLE IF NOT EXISTS access (id INTEGER PRIMARY KEY CHECK(id=1), "
            "token_sha256 TEXT, created TEXT, last_used TEXT, sealed_key TEXT)"
        )
        db.execute("INSERT OR IGNORE INTO access(id) VALUES (1)")

    def state(self) -> dict:
        return {"token": self.token_state(self.row())}

    def issue(self, key: bytes | None) -> dict:
        """A new token, with `key` sealed under it in encrypted mode."""
        issued = super().issue()
        # Nobody has the new token before it is returned, so the key still sealed under the old one goes unused.
        sealed = None if key is None else secure.seal_key(key, issued["token"])
        with self.connect() as db:
            db.execute("UPDATE access SET sealed_key=?", (sealed,))
        return issued

    def withdraw(self):
        with self.connect() as db:
            db.execute("UPDATE access SET token_sha256=NULL,created=NULL,last_used=NULL,sealed_key=NULL")


def install_bots(app: FastAPI, *, inference: bool):
    """The owner's routes under /api/bots, and the gate the auth middleware sends cookie-less /v1 requests through."""
    access = BotAccess(secure.configure()["state_dir"] / ACCESS_FILE)

    def refuse(status: int, detail: str, headers: dict | None = None) -> JSONResponse:
        return JSONResponse({"detail": detail}, status_code=status, headers=headers)

    async def gate(request: Request, call_next):
        # Bot platforms call from their servers. Browsers send Origin, so a web page that reaches this port is
        # turned away even if it learned the token.
        if "origin" in request.headers:
            return refuse(403, "不接受网页发来的请求")
        token = bearer(request)
        row = await run_in_threadpool(access.match, token)
        if row is None:
            return refuse(
                401, "令牌无效或已停用，请在 WeClone 网页的机器人接入页生成", {"WWW-Authenticate": "Bearer"}
            )
        key = None
        if secure.is_encrypted_mode():
            try:
                key = secure.open_sealed_key(row["sealed_key"], token)
            except secure.StorageLocked:
                return refuse(403, STALE.format(url=local_url(request, "#/bots")))
        with secure.storage_context(key):
            response = await call_next(request)
        await run_in_threadpool(access.used)
        return response

    app.state.bot_gate = gate

    @app.get("/api/bots")
    def bots(request: Request):
        return {
            "encrypted": secure.is_encrypted_mode(),
            "inference": inference,
            "api_url": local_url(request, "v1"),
            **access.state(),
        }

    @app.post("/api/bots/token", status_code=201)
    def issue_token():
        # In encrypted mode, the key this request runs with, which is the logged-in session's.
        return access.issue(secure.get_unlocked_key() if secure.is_encrypted_mode() else None)

    @app.delete("/api/bots/token")
    def withdraw_token():
        access.withdraw()
        return {"ok": True}
