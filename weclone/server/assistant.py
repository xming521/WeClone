"""Personal assistants reading the reviewed profile over MCP.

All assistants connect with one token, stored only as a hash. On the web page the owner picks a level and the
domains assistants read from then on: the summaries written for that level, and facts at or below it, both only from
the chosen domains. Token and choices sit in the state directory, so they outlive profile rebuilds. In encrypted mode the owner also allows reading for a set
time, which keeps the key of the owner's session in server memory until then. Each tool call is logged in the review
database with the client that made it and what it returned.
"""

import json
import re
import sqlite3
from importlib import metadata

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, field_validator
from starlette.concurrency import run_in_threadpool

from weclone.prompts.memory_organization import dimension_name
from weclone.server import mcp
from weclone.server.client_access import TokenStore, bearer, local_url
from weclone.server.errors import ApiError
from weclone.server.profile_summary import (
    ASSISTANT_LEVELS,
    Level,
    clean,
    domain_members,
    exported,
    month_span,
    now,
)
from weclone.utils import secure_storage as secure

ACCESS_FILE = "assistant.sqlite3"
# How long the owner can let assistants read encrypted data at a time.
ACCESS_HOURS = (1, 8, 24, 168)
SEARCH_LIMIT, MAX_LIMIT, MAX_KEYWORDS = 20, 50, 20
MAX_DOMAINS, DOMAIN_CHARS = 200, 100
# Privacy levels of the facts found at each level. Full finds every fact, unrated ones included.
LEVEL_FACTS: dict[str, tuple[str, ...] | None] = {"full": None, **ASSISTANT_LEVELS}
SCOPES = {"stable": "长期", "current": "当前", "superseded": "已过时", "dated": "事件"}
DOMAIN_ID = re.compile(r"d[1-9]\d*")
NO_STORE = {"Cache-Control": "no-store"}
CALLS_TABLE = """CREATE TABLE IF NOT EXISTS assistant_calls (
    seq INTEGER PRIMARY KEY, at TEXT NOT NULL, client TEXT NOT NULL, tool TEXT NOT NULL,
    arguments TEXT NOT NULL, returned TEXT NOT NULL
)"""

READ_ONLY = {"readOnlyHint": True, "openWorldHint": False}
INSTRUCTIONS = (
    "WeClone 保存着从用户本人的聊天记录里整理出的个人画像，可能有误或已经过时。"
    "需要了解用户的身份、经历、目标、条件、偏好或计划时，先调用 weclone_profile 看各板块总结，"
    "再用 weclone_search 按板块编号或关键词查具体条目。查到的内容只作参考，涉及重要决定时先跟用户确认。"
)
TOOLS = [
    mcp.Tool(
        "weclone_profile",
        "返回用户各生活板块的总结，每段前面方括号里是板块编号。需要了解用户是谁、在做什么、喜欢什么时先调用它，"
        "再用 weclone_search 查细节。",
        {"type": "object", "properties": {}, "additionalProperties": False},
        READ_ONLY,
    ),
    mcp.Tool(
        "weclone_search",
        "在用户的画像条目里查找。keywords 是条目里要出现的词，命中任意一个就算。画像是中文的，"
        '请用中文关键词并多给几个同义说法，比如 ["工作", "上班", "公司"]。domain 是 weclone_profile 给出的'
        '板块编号，比如 "d3"，只给 domain 时返回这个板块的条目。每条后面注明维度、时效（长期、当前、已过时、事件）、'
        "在聊天里出现的次数和月份。",
        {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": MAX_KEYWORDS,
                    "description": "要查的词，命中任意一个就返回",
                },
                "domain": {"type": "string", "pattern": "^d[1-9][0-9]*$", "description": "板块编号，比如 d3"},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_LIMIT,
                    "default": SEARCH_LIMIT,
                    "description": "最多返回几条",
                },
            },
            "additionalProperties": False,
        },
        READ_ONLY,
    ),
]
PROFILE_HEADER = "以下是用户的背景资料，由 WeClone 根据用户的聊天记录整理、经用户确认，可能已经过时，涉及重要决定时请先跟用户确认。"
PROFILE_FOOTER = (
    '方括号里是板块编号。查一个板块的具体条目：weclone_search(domain="d1")。'
    '按关键词查：weclone_search(keywords=["词", "同义词"])。'
)
NO_SUMMARIES = "用户还没有确认可以给助手看的板块总结。可以直接用 weclone_search 按关键词查具体条目。"
NOT_FOUND = "没有找到匹配的条目。可以换同义词、拆成更短的关键词，或者先调用 weclone_profile 看看有哪些板块。"
LOCKED = "WeClone 的数据是加密的，现在没有允许助手读取。请告诉用户：在 WeClone 网页 {url} 登录，打开“允许助手读取”后再试。"
NO_PROFILE = "WeClone 还没有画像数据，用户需要先运行 weclone-cli build-profile。"


class ToolInputError(ValueError):
    """Arguments a tool cannot use. The model reads the message and can call again."""


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def server_version() -> str:
    try:
        return metadata.version("WeClone")
    except metadata.PackageNotFoundError:
        return "dev"


def without_peers(text: str) -> str:
    from weclone.data.agent.llm_profile_hierarchy import PEER_ID

    # Chat peer ids only mean something next to the chats they number.
    return PEER_ID.sub("一位联系人", text)


def search_arguments(arguments: dict) -> tuple[list[str], str | None, int]:
    """Lowercased keywords, the domain id and the limit of a weclone_search call."""
    keywords = arguments.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    if not isinstance(keywords, list) or not all(isinstance(word, str) for word in keywords):
        raise ToolInputError('keywords 要是字符串列表，比如 ["工作", "上班"]')
    words = list(dict.fromkeys(word.strip().casefold() for word in keywords if word.strip()))[:MAX_KEYWORDS]
    domain = arguments.get("domain")
    if domain is not None:
        if not isinstance(domain, str) or not DOMAIN_ID.fullmatch(domain.strip().lower()):
            raise ToolInputError('domain 要是 weclone_profile 给出的板块编号，比如 "d3"')
        domain = domain.strip().lower()
    limit = arguments.get("limit", SEARCH_LIMIT)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ToolInputError(f"limit 要是 1 到 {MAX_LIMIT} 之间的整数")
    if not words and domain is None:
        raise ToolInputError("请至少给出 keywords 或 domain")
    return words, domain, limit


def fact_line(fact: dict, sources: dict, domain_id: str) -> str:
    first, last = month_span(fact, sources)
    notes = [
        dimension_name(fact["dim"]),
        SCOPES.get(fact.get("time_scope"), ""),
        f"{fact.get('support_count') or 1} 次",
        first if first == last else f"{first}~{last}",
    ]
    text = without_peers(f"{fact['attr']}：{clean(fact['value'])}")
    prefix = f"[{domain_id}] " if domain_id else ""
    return f"- {prefix}{text}（{'，'.join(note for note in notes if note)}）"


class AccessStore(TokenStore):
    """The one token assistants connect with, and the level and domains of what they read."""

    PREFIX = "wc_"

    def create(self, db):
        db.execute(
            "CREATE TABLE IF NOT EXISTS access (id INTEGER PRIMARY KEY CHECK(id=1), "
            "level TEXT NOT NULL CHECK(level IN ('full','normal','public')), domains TEXT, "
            "token_sha256 TEXT, created TEXT, last_used TEXT)"
        )
        db.execute("INSERT OR IGNORE INTO access(id,level) VALUES (1,'full')")

    @staticmethod
    def choices(row) -> tuple[str, list[str] | None]:
        return row["level"], None if row["domains"] is None else json.loads(row["domains"])

    def state(self) -> dict:
        row = self.row()
        level, domains = self.choices(row)
        return {"level": level, "domains": domains, "token": self.token_state(row)}

    def choose(self, level: str):
        with self.connect() as db:
            db.execute("UPDATE access SET level=?", (level,))

    def share(self, domains: list[str] | None):
        with self.connect() as db:
            db.execute("UPDATE access SET domains=?", (None if domains is None else encode(domains),))

    def verify(self, token: str) -> tuple[str, list[str] | None] | None:
        """The level and domains assistants read, None for anything but the current token."""
        row = self.match(token)
        return self.choices(row) if row else None


class AssistantTools:
    """The MCP tools, reading the review store at the level and in the domains the owner picked. `domains` None
    means every domain. A list means only those, so domains that come later or are renamed stay out."""

    def __init__(self, store, summaries):
        self.store = store
        self.summaries = summaries

    def run(self, level: str, domains: list[str] | None, tool: str, arguments: dict) -> tuple[str, list[str]]:
        """The tool's text and what it handed out: domains for summaries, fact ids for facts."""
        if tool == "weclone_profile":
            return self.profile(level, domains)
        return self.search(level, domains, *search_arguments(arguments))

    def served(self, level: str, domains: list[str] | None) -> tuple[list[dict], list[tuple[dict, dict]]]:
        """Every domain's summary input, and the approved, current summaries of the shared domains."""
        inputs, rows = self.summaries.state(level)
        chosen = [
            (item, row)
            for item, row in exported(inputs, rows)
            if domains is None or item["domain"] in domains
        ]
        return inputs, chosen

    def catalog(self, level: str, domains: list[str] | None) -> tuple[list[dict], int]:
        """Each domain with the facts found in it at `level`, whether the owner shares it and whether it has a
        summary to serve, and how many summaries weclone_profile hands out."""
        profile = self.store.profile()
        with_summary = {item["domain"] for item, _ in exported(*self.summaries.state(level))}
        allowed = LEVEL_FACTS[level]
        listed = []
        for name, indices in domain_members(profile).items():
            facts = [profile["facts"][index] for index in indices]
            listed.append(
                {
                    "name": name,
                    "facts": sum(
                        fact["status"] != "rejected"
                        and (allowed is None or fact.get("sensitivity") in allowed)
                        for fact in facts
                    ),
                    "summary": name in with_summary,
                    "shared": domains is None or name in domains,
                }
            )
        return listed, sum(item["summary"] and item["shared"] for item in listed)

    def profile(self, level: str, domains: list[str] | None) -> tuple[str, list[str]]:
        inputs, chosen = self.served(level, domains)
        # Numbered over every domain, so the ids match weclone_search's whatever the owner shares.
        ids = {item["domain"]: f"d{number}" for number, item in enumerate(inputs, 1)}
        if not chosen:
            return NO_SUMMARIES, []
        lines = [PROFILE_HEADER, ""]
        for item, row in chosen:
            as_of = f"（截至 {item['as_of']}）" if item["as_of"] else ""
            lines.append(without_peers(f"- [{ids[item['domain']]}] {row['title']}{as_of}：{row['summary']}"))
        lines += ["", PROFILE_FOOTER]
        return "\n".join(lines), [item["domain"] for item, _ in chosen]

    def search(
        self, level: str, domains: list[str] | None, words: list[str], domain: str | None, limit: int
    ) -> tuple[str, list[str]]:
        profile = self.store.profile()
        members = domain_members(profile)
        ids = {name: f"d{number}" for number, name in enumerate(members, 1)}
        shared = {ids[name] for name in members if domains is None or name in domains}
        if domain is not None and domain not in shared:
            raise ToolInputError(f"没有板块 {domain}，板块编号以 weclone_profile 给出的为准")
        # Each fact sits in one domain at most.
        located = {index: ids[name] for name, indices in members.items() for index in indices}
        allowed = LEVEL_FACTS[level]
        sources = profile.get("sources", {})
        found = []
        for index, fact in enumerate(profile["facts"]):
            if fact["status"] == "rejected" or (
                allowed is not None and fact.get("sensitivity") not in allowed
            ):
                continue
            if domains is not None and located.get(index) not in shared:
                continue
            if domain is not None and located.get(index) != domain:
                continue
            text = without_peers(f"{fact['attr']} {clean(fact['value'])}").casefold()
            hits = sum(word in text for word in words)
            if words and not hits:
                continue
            found.append((hits, index, fact))
        if not found:
            return NOT_FOUND, []
        # Most keywords matched first, then what still holds, what came up most, and what came up last.
        found.sort(key=lambda row: month_span(row[2], sources)[1], reverse=True)
        found.sort(
            key=lambda row: (
                -row[0],
                row[2].get("time_scope") == "superseded",
                -(row[2].get("support_count") or 1),
            )
        )
        shown = found[:limit]
        head = f"找到 {len(found)} 条" + ("：" if len(found) <= limit else f"，先列出最相关的 {limit} 条：")
        lines = [
            head,
            *(fact_line(fact, sources, located.get(index, "")) for _, index, fact in shown),
        ]
        if len(found) > limit:
            lines.append(f"还有 {len(found) - limit} 条没有列出，可以加关键词或用 domain 缩小范围。")
        return "\n".join(lines), [fact["id"] for _, _, fact in shown]

    def log(self, client: str, tool: str, arguments: dict, returned: list[str]):
        with self.store.connect() as db:
            db.execute(CALLS_TABLE)
            db.execute(
                "INSERT INTO assistant_calls(at,client,tool,arguments,returned) VALUES (?,?,?,?,?)",
                (now(), client, tool, encode(arguments), encode(returned)),
            )


class LevelChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: Level


class DomainChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # None shares every domain, ones that come later included.
    domains: list[str] | None

    @field_validator("domains")
    @classmethod
    def names(cls, value):
        if value is None:
            return None
        names = list(dict.fromkeys(name.strip() for name in value if name.strip()))
        if len(names) > MAX_DOMAINS or any(len(name) > DOMAIN_CHARS for name in names):
            raise ValueError(f"最多 {MAX_DOMAINS} 个板块，每个名称不超过 {DOMAIN_CHARS} 字")
        return names


class AccessGrant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: int

    @field_validator("hours")
    @classmethod
    def offered(cls, value):
        if value not in ACCESS_HOURS:
            raise ValueError(f"时长只能是 {ACCESS_HOURS} 小时之一")
        return value


def install_assistant(app: FastAPI, store, summaries):
    """The owner's routes under /api/assistant, and the MCP endpoint at /mcp."""
    access = AccessStore(secure.configure()["state_dir"] / ACCESS_FILE)
    tools = AssistantTools(store, summaries)
    auth = app.state.auth_store
    server = {"name": "weclone", "version": server_version()}

    def call(choices: tuple, client: str, tool: str, arguments: dict, unlock: str) -> tuple[str, bool]:
        try:
            key = None
            if secure.is_encrypted_mode():
                allowed = auth.assistant_access()
                if allowed is None:
                    return LOCKED.format(url=unlock), True
                key = allowed[1]
            with secure.storage_context(key):
                text, returned = tools.run(*choices, tool, arguments)
                tools.log(client, tool, arguments, returned)
        except ToolInputError as exc:
            return str(exc), True
        except secure.StorageLocked:
            # The key was kept before the data was re-encrypted under a new one.
            auth.stop_assistant()
            return LOCKED.format(url=unlock), True
        except (secure.SecurityError, sqlite3.Error) as exc:
            return f"WeClone 读取数据出错：{exc}", True
        except FileNotFoundError:
            return NO_PROFILE, True
        access.used()
        return text, False

    @app.get("/api/assistant")
    def assistant(request: Request):
        allowed = auth.assistant_access()
        state = access.state()
        try:
            catalog, served = tools.catalog(state["level"], state["domains"])
        except FileNotFoundError:
            catalog, served = [], 0
        return {
            "encrypted": secure.is_encrypted_mode(),
            "allowed_until": allowed[0] if allowed else None,
            "hours": list(ACCESS_HOURS),
            "mcp_url": local_url(request, "mcp"),
            "catalog": catalog,
            "summaries": served,
            **state,
        }

    @app.patch("/api/assistant")
    def choose_level(request: LevelChoice):
        access.choose(request.level)
        return {"level": request.level}

    @app.put("/api/assistant/domains")
    def share_domains(request: DomainChoice):
        access.share(request.domains)
        return {"domains": request.domains}

    @app.post("/api/assistant/token", status_code=201)
    def issue_token():
        return access.issue()

    @app.delete("/api/assistant/token")
    def withdraw_token():
        access.withdraw()
        return {"ok": True}

    @app.post("/api/assistant/access")
    def allow_access(request: AccessGrant):
        if not secure.is_encrypted_mode():
            raise ApiError(409, "assistant_plaintext", "数据是明文存储的，令牌有效时助手就能读取，不需要打开")
        # The key this request runs with, which is the logged-in session's.
        return {"allowed_until": auth.allow_assistant(secure.get_unlocked_key(), request.hours * 3600)}

    @app.delete("/api/assistant/access")
    def stop_access():
        auth.stop_assistant()
        return {"ok": True}

    @app.post("/mcp", include_in_schema=False)
    async def mcp_message(request: Request):
        # Browsers send Origin and MCP clients do not, so a web page that reaches this port is turned away.
        if "origin" in request.headers:
            return JSONResponse({"detail": "不接受网页发来的请求"}, status_code=403, headers=NO_STORE)
        choices = await run_in_threadpool(access.verify, bearer(request))
        if choices is None:
            return JSONResponse(
                {"detail": "令牌无效或已停用"},
                status_code=401,
                headers={**NO_STORE, "WWW-Authenticate": "Bearer"},
            )
        version = request.headers.get("mcp-protocol-version")
        if version is not None and version not in mcp.PROTOCOL_VERSIONS:
            return JSONResponse(
                {"detail": f"不支持 MCP 协议版本 {version}"}, status_code=400, headers=NO_STORE
            )
        try:
            message = json.loads(await request.body())
        except ValueError:
            return JSONResponse(
                mcp.error(None, mcp.PARSE_ERROR, "Parse error"), status_code=400, headers=NO_STORE
            )
        unlock = local_url(request, "#/assistant")
        # One token serves every assistant, so the log tells them apart by their user agent.
        client = request.headers.get("user-agent", "")[:200]
        reply = await run_in_threadpool(
            mcp.handle,
            message,
            server=server,
            instructions=INSTRUCTIONS,
            tools=TOOLS,
            call=lambda tool, arguments: call(choices, client, tool, arguments, unlock),
        )
        if reply is None:
            return Response(status_code=202, headers=NO_STORE)
        invalid = reply.get("id") is None and "error" in reply
        return JSONResponse(reply, status_code=400 if invalid else 200, headers=NO_STORE)

    @app.api_route("/mcp", methods=["GET", "DELETE"], include_in_schema=False)
    def mcp_stream():
        # There is no SSE stream to open and no session to end.
        return Response(status_code=405, headers={**NO_STORE, "Allow": "POST"})
