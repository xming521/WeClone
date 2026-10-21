"""The profile facts for one reply: render the chat, route it to topics, fill the fact budget, let an LLM filter."""

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Literal

import pyjson5

from weclone.harness.profile.routing import JevRouter, LLMRouter, ask_json
from weclone.harness.profile.tree import ProfileTree, local_time
from weclone.prompts.harness import PROFILE_FILTER_SCHEMA, chat_state, filter_prompt
from weclone.utils import secure_storage
from weclone.utils.log import logger

MAX_MESSAGES = 20
MAX_MESSAGE_CHARS = 400
# A time line opens each stretch of chat; a pause this long starts a new one.
SESSION_GAP = timedelta(minutes=30)


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]  # user: the contact A; assistant: B, whose reply is being written
    content: str
    time: datetime | None = None


@dataclass(frozen=True)
class ProfileFact:
    index: int  # position in the profile's facts
    tier: Literal["must", "helpful"]
    attr: str
    value: str
    text: str  # "attr：value（months）", as the filter saw it


@dataclass
class ProfileContext:
    facts: list[ProfileFact] = field(
        default_factory=list
    )  # must first, then helpful, each in the filter's order
    topics: list[str] = field(default_factory=list)  # routed topic keys, ranked
    candidates: list[int] = field(default_factory=list)  # fact indices handed to the filter
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None  # routing or filtering failed; the reply then goes ahead without profile facts

    def text(self) -> str:
        return "\n".join(fact.text for fact in self.facts)


def render_state(messages: list[ChatMessage], now: datetime, contact: str | None = None) -> str:
    if not messages or messages[-1].role != "user":
        raise ValueError("最后一条消息必须是对方（user）发的")
    lines, last = [], None
    for message in messages[-MAX_MESSAGES:]:
        if message.time is not None:
            sent = local_time(message.time)
            if last is None or sent - last >= SESSION_GAP:
                lines.append(f"（{sent:%Y-%m-%d %H:%M}）")
            last = sent
        content = message.content
        if len(content) > MAX_MESSAGE_CHARS:
            content = content[:MAX_MESSAGE_CHARS] + "…"
        lines.append(f"{'A' if message.role == 'user' else 'B'}: {content}")
    return chat_state("\n".join(lines), f"{now:%Y-%m-%d %H:%M}", contact)


class LLMFilter:
    """Picks the candidates the reply needs, in two tiers: must and helpful."""

    def __init__(self, tree: ProfileTree, client):
        self.tree = tree
        self.client = client

    @staticmethod
    def validate(payload, ids: dict[str, int]) -> tuple[list[int], list[int]]:
        tiers, dropped = [], []
        for name in ("must", "helpful"):
            values = payload[name]
            if not isinstance(values, list):
                raise ValueError(f"{name} 必须是列表")
            dropped += [value for value in values if value not in ids]
            tiers.append([ids[value] for value in dict.fromkeys(values) if value in ids])
        must, helpful = tiers
        if dropped:
            if not must and not helpful:
                raise ValueError(f"must 和 helpful 里的编号都不在候选条目里 {dropped}")
            logger.warning(f"画像过滤返回了不在候选条目里的编号，已丢弃：{dropped}")
        return must, [k for k in helpful if k not in must]

    def pick(self, state: str, candidates: list[int], now: datetime) -> tuple[list[int], list[int]]:
        ids = {f"F{k}": k for k in candidates}
        prompt = filter_prompt(state, [f"F{k} {self.tree.fact_text(k, now)}" for k in candidates])
        return ask_json(
            self.client,
            prompt,
            PROFILE_FILTER_SCHEMA,
            lambda payload: self.validate(payload, ids),
            "harness_profile_filter",
        )


class ProfileContextSelector:
    def __init__(self, tree: ProfileTree, router, filter: LLMFilter, fact_budget: int = 30, client=None):
        self.tree = tree
        self.router = router
        self.filter = filter
        self.fact_budget = fact_budget
        self.client = client

    def select(
        self, messages: list[ChatMessage], now: datetime | None = None, contact: str | None = None
    ) -> ProfileContext:
        """`messages` ends with A's message that B is replying to; `contact` is B's name for A."""
        now = local_time(now) if now else datetime.now()
        state = render_state(messages, now, contact)
        context = ProfileContext()
        try:
            start = perf_counter()
            context.topics = self.router.route(state)
            context.timings_ms["route"] = (perf_counter() - start) * 1000
            context.candidates = self.tree.fill(context.topics, now, self.fact_budget)
            if not context.candidates:
                return context
            start = perf_counter()
            must, helpful = self.filter.pick(state, context.candidates, now)
            context.timings_ms["filter"] = (perf_counter() - start) * 1000
        except Exception as exc:
            context.error = f"{type(exc).__name__}: {exc}"
            logger.warning(f"画像上下文选择失败，本次回复不带画像：{context.error}")
            return context
        context.facts = [
            ProfileFact(
                k, tier, self.tree.facts[k]["attr"], self.tree.facts[k]["value"], self.tree.fact_text(k, now)
            )
            for tier, picked in (("must", must), ("helpful", helpful))
            for k in picked
        ]
        return context

    def close(self) -> None:
        if self.client is not None:
            self.client.close()


def build_profile_context_selector(
    config_path: str | Path | None = None, profile_path: str | Path | None = None
) -> ProfileContextSelector:
    """From settings.jsonc harness_args.profile_context; the LLM backend reuses codex_exec_args or make_dataset_args."""
    from weclone.core.inference.llm_client import build_llm_client
    from weclone.utils.config_models import CodexExecArgs, HarnessArgs

    secure_storage.configure(config_path)
    path = Path(config_path or os.environ.get("WECLONE_CONFIG_PATH", secure_storage.ROOT / "settings.jsonc"))
    document = pyjson5.loads(path.read_text(encoding="utf-8"))
    args = HarnessArgs.model_validate(document.get("harness_args") or {}).profile_context
    tree = ProfileTree.load(profile_path or args.profile_path)
    if args.llm_provider == "codex_exec":
        codex = CodexExecArgs.model_validate(document.get("codex_exec_args") or {})
        client = build_llm_client(
            "codex_exec",
            model=args.model or codex.model,
            effort=args.effort or codex.effort or "low",
            timeout=args.llm_timeout,
            command=codex.command or "codex",
            sandbox=codex.sandbox or "read-only",
            isolated_cwd=True,
            config_path=path,
        )
    else:
        client = build_llm_client("api", config_path=path, timeout=args.llm_timeout)
    if args.router == "jev":
        router = JevRouter(tree, args.jev_url, args.jev_api_key, args.jev_top_domains, args.jev_timeout)
    else:
        router = LLMRouter(tree, client, args.llm_route_max_topics)
    return ProfileContextSelector(tree, router, LLMFilter(tree, client), args.fact_budget, client)
