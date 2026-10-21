"""Routers: from the chat state to a ranked list of profile topic keys."""

import json
import urllib.request
from collections.abc import Callable
from typing import Any

from weclone.harness.profile.tree import ProfileTree
from weclone.prompts.harness import (
    PROFILE_DOMAIN_QUESTION,
    PROFILE_ROUTE_SCHEMA,
    PROFILE_TOPIC_QUESTION,
    route_prompt,
)
from weclone.utils.log import logger


def ask_json(client, prompt: str, schema: dict, validate: Callable[[Any], Any], stage: str) -> Any:
    """One LLM call with a JSON schema; a failed or invalid answer is retried once, with the problem appended."""
    from weclone.core.inference.llm_client import LLMRequest

    error = None
    for attempt in range(2):
        text = (
            prompt
            if error is None
            else f"{prompt}\n\n上一次输出未通过检查：{error}。请修正后重新输出完整结果。"
        )
        response = client.generate(
            # No max_tokens: with the default 1024 a reasoning model can spend them all before answering.
            LLMRequest.from_prompt(
                text,
                max_tokens=None,
                json_mode=True,
                json_schema=schema,
                metadata={"stage": stage, "attempt": attempt},
            )
        )
        if not response.ok:
            error = response.error or "empty response"
            continue
        try:
            return validate(response.parsed_json)
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
    raise RuntimeError(f"{stage} failed: {error}")


class JevRouter:
    """Two-stage choice on a Jev-compatible systemone endpoint: the domain, then topics in the top domains.

    A topic scores its domain's probability times its own probability within the domain.
    """

    name = "jev"

    def __init__(
        self,
        tree: ProfileTree,
        url: str,
        api_key: str | None = None,
        top_domains: int = 3,
        timeout: float = 30.0,
    ):
        self.tree = tree
        self.url = url
        self.api_key = api_key
        self.top_domains = top_domains
        self.timeout = timeout
        self.topic_criteria = {
            str(d): {topic.key.split("|")[1]: topic.text for topic in topics}
            for d in range(len(tree.domains))
            if (topics := tree.domain_topics(d))
        }
        self.domain_question = {
            "type": "choice",
            "instructions": PROFILE_DOMAIN_QUESTION,
            "criteria": {d: "{}：{}".format(*tree.domains[int(d)]) for d in self.topic_criteria},
        }

    def ask(self, state: str, questions: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps({"state": state, "questions": questions}, ensure_ascii=False).encode()
        request = urllib.request.Request(self.url, data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)["answers"]

    def route(self, state: str) -> list[str]:
        domain = self.ask(state, {"domain": self.domain_question})["domain"]["probabilities"]
        top = [d for d in sorted(domain, key=domain.get, reverse=True) if d in self.topic_criteria]
        top = top[: self.top_domains]
        within, questions = {}, {}
        for d in top:
            criteria = self.topic_criteria[d]
            if len(criteria) == 1:
                within[d] = dict.fromkeys(criteria, 1.0)
            else:
                questions[f"d{d}"] = {
                    "type": "choice",
                    "instructions": PROFILE_TOPIC_QUESTION,
                    "criteria": criteria,
                }
        if questions:
            answers = self.ask(state, questions)
            within |= {q[1:]: answers[q]["probabilities"] for q in questions}
        scores = {
            f"{d}|{t}": domain[d] * p
            for d in top
            for t, p in within[d].items()
            if t in self.topic_criteria[d]
        }
        return sorted(scores, key=scores.get, reverse=True)


class LLMRouter:
    """One LLM call that sees the topic tree (names and attributes) and picks up to max_topics topics, ranked."""

    name = "llm"

    def __init__(self, tree: ProfileTree, client, max_topics: int = 8):
        self.tree = tree
        self.client = client
        self.max_topics = max_topics
        self.tree_text = tree.tree_text()

    def validate(self, payload) -> list[str]:
        topics = payload["topics"]
        if not isinstance(topics, list) or not all(isinstance(key, str) for key in topics):
            raise ValueError("topics 必须是字符串列表")
        unknown = [key for key in topics if key not in self.tree.topics]
        if unknown:
            if len(unknown) == len(topics):
                raise ValueError(f"未知主题编号 {unknown}")
            logger.warning(f"画像路由返回了不存在的主题编号，已丢弃：{unknown}")
        known = [key for key in topics if key in self.tree.topics]
        return list(dict.fromkeys(known))[: self.max_topics]

    def route(self, state: str) -> list[str]:
        prompt = route_prompt(self.tree_text, state, self.max_topics)
        return ask_json(self.client, prompt, PROFILE_ROUTE_SCHEMA, self.validate, "harness_profile_route")
