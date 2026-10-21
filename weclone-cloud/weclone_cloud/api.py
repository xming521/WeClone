"""OpenAI-compatible chat client: retries, structured output, on-disk cache.

Design notes
------------
* Only ``/chat/completions`` is used, so any OpenAI-compatible gateway works
  (DeepSeek, Qwen DashScope compatible-mode, vLLM, OpenRouter, ...).
* Retries cover transport errors, 429 and 5xx; 4xx are surfaced immediately
  because they usually mean a bad key or a bad request body.
* ``json_schema`` is attempted first and transparently downgraded to
  ``json_object`` when the gateway rejects the field, which is common.
* Cache keys are content addressed (model + messages + schema), so re-running a
  scoring pass after a crash costs nothing for already-scored rows.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import httpx

from .config import ApiSettings
from .errors import ApiError

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_SCHEMA_FALLBACK_HINTS = ("response_format", "json_schema", "unsupported", "not support")


@dataclass
class Completion:
    """One assistant reply plus bookkeeping used for cost/quality reporting."""

    text: str
    data: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    finish_reason: str | None = None
    reasoning: str = ""
    cached: bool = False
    attempts: int = 1
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.text)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "usage": self.usage,
            "model": self.model,
            "finish_reason": self.finish_reason,
        }


def _scan_json_object(text: str) -> dict[str, Any] | None:
    """Return the first balanced ``{...}`` block that parses, ignoring braces in strings."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    return value if isinstance(value, dict) else None
        start = text.find("{", start + 1)
    return None


def parse_json_object(text: str | None) -> dict[str, Any] | None:
    """Parse a JSON object out of a model reply (handles fenced blocks and prose)."""
    if not text:
        return None
    body = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", body, re.S)
    if fenced:
        body = fenced.group(1).strip()
    try:
        value = json.loads(body)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    return _scan_json_object(body)


class CompletionCache:
    """Tiny content-addressed JSON cache (one file per key, atomic replace)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        target = self._path(key)
        with self._lock:
            tmp = target.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                tmp.replace(target)
            except OSError:
                # 缓存写失败不应让整批任务失败
                tmp.unlink(missing_ok=True)


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500] or response.reason_phrase
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)
        if error:
            return str(error)
        if payload.get("message"):
            return str(payload["message"])
    return response.text[:500]


class ChatClient:
    """Blocking client; safe to use from a thread pool."""

    def __init__(
        self,
        settings: ApiSettings,
        *,
        cache: CompletionCache | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._key = settings.resolved_key
        self._cache = cache
        self._client = httpx.Client(
            timeout=httpx.Timeout(settings.timeout, connect=min(15.0, settings.timeout)),
            transport=transport,
            follow_redirects=True,
        )
        # 线程安全的用量累计，供 CLI 打印成本估算
        self._usage_lock = threading.Lock()
        self.usage_total: dict[str, int] = {
            "requests": 0,
            "cache_hits": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def record_usage(self, usage: dict[str, Any] | None, *, cached: bool = False) -> None:
        with self._usage_lock:
            if cached:
                self.usage_total["cache_hits"] += 1
                return
            self.usage_total["requests"] += 1
            for key in ("prompt_tokens", "completion_tokens"):
                value = (usage or {}).get(key)
                if isinstance(value, int):
                    self.usage_total[key] += value

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ChatClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- requests ----------------------------------------------------------
    @property
    def url(self) -> str:
        return f"{self.settings.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        headers.update(self.settings.extra_headers)
        return headers

    def _payload(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": list(messages),
            "temperature": self.settings.temperature if temperature is None else temperature,
        }
        limit = self.settings.max_tokens if max_tokens is None else max_tokens
        if limit:
            payload["max_tokens"] = int(limit)
        if json_schema is not None and self.settings.json_mode == "schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "weclone_result", "strict": True, "schema": json_schema},
            }
        elif json_schema is not None and self.settings.json_mode == "object":
            payload["response_format"] = {"type": "json_object"}
        thinking = self.settings.resolved_thinking
        if thinking in {"enabled", "disabled"}:
            payload["thinking"] = {"type": thinking}
        if self.settings.extra_body:
            payload.update(self.settings.extra_body)
        return payload

    def _cache_key(self, payload: dict[str, Any]) -> str:
        basis = json.dumps(
            {
                "base_url": self.settings.base_url,
                "payload": {key: value for key, value in payload.items() if key != "messages"},
                "messages": payload.get("messages"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool | None = None,
    ) -> Completion:
        payload = self._payload(
            messages, json_schema=json_schema, temperature=temperature, max_tokens=max_tokens
        )
        cache_enabled = self._cache is not None
        if use_cache is not None:
            cache_enabled = use_cache and self._cache is not None
        key = self._cache_key(payload)
        if cache_enabled and self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                completion = Completion(
                    text=str(hit.get("text") or ""),
                    data=parse_json_object(str(hit.get("text") or "")),
                    usage=dict(hit.get("usage") or {}),
                    model=str(hit.get("model") or self.settings.model),
                    finish_reason=hit.get("finish_reason"),
                    cached=True,
                )
                self.record_usage(completion.usage, cached=True)
                return completion

        started = time.perf_counter()
        completion = self._execute(payload)
        completion.elapsed = time.perf_counter() - started
        self.record_usage(completion.usage)
        if cache_enabled and self._cache is not None:
            self._cache.put(key, completion.as_dict())
        return completion

    def _execute(self, payload: dict[str, Any]) -> Completion:
        try:
            data = self._post(payload)
        except ApiError as exc:
            downgraded = self._downgrade(payload, exc)
            if downgraded is None:
                raise
            data = self._post(downgraded)
            payload = downgraded
        completion = self._parse(data)
        if completion.text.strip() or "response_format" not in payload:
            return completion
        # DeepSeek 文档明确：JSON Output 偶尔返回空 content。去掉 response_format 重试一次，
        # 让模型按 prompt 里的示例直接吐 JSON（parse_json_object 能从文本里抠出来）。
        plain = dict(payload)
        plain.pop("response_format", None)
        retry = self._parse(self._post(plain))
        retry.attempts = completion.attempts + 1
        return retry if retry.text.strip() else completion

    def _downgrade(self, payload: dict[str, Any], exc: ApiError) -> dict[str, Any] | None:
        response_format = payload.get("response_format") or {}
        if exc.status != 400 or response_format.get("type") != "json_schema":
            return None
        lowered = (exc.body or str(exc)).lower()
        if not any(hint in lowered for hint in _SCHEMA_FALLBACK_HINTS):
            return None
        degraded = dict(payload)
        degraded["response_format"] = {"type": "json_object"}
        return degraded

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempts = 0
        last_error: Exception | None = None
        while attempts < max(1, self.settings.max_retries):
            attempts += 1
            try:
                response = self._client.post(self.url, json=payload, headers=self._headers())
            except httpx.HTTPError as exc:
                last_error = exc
                if attempts >= self.settings.max_retries:
                    break
                time.sleep(self._delay(attempts))
                continue

            if response.status_code in RETRYABLE_STATUS and attempts < self.settings.max_retries:
                time.sleep(self._delay(attempts, response))
                continue
            if response.status_code >= 400:
                raise ApiError(response.status_code, _error_message(response), response.text)
            try:
                data = response.json()
            except ValueError as exc:
                raise ApiError(200, "响应不是合法 JSON", response.text) from exc
            if not isinstance(data, dict):
                raise ApiError(200, "响应顶层不是对象", response.text)
            return data

        raise ApiError(0, f"请求失败（重试 {attempts} 次）：{last_error}") from last_error

    @staticmethod
    def _delay(attempt: int, response: httpx.Response | None = None) -> float:
        if response is not None:
            header = response.headers.get("retry-after")
            if header:
                try:
                    return max(0.0, float(header))
                except ValueError:
                    pass
        base = min(30.0, 1.5**attempt)
        return base * (0.6 + random.random() * 0.8)

    def _parse(self, data: dict[str, Any]) -> Completion:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ApiError(200, "响应缺少 choices", json.dumps(data, ensure_ascii=False)[:1000])
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        text = message.get("content")
        if isinstance(text, list):  # 少数网关返回 content parts
            text = "".join(str(part.get("text") or "") for part in text if isinstance(part, dict))
        text = str(text or "")
        return Completion(
            text=text,
            data=parse_json_object(text),
            usage=dict(data.get("usage") or {}),
            model=str(data.get("model") or self.settings.model),
            finish_reason=choice.get("finish_reason"),
            reasoning=str(message.get("reasoning_content") or ""),
            attempts=1,
        )


def build_client(settings: ApiSettings, *, cache_dir: str | Path | None = None) -> ChatClient:
    cache = CompletionCache(cache_dir) if cache_dir else None
    return ChatClient(settings, cache=cache)
