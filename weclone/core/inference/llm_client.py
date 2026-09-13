import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from queue import Queue
from threading import Condition, Event, Thread
from typing import Any, Callable, Iterable, Literal, Protocol
from urllib.parse import urlparse

import httpx
import pyjson5
from openai import BadRequestError, OpenAI
from pydantic import BaseModel

from ._common import (
    API_TIMEOUT_SECONDS,
    OPENAI_API_MAX_RETRIES,
    OPENAI_API_BASE_DELAY,
    OPENAI_API_MAX_DELAY,
    OPENAI_API_BACKOFF_FACTOR,
    OPENAI_API_JITTER,
    calculate_retry_delay,
    logger,
    openrouter_proxy_url,
    project_root,
)
from .audit import LLMAuditCall, LLMAuditLogger

ProviderName = Literal["codex_exec", "api"]
Message = dict[str, Any]
ParsedJson = dict[str, Any] | list[Any]
RUNTIME_ROOT = project_root()
CODEX_EXEC_RUN_DIR = Path(
    os.environ.get("CODEX_EXEC_RUN_DIR", RUNTIME_ROOT / "logs" / "codex_exec")
).expanduser()


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = OPENAI_API_MAX_RETRIES
    base_delay: float = OPENAI_API_BASE_DELAY
    max_delay: float = OPENAI_API_MAX_DELAY
    backoff_factor: float = OPENAI_API_BACKOFF_FACTOR
    jitter: bool = OPENAI_API_JITTER
    retry_statuses: tuple[int, ...] | None = None
    retry_exceptions: tuple[type[Exception], ...] = (Exception,)

    def accepts(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status is not None:
            return status in self.retry_statuses if self.retry_statuses is not None else status != 400
        return isinstance(exc, self.retry_exceptions)

    def delay(self, attempt: int) -> float:
        return calculate_retry_delay(
            attempt, self.base_delay, self.max_delay, self.backoff_factor, self.jitter
        )


@dataclass
class LLMRequest:
    messages: list[Message]
    model: str | None = None
    provider: ProviderName | str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = 1024
    timeout: float | None = None
    stream: bool = False
    json_mode: bool = False
    json_schema: dict[str, Any] | None = None
    response_model: type[BaseModel] | None = None
    retry_policy: RetryPolicy | None = None
    extra_body: dict[str, Any] | None = None
    effort: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_prompt(cls, prompt: str, **kwargs: Any) -> "LLMRequest":
        return cls(messages=[{"role": "user", "content": prompt}], **kwargs)


@dataclass
class LLMResponse:
    ok: bool
    text: str | None = None
    error: str | None = None
    parsed_json: ParsedJson | None = None
    raw: Any = None
    provider: str = ""
    model: str = ""
    elapsed_s: float | None = None
    cost_usd: float | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    parsed_model: BaseModel | None = None


PromptLike = LLMRequest | str | list[Message]


class LLMClient(Protocol):
    provider: str

    def chat(self, prompt: PromptLike, **kwargs: Any) -> LLMResponse: ...

    def chat_batch(self, prompts: Iterable[PromptLike], **kwargs: Any) -> list[LLMResponse]: ...

    def generate(self, request: LLMRequest) -> LLMResponse: ...

    def chat_async(self, prompt: PromptLike, **kwargs: Any) -> Future[LLMResponse]: ...

    def generate_batch(
        self, requests: Iterable[LLMRequest], *, callback: Callable[[int, LLMResponse], None] | None = None
    ) -> list[LLMResponse]: ...

    def close(self, wait: bool = True, cancel_futures: bool = False) -> None: ...


def messages_to_prompt(messages: list[Message]) -> str:
    for message in messages:
        if not isinstance(message.get("content", ""), str):
            raise ValueError(
                "codex_exec accepts text messages only; use the API backend for multimodal input"
            )
    if len(messages) == 1 and messages[0].get("role") == "user":
        return messages[0].get("content", "")

    rendered = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        rendered.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(rendered)


def _response_audit_payload(response: LLMResponse) -> dict[str, Any]:
    return {
        "ok": response.ok,
        "text": response.text,
        "error": response.error,
        "parsed_json": response.parsed_json,
        "parsed_model": response.parsed_model,
        "provider": response.provider,
        "model": response.model,
        "elapsed_s": response.elapsed_s,
        "cost_usd": response.cost_usd,
        "finish_reason": response.finish_reason,
        "metadata": response.metadata,
    }


def _finish_audit(call: LLMAuditCall, response: LLMResponse) -> LLMResponse:
    call.finish(_response_audit_payload(response))
    return response


def _provider_request_id(raw: Any) -> str | None:
    value = getattr(raw, "_request_id", None)
    return str(value) if value else None


def _api_response_metadata(raw: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {"http_status": 200}
    usage = getattr(raw, "usage", None)
    if usage is not None:
        metadata["usage"] = usage.model_dump(mode="json") if hasattr(usage, "model_dump") else usage
    choices = getattr(raw, "choices", None)
    if choices:
        message = choices[0].message
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
        if reasoning:
            metadata["reasoning"] = reasoning if isinstance(reasoning, str) else str(reasoning)
    return metadata


def make_request(prompt: PromptLike, **kwargs: Any) -> LLMRequest:
    overrides = kwargs
    if isinstance(prompt, LLMRequest):
        return replace(prompt, **overrides) if overrides else prompt
    if isinstance(prompt, str):
        return LLMRequest.from_prompt(prompt, **overrides)
    return LLMRequest(messages=prompt, **overrides)


def parse_json_from_text(text: str | None) -> ParsedJson:
    if text is None:
        raise ValueError("empty response")
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start_obj = stripped.find("{")
    start_arr = stripped.find("[")
    starts = [pos for pos in (start_obj, start_arr) if pos >= 0]
    if not starts:
        raise ValueError(f"no JSON object or array in response: {stripped[:200]!r}")
    start = min(starts)
    opening = stripped[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(stripped)):
        char = stripped[idx]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return json.loads(stripped[start : idx + 1])
    raise ValueError(f"unbalanced JSON in response: {stripped[:200]!r}")


def _maybe_parse_response_json(request: LLMRequest, text: str | None) -> ParsedJson | None:
    if request.json_mode or request.json_schema or request.response_model:
        return parse_json_from_text(text)
    return None


def _classify_error(text: str) -> str:
    lowered = (text or "").lower()
    if any(key in lowered for key in ("overload", "rate limit", "rate_limit", "429", "too many requests")):
        return "overload/rate_limit"
    if any(key in lowered for key in ("usage limit", "quota", "exceeded", "out of credit", "insufficient")):
        return "quota"
    if any(key in lowered for key in ("auth", "unauthorized", "401", "login")):
        return "auth"
    return "other"


def _response_format_type_unavailable(exc: BadRequestError) -> bool:
    return "response_format type is unavailable" in str(exc).lower()


def _is_deepseek_api(base_url: str | None) -> bool:
    return (urlparse(base_url or "").hostname or "").lower() == "api.deepseek.com"


class BaseBatchMixin:
    provider: str
    model: str | None

    def _init_runtime(self, max_workers: int, audit_logger: LLMAuditLogger | None) -> None:
        self.max_workers = max_workers
        self.audit_logger = audit_logger or LLMAuditLogger()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._stop_event = Event()
        self._resources_closed = Event()
        self._condition = Condition()
        self._active_calls = 0
        self._closed = False

    def chat(self, prompt: PromptLike, **kwargs: Any) -> LLMResponse:
        return self.generate(make_request(prompt, **kwargs))

    def chat_async(self, prompt: PromptLike, **kwargs: Any) -> Future[LLMResponse]:
        request = make_request(prompt, **kwargs)
        with self._condition:
            if self._closed:
                future: Future[LLMResponse] = Future()
                future.set_result(self.generate(request))
                return future
            future = self._executor.submit(self.generate, request)

        def audit_cancelled(done: Future[LLMResponse]) -> None:
            if done.cancelled():
                self._cancelled_response(request)

        future.add_done_callback(audit_cancelled)
        return future

    def chat_batch(
        self,
        prompts: Iterable[PromptLike],
        *,
        callback: Callable[[int, LLMResponse], None] | None = None,
        **kwargs: Any,
    ) -> list[LLMResponse]:
        return self.generate_batch((make_request(prompt, **kwargs) for prompt in prompts), callback=callback)

    def generate_batch(
        self, requests: Iterable[LLMRequest], *, callback: Callable[[int, LLMResponse], None] | None = None
    ) -> list[LLMResponse]:
        request_list = list(requests)
        results: list[LLMResponse | None] = [None] * len(request_list)
        futures = {self.chat_async(request): i for i, request in enumerate(request_list)}
        completed: Queue[Future[LLMResponse]] = Queue()
        for future in futures:
            future.add_done_callback(completed.put)
        try:
            for _ in futures:
                future = completed.get()
                i = futures[future]
                try:
                    result = future.result()
                except CancelledError:
                    result = self._error_response(
                        request_list[i], CancelledError("request cancelled before execution")
                    )
                results[i] = result
                if callback is not None:
                    callback(i, result)
        except KeyboardInterrupt:
            self.close(wait=False, cancel_futures=True)
            raise
        return [result for result in results if result is not None]

    def _error_response(self, request: LLMRequest, exc: Exception) -> LLMResponse:
        metadata: dict[str, Any] = {**getattr(exc, "partial_metadata", {}), "error_type": type(exc).__name__}
        status = getattr(exc, "status_code", None)
        if status is not None:
            metadata["http_status"] = status
        return LLMResponse(
            ok=False,
            text=getattr(exc, "partial_text", None),
            error=f"{type(exc).__name__}: {exc}",
            provider=self.provider,
            model=request.model or self.model or "",
            metadata=metadata,
        )

    def _cancelled_response(self, request: LLMRequest) -> LLMResponse:
        with self.audit_logger.start_call(
            request=request, provider=self.provider, model=request.model or self.model or ""
        ) as call:
            return _finish_audit(call, self._error_response(request, CancelledError("client is closed")))

    def generate(self, request: LLMRequest) -> LLMResponse:
        with self._condition:
            if self._stop_event.is_set():
                return self._cancelled_response(request)
            self._active_calls += 1
        started = time.monotonic()
        try:
            with self.audit_logger.start_call(
                request=request,
                provider=self.provider,
                model=request.model or self.model or "",
                backend=self._backend_metadata(),
            ) as call:
                try:
                    result = self._generate(request, call)
                    if result.ok and (request.json_mode or request.json_schema or request.response_model):
                        try:
                            if result.finish_reason in {"length", "content_filter"}:
                                raise ValueError(
                                    f"incomplete structured response: finish_reason={result.finish_reason}"
                                )
                            result.parsed_json = _maybe_parse_response_json(request, result.text)
                            if request.response_model is not None:
                                result.parsed_model = request.response_model.model_validate(
                                    result.parsed_json
                                )
                        except Exception as exc:
                            result.ok = False
                            result.error = f"json validation failed: {type(exc).__name__}: {exc}"
                            result.metadata["error_type"] = type(exc).__name__
                except Exception as exc:
                    result = self._error_response(request, exc)
                result.elapsed_s = round(time.monotonic() - started, 3)
                return _finish_audit(call, result)
        except KeyboardInterrupt:
            self.close(wait=False, cancel_futures=True)
            raise
        finally:
            with self._condition:
                self._active_calls -= 1
                self._condition.notify_all()

    def _backend_metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    def _generate(self, request: LLMRequest, call: LLMAuditCall) -> LLMResponse:
        raise NotImplementedError

    def _close_resources(self) -> None:
        pass

    def close(self, wait: bool = True, cancel_futures: bool = False) -> None:
        with self._condition:
            already_closed = self._closed
            self._closed = True
            self._stop_event.set()
        if already_closed:
            if wait:
                self._resources_closed.wait()
            return
        self._executor.shutdown(wait=False, cancel_futures=cancel_futures)

        def finish() -> None:
            with self._condition:
                self._condition.wait_for(lambda: self._active_calls == 0)
            try:
                self._executor.shutdown(wait=True)
                self._close_resources()
            finally:
                self._resources_closed.set()

        if wait:
            finish()
        else:
            Thread(target=finish, name="llm-client-close", daemon=True).start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        interrupted = exc_type is not None and issubclass(exc_type, KeyboardInterrupt)
        self.close(wait=not interrupted, cancel_futures=interrupted)


class OpenAICompatibleClient(BaseBatchMixin):
    provider = "api"

    def __init__(
        self,
        api_key: str,
        base_url: str | None,
        model: str | None = None,
        max_workers: int = 10,
        timeout: float = API_TIMEOUT_SECONDS,
        audit_logger: LLMAuditLogger | None = None,
        retry_policy: RetryPolicy | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.retry_policy = retry_policy or RetryPolicy()
        self.is_deepseek_api = _is_deepseek_api(base_url)
        proxy_url = openrouter_proxy_url(base_url or "")
        self.http_client = httpx.Client(proxy=proxy_url, timeout=timeout) if proxy_url else None
        self.client = OpenAI(
            api_key=api_key, base_url=base_url, max_retries=0, timeout=timeout, http_client=self.http_client
        )
        self._init_runtime(max_workers, audit_logger)

    def _backend_metadata(self) -> dict[str, Any]:
        return {"base_url": self.base_url}

    def _read_response(self, raw: Any, request: LLMRequest, model: str) -> LLMResponse:
        if not request.stream:
            choice = raw.choices[0]
            text = choice.message.content
            return LLMResponse(
                ok=bool((text or "").strip()),
                text=text,
                raw=raw,
                provider=self.provider,
                model=getattr(raw, "model", None) or model,
                finish_reason=choice.finish_reason,
                metadata=_api_response_metadata(raw),
                error=None if (text or "").strip() else "empty response",
            )
        texts: list[str] = []
        reasoning: list[str] = []
        metadata: dict[str, Any] = {"http_status": 200}
        finish_reason = None
        try:
            for chunk in raw:
                if self._stop_event.is_set():
                    raise CancelledError("stream interrupted")
                if getattr(chunk, "usage", None) is not None:
                    metadata["usage"] = chunk.usage.model_dump(mode="json")
                for choice in chunk.choices:
                    if choice.index != 0:
                        continue
                    delta = choice.delta
                    if delta.content:
                        texts.append(delta.content)
                    detail = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
                    if detail:
                        reasoning.append(str(detail))
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
        except Exception as exc:
            exc.partial_text = "".join(texts)
            exc.partial_metadata = metadata
            raise
        finally:
            raw.close()
        text = "".join(texts)
        if reasoning:
            metadata["reasoning"] = "".join(reasoning)
        return LLMResponse(
            ok=bool(text.strip()),
            text=text,
            provider=self.provider,
            model=model,
            error=None if text.strip() else "empty response",
            finish_reason=finish_reason,
            metadata=metadata,
        )

    def _generate(self, request: LLMRequest, call: LLMAuditCall) -> LLMResponse:
        model = request.model or self.model
        if not model:
            raise ValueError("model is required for api backend")
        params: dict[str, Any] = {"model": model, "messages": request.messages, "stream": request.stream}
        for key in ("temperature", "top_p", "max_tokens", "timeout"):
            value = getattr(request, key)
            if value is not None:
                params[key] = value
        if request.extra_body:
            params["extra_body"] = dict(request.extra_body)
        if request.stream:
            params["stream_options"] = {"include_usage": True}
        if request.json_schema and not self.is_deepseek_api:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "llm_response", "schema": request.json_schema, "strict": True},
            }
        elif request.json_mode or request.json_schema or request.response_model:
            params["response_format"] = {"type": "json_object"}
        policy = request.retry_policy or self.retry_policy
        empty_retries = 0
        for index in range(policy.max_retries + 1):
            if self._stop_event.is_set():
                raise CancelledError("request interrupted")
            attempt = index + 1
            call.start_attempt(attempt, parameters={k: v for k, v in params.items() if k != "messages"})
            raw = None
            try:
                raw = self.client.chat.completions.create(**params)
                result = self._read_response(raw, request, model)
            except Exception as exc:
                fallback = (
                    isinstance(exc, BadRequestError)
                    and request.json_schema is not None
                    and params.get("response_format", {}).get("type") == "json_schema"
                    and _response_format_type_unavailable(exc)
                )
                retry = (
                    index < policy.max_retries
                    and (fallback or policy.accepts(exc))
                    and not self._stop_event.is_set()
                )
                delay = 0.0 if fallback else policy.delay(index) if retry else None
                call.finish_attempt(
                    attempt,
                    status="failed",
                    error=exc,
                    provider_request_id=getattr(exc, "request_id", None),
                    will_retry=retry,
                    retry_reason=("json_schema_unsupported" if fallback else "provider_error")
                    if retry
                    else None,
                    retry_delay_s=delay,
                )
                if not retry:
                    raise
                logger.warning(
                    "LLM attempt {} failed: {}; retrying in {}s", attempt, type(exc).__name__, delay
                )
                if fallback:
                    params["response_format"] = {"type": "json_object"}
                if self._stop_event.wait(delay or 0):
                    raise CancelledError("retry interrupted")
                continue
            retry_empty = (
                not result.ok
                and bool(request.json_schema or request.json_mode or request.response_model)
                and empty_retries < 1
                and index < policy.max_retries
            )
            call.finish_attempt(
                attempt,
                status="succeeded" if result.ok else "empty_response",
                response=raw if not request.stream else _response_audit_payload(result),
                provider_request_id=_provider_request_id(raw),
                will_retry=retry_empty,
                retry_reason="empty_json" if retry_empty else None,
            )
            if retry_empty:
                empty_retries += 1
                continue
            return result
        raise ValueError("retry policy max_retries must be nonnegative")

    def _close_resources(self) -> None:
        self.client.close()
        if self.http_client is not None:
            self.http_client.close()


class CodexExecClient(BaseBatchMixin):
    provider = "codex_exec"

    def __init__(
        self,
        model: str | None = None,
        effort: str = "low",
        max_workers: int = 10,
        timeout: int = 120,
        command: str = "codex",
        sandbox: str = "read-only",
        cwd: str | Path | None = None,
        extra_args: Iterable[str] | None = None,
        enable_web_search: bool = False,
        audit_logger: LLMAuditLogger | None = None,
        isolated_cwd: bool = False,
    ):
        if isolated_cwd and cwd is not None:
            raise ValueError("cwd and isolated_cwd cannot be used together")
        self.model = model
        self.effort = effort
        self.max_workers = max_workers
        self.timeout = timeout
        self.command = command
        self.sandbox = sandbox
        self.cwd = Path(cwd).resolve() if cwd else RUNTIME_ROOT
        self.requests_dir = CODEX_EXEC_RUN_DIR / "requests"
        self.extra_args = list(extra_args or [])
        self.enable_web_search = enable_web_search
        self.isolated_cwd = isolated_cwd
        self._init_runtime(max_workers, audit_logger)

    def _build_command(
        self,
        request: LLMRequest,
        *,
        output_path: Path,
        schema_path: Path | None,
        cwd: Path,
    ) -> list[str]:
        model = request.model or self.model
        if not model:
            raise ValueError("model is required for codex exec backend")

        cmd = [self.command]
        if self.enable_web_search:
            cmd.append("--search")
        cmd += [
            "exec",
            "--json",
            "--color",
            "never",
            "--ephemeral",
            "--skip-git-repo-check",
            "--cd",
            str(cwd),
            "--sandbox",
            self.sandbox,
            "--output-last-message",
            str(output_path),
            "--model",
            model,
        ]
        effort = request.effort or self.effort
        if effort:
            cmd += ["-c", f"model_reasoning_effort={json.dumps(effort)}"]
        if schema_path is not None:
            cmd += ["--output-schema", str(schema_path)]
        cmd += self.extra_args
        cmd.append("-")
        return cmd

    def _backend_metadata(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "cwd": self.cwd,
            "sandbox": self.sandbox,
            "enable_web_search": self.enable_web_search,
            "extra_args": self.extra_args,
            "isolated_cwd": self.isolated_cwd,
        }

    def _generate(self, request: LLMRequest, call: LLMAuditCall) -> LLMResponse:
        model = request.model or self.model or ""
        prompt = messages_to_prompt(request.messages)
        if request.stream:
            raise ValueError("codex_exec does not support stream=True")
        timeout = request.timeout if request.timeout is not None else self.timeout
        if not self.isolated_cwd:
            self.requests_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="request-",
            dir=None if self.isolated_cwd else self.requests_dir,
            ignore_cleanup_errors=True,
        ) as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / "last_message.txt"
            schema_path = None
            if request.json_schema:
                schema_path = tmp_path / "output_schema.json"
                schema_path.write_text(json.dumps(request.json_schema, ensure_ascii=False), encoding="utf-8")
            run_cwd = tmp_path if self.isolated_cwd else self.cwd
            cmd = self._build_command(request, output_path=output_path, schema_path=schema_path, cwd=run_cwd)
            call.start_attempt(
                1,
                parameters={
                    "model": model,
                    "effort": request.effort or self.effort,
                    "timeout": timeout,
                    "json_schema": request.json_schema,
                    **self._backend_metadata(),
                    "cwd": run_cwd,
                },
            )
            proc = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=os.name == "posix",
                )
                started = time.monotonic()
                pending_input = prompt
                while True:
                    if self._stop_event.is_set():
                        raise CancelledError("codex execution interrupted")
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(cmd, timeout)
                    try:
                        stdout, stderr = proc.communicate(input=pending_input, timeout=min(0.25, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        pending_input = None
                output_text = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else ""
            except BaseException as exc:
                if proc is not None and proc.poll() is None:
                    if os.name == "posix":
                        import signal

                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        proc.kill()
                    proc.communicate()
                call.finish_attempt(
                    1, status="timeout" if isinstance(exc, subprocess.TimeoutExpired) else "failed", error=exc
                )
                raise
        raw_response = {
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "output": output_text,
        }
        call.finish_attempt(
            1, status="succeeded" if proc.returncode == 0 else "failed", response=raw_response
        )
        metadata: dict[str, Any] = {}
        web_search_calls = 0
        event_text = ""
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = event.get("item", {})
            if event.get("type") == "item.completed" and isinstance(item, dict):
                if item.get("type") == "web_search":
                    web_search_calls += 1
                elif item.get("type") == "agent_message":
                    event_text = str(item.get("text", ""))
            if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                metadata["usage"] = {
                    str(k): v
                    for k, v in event["usage"].items()
                    if isinstance(v, int) and not isinstance(v, bool)
                }
        if self.enable_web_search:
            metadata["web_search_calls"] = web_search_calls
        text = output_text or event_text
        error = None
        if proc.returncode != 0:
            detail = f"{stderr} || {output_text or stdout[:400]}"
            error = f"returncode={proc.returncode}[{_classify_error(detail)}]: {detail[:400]}"
            metadata["error_type"] = "CodexExecError"
        elif not text:
            error = "empty response"
        return LLMResponse(
            ok=error is None,
            text=text or None,
            error=error,
            raw=raw_response,
            provider=self.provider,
            model=model,
            metadata=metadata,
        )


def normalize_provider(provider: str) -> str:
    provider = provider.lower().strip().replace("-", "_")
    if provider in {"codex", "codex_exec"}:
        return "codex_exec"
    if provider in {"openai", "deepseek", "openrouter", "openai_compatible", "api"}:
        return "api"
    raise ValueError(f"unknown llm provider: {provider}")


def load_api_config(config_path: str | Path) -> dict[str, str]:
    config_file = Path(config_path)
    config_data = pyjson5.loads(config_file.read_text(encoding="utf-8"))
    make_dataset_args = config_data.get("make_dataset_args", {})
    api_key = make_dataset_args.get("llm_api_key")
    base_url = make_dataset_args.get("base_url")
    model = make_dataset_args.get("model_name")
    missing = [
        name
        for name, value in (
            ("make_dataset_args.llm_api_key", api_key),
            ("make_dataset_args.base_url", base_url),
            ("make_dataset_args.model_name", model),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing API config fields in {config_file}: {missing}")
    return {
        "api_key": str(api_key).strip(),
        "base_url": str(base_url).strip(),
        "model": str(model).strip(),
    }


def build_llm_client(
    provider: ProviderName | str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    model_name: str | None = None,
    config_path: str | Path | None = None,
    max_workers: int = 10,
    timeout: float | None = None,
    effort: str = "low",
    command: str = "codex",
    sandbox: str = "read-only",
    audit_logger: LLMAuditLogger | None = None,
    retry_policy: RetryPolicy | None = None,
    enable_web_search: bool = False,
    cwd: str | Path | None = None,
    extra_args: Iterable[str] | None = None,
    isolated_cwd: bool = False,
) -> LLMClient:
    normalized_provider = normalize_provider(provider)
    resolved_model = model or model_name

    if normalized_provider == "codex_exec":
        return CodexExecClient(
            model=resolved_model,
            effort=effort,
            max_workers=max_workers,
            timeout=timeout or 120,
            command=command,
            sandbox=sandbox,
            audit_logger=audit_logger,
            enable_web_search=enable_web_search,
            cwd=cwd,
            extra_args=extra_args,
            isolated_cwd=isolated_cwd,
        )

    if config_path is not None:
        api_config = load_api_config(config_path)
        api_key = api_config["api_key"]
        base_url = api_config["base_url"]
        resolved_model = api_config["model"]

    missing = [
        name
        for name, value in (
            ("api_key", api_key),
            ("base_url", base_url),
            ("model", resolved_model),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing API client arguments: {missing}")

    return OpenAICompatibleClient(
        api_key=str(api_key),
        base_url=str(base_url),
        model=str(resolved_model),
        max_workers=max_workers,
        timeout=timeout or API_TIMEOUT_SECONDS,
        audit_logger=audit_logger,
        retry_policy=retry_policy,
    )


__all__ = [
    "CodexExecClient",
    "LLMClient",
    "LLMAuditLogger",
    "LLMRequest",
    "RetryPolicy",
    "LLMResponse",
    "OpenAICompatibleClient",
    "build_llm_client",
    "load_api_config",
    "make_request",
    "messages_to_prompt",
    "normalize_provider",
    "parse_json_from_text",
]
