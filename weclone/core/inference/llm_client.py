import json
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Literal, Protocol

import httpx
import pyjson5
from openai import OpenAI

from .online_infer import (
    API_TIMEOUT_SECONDS,
    OPENAI_API_MAX_RETRIES,
    _calculate_retry_delay,
    _openrouter_proxy_url,
)
from weclone.utils.log import logger


ProviderName = Literal["codex_exec", "api"]
Message = dict[str, str]
ParsedJson = dict[str, Any] | list[Any]
LLM_REQUEST_LOG_DIR = Path(__file__).resolve().parents[3] / "logs" / "llm_requests"
_LLM_REQUEST_LOG_LOCK = Lock()


@dataclass
class LLMRequest:
    messages: list[Message]
    model: str | None = None
    provider: ProviderName | str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = 1024
    timeout: int | None = None
    stream: bool = False
    json_mode: bool = False
    json_schema: dict[str, Any] | None = None
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


PromptLike = LLMRequest | str | list[Message]


class LLMClient(Protocol):
    provider: str

    def chat(self, prompt: PromptLike, **kwargs: Any) -> LLMResponse:
        ...

    def chat_batch(self, prompts: Iterable[PromptLike], **kwargs: Any) -> list[LLMResponse]:
        ...

    def generate(self, request: LLMRequest) -> LLMResponse:
        ...

    def generate_batch(self, requests: Iterable[LLMRequest]) -> list[LLMResponse]:
        ...

    def close(self) -> None:
        ...


def messages_to_prompt(messages: list[Message]) -> str:
    if len(messages) == 1 and messages[0].get("role") == "user":
        return messages[0].get("content", "")

    rendered = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        rendered.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(rendered)


def _log_llm_request(request: LLMRequest, *, provider: str, model: str) -> None:
    now = datetime.now().astimezone()
    log_path = LLM_REQUEST_LOG_DIR / f"{now:%Y-%m-%d}.log"
    payload = {
        "timestamp": now.isoformat(timespec="milliseconds"),
        "provider": provider,
        "model": model,
        "raw_input": {
            "messages": request.messages,
            "prompt": messages_to_prompt(request.messages),
        },
        "request": asdict(request),
    }
    try:
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with _LLM_REQUEST_LOG_LOCK:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:
        logger.warning(f"Failed to write LLM request log: {type(exc).__name__}: {exc}")


def make_request(prompt: PromptLike, **kwargs: Any) -> LLMRequest:
    overrides = {key: value for key, value in kwargs.items() if value is not None}
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
    if request.json_mode or request.json_schema:
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


class BaseBatchMixin:
    max_workers: int

    def chat(self, prompt: PromptLike, **kwargs: Any) -> LLMResponse:
        return self.generate(make_request(prompt, **kwargs))

    def chat_batch(self, prompts: Iterable[PromptLike], **kwargs: Any) -> list[LLMResponse]:
        return self.generate_batch(make_request(prompt, **kwargs) for prompt in prompts)

    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    def generate_batch(self, requests: Iterable[LLMRequest]) -> list[LLMResponse]:
        request_list = list(requests)
        if not request_list:
            return []

        results: list[LLMResponse | None] = [None] * len(request_list)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self.generate, request): idx for idx, request in enumerate(request_list)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    request = request_list[idx]
                    results[idx] = LLMResponse(
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                        provider=request.provider or getattr(self, "provider", ""),
                        model=request.model or getattr(self, "model", "") or "",
                    )
        return [result for result in results if result is not None]


class OpenAICompatibleClient(BaseBatchMixin):
    provider = "api"

    def __init__(
        self,
        api_key: str,
        base_url: str | None,
        model: str | None = None,
        max_workers: int = 10,
        timeout: int = API_TIMEOUT_SECONDS,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_workers = max_workers
        self.timeout = timeout
        proxy_url = _openrouter_proxy_url(base_url or "")
        self.http_client = httpx.Client(proxy=proxy_url, timeout=timeout) if proxy_url else None
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=timeout,
            http_client=self.http_client,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        model = self.model
        if not model:
            raise ValueError("model is required for api backend")
        _log_llm_request(request, provider=self.provider, model=model)

        params: dict[str, Any] = {
            "model": model,
            "messages": request.messages,
            "stream": request.stream,
            "max_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.extra_body:
            params["extra_body"] = request.extra_body
        if request.timeout is not None:
            params["timeout"] = request.timeout
        if request.json_schema:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "llm_response",
                    "schema": request.json_schema,
                    "strict": True,
                },
            }
        elif request.json_mode:
            params["response_format"] = {"type": "json_object"}

        t0 = time.time()
        for attempt in range(OPENAI_API_MAX_RETRIES + 1):
            try:
                raw = self.client.chat.completions.create(**params)
                elapsed = round(time.time() - t0, 3)
                choice = raw.choices[0]
                text = choice.message.content
                if text is None:
                    return LLMResponse(
                        ok=False,
                        error="empty response",
                        raw=raw,
                        provider=self.provider,
                        model=model,
                        elapsed_s=elapsed,
                        finish_reason=choice.finish_reason,
                    )
                try:
                    parsed_json = _maybe_parse_response_json(request, text)
                except Exception as exc:
                    return LLMResponse(
                        ok=False,
                        text=text,
                        error=f"json parse fail: {type(exc).__name__}: {exc}",
                        raw=raw,
                        provider=self.provider,
                        model=model,
                        elapsed_s=elapsed,
                        finish_reason=choice.finish_reason,
                    )
                return LLMResponse(
                    ok=True,
                    text=text.strip(),
                    parsed_json=parsed_json,
                    raw=raw,
                    provider=self.provider,
                    model=model,
                    elapsed_s=elapsed,
                    finish_reason=choice.finish_reason,
                )
            except Exception as exc:
                if attempt < OPENAI_API_MAX_RETRIES:
                    delay = _calculate_retry_delay(attempt)
                    logger.warning(
                        f"OpenAI-compatible API failed: {type(exc).__name__}: {exc}; "
                        f"retry {attempt + 1}/{OPENAI_API_MAX_RETRIES + 1} after {delay:.2f}s"
                    )
                    time.sleep(delay)
                    continue
                elapsed = round(time.time() - t0, 3)
                return LLMResponse(
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    provider=self.provider,
                    model=model,
                    elapsed_s=elapsed,
                )

    def close(self) -> None:
        self.client.close()
        if self.http_client is not None:
            self.http_client.close()

    def __enter__(self) -> "OpenAICompatibleClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


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
    ):
        self.model = model
        self.effort = effort
        self.max_workers = max_workers
        self.timeout = timeout
        self.command = command
        self.sandbox = sandbox
        self.cwd = Path(cwd) if cwd else None
        self.extra_args = list(extra_args or [])

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

        cmd = [
            self.command,
            "exec",
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

    def generate(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.model or ""
        prompt = messages_to_prompt(request.messages)
        timeout = request.timeout or self.timeout
        _log_llm_request(request, provider=self.provider, model=model)
        t0 = time.time()

        with tempfile.TemporaryDirectory(prefix="codex-exec-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / "last_message.txt"
            schema_path = None
            if request.json_schema:
                schema_path = tmp_path / "output_schema.json"
                schema_path.write_text(
                    json.dumps(request.json_schema, ensure_ascii=False),
                    encoding="utf-8",
                )
            cmd = self._build_command(
                request,
                output_path=output_path,
                schema_path=schema_path,
                cwd=self.cwd or tmp_path,
            )

            try:
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return LLMResponse(
                    ok=False,
                    error=f"timeout>{timeout}s",
                    provider=self.provider,
                    model=model,
                    elapsed_s=round(time.time() - t0, 3),
                )

            output_text = ""
            if output_path.exists():
                output_text = output_path.read_text(encoding="utf-8").strip()

        elapsed = round(time.time() - t0, 3)
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

        if proc.returncode != 0:
            detail = output_text or stdout[:400]
            blob = f"{stderr} || {detail}"
            return LLMResponse(
                ok=False,
                error=f"returncode={proc.returncode}[{_classify_error(blob)}]: {blob[:400]}",
                raw={"stdout": stdout, "stderr": stderr, "output": output_text},
                provider=self.provider,
                model=model,
                elapsed_s=elapsed,
            )

        text = output_text or stdout
        if not text:
            return LLMResponse(
                ok=False,
                error="empty response",
                raw={"stdout": stdout, "stderr": stderr},
                provider=self.provider,
                model=model,
                elapsed_s=elapsed,
            )
        try:
            parsed_json = _maybe_parse_response_json(request, text)
        except Exception as exc:
            return LLMResponse(
                ok=False,
                text=text,
                error=f"json parse fail: {type(exc).__name__}: {exc}",
                raw={"stdout": stdout, "stderr": stderr, "output": output_text},
                provider=self.provider,
                model=model,
                elapsed_s=elapsed,
            )

        return LLMResponse(
            ok=True,
            text=text.strip(),
            parsed_json=parsed_json,
            raw={"stdout": stdout, "stderr": stderr, "output": output_text},
            provider=self.provider,
            model=model,
            elapsed_s=elapsed,
        )

    def close(self) -> None:
        return None

    def __enter__(self) -> "CodexExecClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


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
    timeout: int | None = None,
    effort: str = "low",
    command: str = "codex",
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
        )

    if config_path is None:
        raise ValueError("config_path is required for api backend")
    api_config = load_api_config(config_path)
    return OpenAICompatibleClient(
        api_key=api_config["api_key"],
        base_url=api_config["base_url"],
        model=api_config["model"],
        max_workers=max_workers,
        timeout=timeout or API_TIMEOUT_SECONDS,
    )


__all__ = [
    "CodexExecClient",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "OpenAICompatibleClient",
    "build_llm_client",
    "load_api_config",
    "make_request",
    "messages_to_prompt",
    "normalize_provider",
    "parse_json_from_text",
]
