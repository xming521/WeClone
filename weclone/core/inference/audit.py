from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock, get_ident
from typing import Any, Mapping
from uuid import uuid4

from ._common import logger, project_root


AUDIT_LOG_DIR_ENV = "LLM_AUDIT_LOG_DIR"
DEFAULT_AUDIT_LOG_DIR = Path(
    os.environ.get(
        AUDIT_LOG_DIR_ENV, os.environ.get("LLM_REQUEST_LOG_DIR", project_root() / "logs" / "llm_audit")
    )
).expanduser()
AUDIT_SCHEMA_VERSION = 1
_AUDIT_WRITE_LOCK = Lock()
_REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "cookie",
    "credentials",
    "password",
    "proxy_authorization",
    "refresh_token",
    "secret",
    "set_cookie",
    "access_token",
}
_SENSITIVE_SUFFIXES = (
    "_api_key",
    "_credential",
    "_password",
    "_secret",
    "_access_token",
    "_refresh_token",
)


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_SUFFIXES)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        schema = getattr(value, "model_json_schema", None)
        return (
            {"type": f"{value.__module__}.{value.__qualname__}", "schema": schema()}
            if schema
            else f"{value.__module__}.{value.__qualname__}"
        )
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(mode="json"))
        except TypeError:
            return _jsonable(model_dump())
    if isinstance(value, Mapping):
        return {
            str(key): _REDACTED if _is_sensitive_key(key) else _jsonable(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return str(value)


def _exception_payload(exc: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    for attribute in ("status_code", "request_id", "body"):
        value = getattr(exc, attribute, None)
        if value is not None:
            payload[attribute] = _jsonable(value)
    return payload


class LLMAuditLogger:
    """Append-only JSONL audit sink for synchronous LLM calls."""

    def __init__(
        self,
        directory: str | Path | None = None,
        *,
        enabled: bool = True,
        strict: bool = False,
    ) -> None:
        self.directory = Path(directory or DEFAULT_AUDIT_LOG_DIR).expanduser()
        self.enabled = enabled
        self.strict = strict

    def start_call(
        self,
        *,
        request: Any,
        provider: str,
        model: str,
        backend: Mapping[str, Any] | None = None,
    ) -> "LLMAuditCall":
        return LLMAuditCall(
            logger=self,
            request=request,
            provider=provider,
            model=model,
            backend=backend,
        )

    def _write(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return

        now = datetime.now().astimezone()
        record = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "timestamp": now.isoformat(timespec="milliseconds"),
            "pid": os.getpid(),
            "thread_id": get_ident(),
            **payload,
        }
        try:
            line = (json.dumps(_jsonable(record), ensure_ascii=False) + "\n").encode("utf-8")
            log_path = self.directory / f"{now:%Y-%m-%d}.jsonl"
            with _AUDIT_WRITE_LOCK:
                self.directory.mkdir(parents=True, exist_ok=True)
                fd = os.open(
                    log_path,
                    os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                    0o600,
                )
                try:
                    fchmod = getattr(os, "fchmod", None)
                    if fchmod is not None:
                        fchmod(fd, 0o600)
                    remaining = memoryview(line)
                    while remaining:
                        written = os.write(fd, remaining)
                        if written <= 0:
                            raise OSError("failed to append LLM audit event")
                        remaining = remaining[written:]
                finally:
                    os.close(fd)
        except Exception as exc:
            if self.strict:
                raise
            logger.warning(f"Failed to write LLM audit event: {type(exc).__name__}: {exc}")


class LLMAuditCall:
    def __init__(
        self,
        *,
        logger: LLMAuditLogger,
        request: Any,
        provider: str,
        model: str,
        backend: Mapping[str, Any] | None,
    ) -> None:
        self.logger = logger
        self.call_id = uuid4().hex
        self.provider = provider
        self.model = model
        self._started_at = time.monotonic()
        self._attempt_started_at: dict[int, float] = {}
        self._sequence = 0
        self._finished = False
        self._emit(
            "call.started",
            request=request,
            backend=backend or {},
        )

    def __enter__(self) -> "LLMAuditCall":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> None:
        if self._finished:
            return
        if exc is not None:
            self.finish_exception(exc)
            return
        self.finish(
            {
                "ok": False,
                "error": {
                    "type": "IncompleteAuditCall",
                    "message": "LLM call exited without a final result",
                },
            }
        )

    def start_attempt(
        self,
        attempt: int,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        self._attempt_started_at[attempt] = time.monotonic()
        self._emit(
            "attempt.started",
            attempt=attempt,
            parameters=parameters or {},
        )

    def finish_attempt(
        self,
        attempt: int,
        *,
        status: str,
        response: Any = None,
        error: BaseException | Mapping[str, Any] | None = None,
        provider_request_id: str | None = None,
        will_retry: bool = False,
        retry_reason: str | None = None,
        retry_delay_s: float | None = None,
    ) -> None:
        started_at = self._attempt_started_at.pop(attempt, self._started_at)
        payload: dict[str, Any] = {
            "attempt": attempt,
            "status": status,
            "elapsed_s": round(time.monotonic() - started_at, 3),
            "will_retry": will_retry,
        }
        if response is not None:
            payload["response"] = response
        if isinstance(error, BaseException):
            payload["error"] = _exception_payload(error)
        elif error is not None:
            payload["error"] = error
        if provider_request_id:
            payload["provider_request_id"] = provider_request_id
        if retry_reason:
            payload["retry_reason"] = retry_reason
        if retry_delay_s is not None:
            payload["retry_delay_s"] = round(retry_delay_s, 3)
        self._emit("attempt.finished", **payload)

    def finish(self, result: Mapping[str, Any]) -> None:
        if self._finished:
            return
        self._finished = True
        self._emit(
            "call.finished",
            elapsed_s=round(time.monotonic() - self._started_at, 3),
            result=result,
        )

    def finish_exception(self, exc: BaseException) -> None:
        self.finish({"ok": False, "error": _exception_payload(exc)})

    def _emit(self, event: str, **payload: Any) -> None:
        self._sequence += 1
        self.logger._write(
            {
                "event": event,
                "sequence": self._sequence,
                "call_id": self.call_id,
                "provider": self.provider,
                "model": self.model,
                **payload,
            }
        )


__all__ = [
    "AUDIT_LOG_DIR_ENV",
    "AUDIT_SCHEMA_VERSION",
    "DEFAULT_AUDIT_LOG_DIR",
    "LLMAuditCall",
    "LLMAuditLogger",
]
