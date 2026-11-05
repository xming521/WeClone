"""Cloud fine-tuning over an OpenAI-compatible ``/files`` + ``/fine_tuning/jobs`` API.

Only two request shapes are assumed, because that is the de-facto standard:

* ``POST {base}/files`` (multipart, ``purpose=fine-tune``) -> ``{"id": "file-..."}``
* ``POST {base}/fine_tuning/jobs`` -> ``{"id": "ftjob-...", "status": "..."}``
* ``GET  {base}/fine_tuning/jobs/{id}`` / ``/events`` / ``POST .../cancel``

Gateways that implement the same routes only under a different prefix can be
handled by pointing ``finetune.base_url`` at that prefix.  Providers without any
machine API should use ``provider = "generic"`` and upload the JSONL manually.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import FinetuneSettings
from .errors import ApiError, FineTuneError

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
TERMINAL_STATUS = frozenset({"succeeded", "failed", "cancelled"})
FAILED_STATUS = frozenset({"failed", "cancelled"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class JobState:
    file_id: str = ""
    validation_file_id: str = ""
    job_id: str = ""
    status: str = ""
    model: str = ""
    base_model: str = ""
    epochs: int = 0
    created_at: str = ""
    updated_at: str = ""
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "validation_file_id": self.validation_file_id,
            "job_id": self.job_id,
            "status": self.status,
            "model": self.model,
            "base_model": self.base_model,
            "epochs": self.epochs,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JobState":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in payload.items() if key in allowed})


def load_state(path: str | Path) -> JobState:
    target = Path(path)
    if not target.is_file():
        return JobState()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return JobState()
    return JobState.from_dict(payload) if isinstance(payload, dict) else JobState()


def save_state(path: str | Path, state: JobState) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return target


class FineTuneClient:
    def __init__(
        self,
        settings: FinetuneSettings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._key = settings.resolved_key
        self._client = httpx.Client(
            timeout=httpx.Timeout(settings.timeout, connect=30.0),
            transport=transport,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "FineTuneClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- transport ---------------------------------------------------------
    def _headers(self, *, json_body: bool = True) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._key}"}
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.settings.base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                response = self._client.request(method, url, **kwargs)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == 5:
                    break
                time.sleep(min(30.0, 2.0**attempt))
                continue
            if response.status_code in RETRYABLE_STATUS and attempt < 5:
                time.sleep(min(30.0, 2.0**attempt) * (0.6 + random.random() * 0.8))
                continue
            if response.status_code >= 400:
                raise ApiError(response.status_code, _error_message(response), response.text)
            try:
                payload = response.json()
            except ValueError as exc:
                raise ApiError(200, "响应不是合法 JSON", response.text) from exc
            return payload if isinstance(payload, dict) else {"data": payload}
        raise ApiError(0, f"{method} {url} 失败：{last_error}") from last_error

    # -- resources ---------------------------------------------------------
    def upload(self, path: str | Path, *, purpose: str = "fine-tune") -> str:
        target = Path(path)
        if not target.is_file():
            raise FineTuneError(f"训练文件不存在：{target}")
        with target.open("rb") as handle:
            payload = self._request(
                "POST",
                "/files",
                headers=self._headers(json_body=False),
                data={"purpose": purpose},
                files={"file": (target.name, handle, "application/jsonl")},
            )
        file_id = payload.get("id")
        if not isinstance(file_id, str) or not file_id:
            raise FineTuneError(f"上传失败，响应缺少 id：{json.dumps(payload, ensure_ascii=False)[:500]}")
        return file_id

    def create(
        self,
        training_file: str,
        *,
        validation_file: str | None = None,
        model: str | None = None,
        epochs: int | None = None,
        suffix: str | None = None,
        batch_size: int | str | None = None,
        learning_rate_multiplier: float | str | None = None,
    ) -> dict[str, Any]:
        settings = self.settings
        model = model or settings.base_model
        epochs = settings.epochs if epochs is None else epochs
        suffix = settings.suffix if suffix is None else suffix
        batch_size = settings.batch_size if batch_size is None else batch_size
        lrm = (
            settings.learning_rate_multiplier
            if learning_rate_multiplier is None
            else learning_rate_multiplier
        )

        hyperparameters: dict[str, Any] = {"n_epochs": int(epochs)}
        if batch_size != "auto":
            hyperparameters["batch_size"] = batch_size
        if lrm != "auto":
            hyperparameters["learning_rate_multiplier"] = lrm

        body: dict[str, Any] = {"model": model, "training_file": training_file}
        if validation_file:
            body["validation_file"] = validation_file
        if suffix:
            body["suffix"] = suffix
        if settings.api_style == "method":
            body["method"] = {
                "type": "supervised",
                "supervised": {"hyperparameters": hyperparameters},
            }
        else:
            body["hyperparameters"] = hyperparameters

        payload = self._request("POST", "/fine_tuning/jobs", headers=self._headers(), json=body)
        job_id = payload.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise FineTuneError(f"创建任务失败：{json.dumps(payload, ensure_ascii=False)[:500]}")
        return payload

    def retrieve(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/fine_tuning/jobs/{job_id}", headers=self._headers())

    def events(self, job_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        payload = self._request(
            "GET", f"/fine_tuning/jobs/{job_id}/events", headers=self._headers(), params={"limit": limit}
        )
        items = payload.get("data")
        return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/fine_tuning/jobs/{job_id}/cancel", headers=self._headers())

    def wait(
        self,
        job_id: str,
        *,
        interval: float | None = None,
        on_poll: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        delay = self.settings.poll_interval if interval is None else interval
        last_status = ""
        while True:
            job = self.retrieve(job_id)
            status = str(job.get("status") or "")
            if on_poll is not None and status != last_status:
                on_poll(job)
            last_status = status
            if status in TERMINAL_STATUS:
                if status in FAILED_STATUS:
                    raise FineTuneError(f"任务 {job_id} 结束状态为 {status}", job=job)
                return job
            time.sleep(max(0.0, delay))


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500] or response.reason_phrase
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
        if payload.get("message"):
            return str(payload["message"])
    return response.text[:500]
