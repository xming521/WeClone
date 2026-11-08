"""The model service on the settings page: the backend profile building uses and its request settings. They are read
from settings.jsonc and written back in place, so its comments and every other setting stay as they are. The API key
is shown to the signed-in owner like the rest, as desktop model clients do; the page hides it until asked."""

import hashlib
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Literal

import pyjson5
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from weclone.data.agent import distill_profile
from weclone.data.agent import profile_pipeline as pipeline
from weclone.server.errors import ApiError
from weclone.server.profile_build import config_path
from weclone.utils import jsonc_edit

# Where each field is kept. The API endpoint is make_dataset_args', which SFT data cleaning reads too; the other API
# request settings are agent_distill_args', next to the backend choice.
CODEX_FIELDS = {
    "model": ("codex_exec_args", "model"),
    "effort": ("codex_exec_args", "effort"),
    "batch_size": ("codex_exec_args", "batch_size"),
    "timeout": ("codex_exec_args", "timeout"),
}
API_FIELDS = {
    "base_url": ("make_dataset_args", "base_url"),
    "key": ("make_dataset_args", "llm_api_key"),
    "model": ("make_dataset_args", "model_name"),
    "context_window": ("agent_distill_args", "context_window"),
    "reasoning_effort": ("agent_distill_args", "reasoning_effort"),
    "extra_body": ("agent_distill_args", "extra_body"),
    "structured_output": ("agent_distill_args", "structured_output"),
    "batch_size": ("agent_distill_args", "batch_size"),
    "timeout": ("agent_distill_args", "timeout"),
}
PROVIDER = ("agent_distill_args", "llm_provider")
# What the backend uses for a field the file leaves out; such a field is not written while it keeps that value. A
# text field left out reads as empty, as the form shows it.
DEFAULTS = {
    "batch_size": distill_profile.DEFAULT_BATCH_SIZE,
    "timeout": distill_profile.DEFAULT_TIMEOUT,
    "model": "",
    "effort": "",
    "base_url": "",
    "key": "",
}
# The fields each request is built from; extra_body may add to a request but not change what it asks.
RESERVED_BODY = {"model", "messages", "stream"}
# The test asks for JSON the way profile building does, so it finds the response format the endpoint takes.
TEST_PROMPT = 'Reply with the JSON {"ok": true}.'
TEST_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
MODELS_TIMEOUT = 10

Provider = Literal["codex_exec", "api"]


class _Fields(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CodexSettings(_Fields):
    model: str = ""
    effort: str = ""
    batch_size: int = Field(ge=1)
    timeout: int = Field(ge=1)


class ApiSettings(_Fields):
    base_url: str = ""
    key: str = ""
    model: str = ""
    context_window: int | None = Field(default=None, ge=1)
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] | None = None
    structured_output: Literal["json_schema", "json_object"] | None = None
    batch_size: int = Field(ge=1)
    timeout: int = Field(ge=1)

    @field_validator("base_url")
    @classmethod
    def http_url(cls, value: str) -> str:
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value.rstrip("/")

    @field_validator("extra_body")
    @classmethod
    def free_fields(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value and RESERVED_BODY & value.keys():
            raise ValueError(f"extra_body cannot set {', '.join(sorted(RESERVED_BODY & value.keys()))}")
        return value or None


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Provider
    codex: CodexSettings
    api: ApiSettings


class SavedSettings(ModelSettings):
    # The file as it was read, so a change made elsewhere since is not overwritten.
    version: str


def _get(data: dict, path: tuple[str, ...]) -> Any:
    for key in path:
        data = data.get(key) if isinstance(data, dict) else None
    return data


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _read_text() -> tuple[Path, str, dict]:
    path = config_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ApiError(404, "settings_missing", f"找不到 {path}，请先按文档复制 settings.template.jsonc")
    try:
        data = pyjson5.loads(text)
    except Exception as exc:
        raise ApiError(422, "settings_unreadable", f"{path} 读取失败：{exc}")
    return path, text, data


def _value(data: dict, name: str, path: tuple[str, ...]) -> Any:
    value = _get(data, path)
    return DEFAULTS.get(name) if value is None else value


def read_settings() -> dict:
    _, text, data = _read_text()
    try:
        provider = distill_profile.normalize_llm_provider(_get(data, PROVIDER))
    except ValueError as exc:
        raise ApiError(422, "settings_unreadable", str(exc))
    return {
        "version": _digest(text),
        "provider": provider,
        "codex": {name: _value(data, name, path) for name, path in CODEX_FIELDS.items()},
        "api": {name: _value(data, name, path) for name, path in API_FIELDS.items()},
    }


def _listed_window(entry: dict) -> int | None:
    """The context window a /models entry states. Providers that list one name it differently (context_window,
    context_length, max_context_length), always with "context" in the name, except vLLM's max_model_len."""
    for name, value in entry.items():
        if ("context" in name.lower() or name == "max_model_len") and type(value) is int and value > 0:
            return value
    return None


def _changes(data: dict, settings: ModelSettings) -> list[tuple[tuple[str, ...], Any]]:
    """The values to write, None for a key to remove. A required field left empty is an error for the backend in use
    and leaves the file as it is for the other one."""
    required = {"codex_exec": ("model", "effort"), "api": ("base_url", "key", "model")}
    changes = [(PROVIDER, settings.provider)] if _get(data, PROVIDER) != settings.provider else []
    for provider, fields, values in (
        ("codex_exec", CODEX_FIELDS, settings.codex),
        ("api", API_FIELDS, settings.api),
    ):
        for name, path in fields.items():
            value = getattr(values, name)
            if value in ("", None) and name in required[provider]:
                if provider == settings.provider:
                    raise ApiError(422, "model_settings_required", f"没有填 {name}", field=name)
                continue
            if value == "":
                value = None
            current = _get(data, path)
            if value == current or (current is None and value == DEFAULTS.get(name)):
                continue
            changes.append((path, value))
    return changes


def _write(path: Path, text: str) -> None:
    """Replaces the file in one step, with the permissions it had."""
    path = path.resolve()
    mode = path.stat().st_mode & 0o777
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(text)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class ModelSettingsStore:
    def __init__(self):
        self.lock = threading.Lock()

    def save(self, settings: SavedSettings) -> dict:
        with self.lock:
            path, text, data = _read_text()
            if _digest(text) != settings.version:
                raise ApiError(409, "settings_changed", "settings.jsonc 在别处改过了，请重新打开设置页")
            changes = _changes(data, settings)
            for key, value in changes:
                text = (
                    jsonc_edit.remove_value(text, key)
                    if value is None
                    else jsonc_edit.set_value(text, key, value)
                )
            if changes:
                _write(path, text)
            return read_settings()

    def test(self, settings: ModelSettings) -> dict:
        """Codex: whether the CLI is there and signed in. API: one short request to the model as filled in, in the
        first response format the endpoint takes, and the models the endpoint lists, which not every endpoint does."""
        _, _, data = _read_text()
        if settings.provider == "codex_exec":
            problems = pipeline.codex_problems(str(_get(data, ("codex_exec_args", "command")) or "codex"))
            return {"ok": not problems, "problem": problems[0] if problems else None}
        from weclone.core.inference import OpenAICompatibleClient, RetryPolicy
        from weclone.core.inference.llm_client import STRUCTURED_OUTPUTS

        api = settings.api
        if not (api.base_url and api.key):
            raise ApiError(422, "model_settings_required", "API 的地址和 Key 都要填")
        output = None
        with OpenAICompatibleClient(
            api_key=api.key,
            base_url=api.base_url,
            model=api.model or None,
            max_workers=1,
            timeout=api.timeout,
            retry_policy=RetryPolicy(max_retries=0),
            reasoning_effort=api.reasoning_effort,
            extra_body=api.extra_body,
        ) as client:
            started = time.monotonic()
            try:
                listed = [entry.to_dict() for entry in client.client.models.list(timeout=MODELS_TIMEOUT)]
                refused = None
            except Exception as exc:
                listed, refused = None, exc
            if api.model:
                # A format the endpoint refuses comes back as 400, and the next one is tried.
                for output in STRUCTURED_OUTPUTS:
                    client.structured_output = output
                    started = time.monotonic()
                    response = client.chat(TEST_PROMPT, json_schema=TEST_SCHEMA, max_tokens=None)
                    if response.ok or response.metadata.get("http_status") != 400:
                        break
                ok, error = response.ok, response.error
            else:
                # With no model to ask yet, listing the models is the test, and they are what to choose from.
                ok, error = listed is not None, refused
        entry = next((entry for entry in listed or [] if entry.get("id") == api.model), {})
        return {
            "ok": ok,
            "seconds": round(time.monotonic() - started, 1),
            "models": None
            if listed is None
            else sorted({entry["id"] for entry in listed if entry.get("id")}),
            "structured_output": output if ok else None,
            "context_window": _listed_window(entry),
            # The provider's own words, without the key should it quote it.
            "error": None if ok else str(error).replace(api.key, "***"),
        }


def install_model_settings(app: FastAPI, *, demo: bool) -> ModelSettingsStore:
    store = ModelSettingsStore()

    def writable() -> None:
        # The demo shows no settings of the machine it runs on.
        if demo:
            raise ApiError(409, "demo_settings_fixed", "演示模式不能修改这项设置")

    @app.get("/api/settings/model")
    def model_settings():
        writable()
        return read_settings()

    @app.put("/api/settings/model")
    def model_settings_save(request: SavedSettings):
        writable()
        return store.save(request)

    @app.post("/api/settings/model/test")
    def model_settings_test(request: ModelSettings):
        writable()
        return store.test(request)

    return store
