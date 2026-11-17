"""Configuration for the cloud pipeline.

Secrets never live in the config file: each section names an environment variable
(``api_key_env``).  A literal ``api_key`` field exists only for throwaway local
experiments and is reported as a warning by ``weclone-cloud doctor``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import pyjson5
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ConfigError
from .providers import API_PROVIDER_REGISTRY, MINIMAX_PROVIDER

DEFAULT_CONFIG_NAME = "cloud.jsonc"


class _Section(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ApiSettings(_Section):
    """OpenAI-compatible chat endpoint used for scoring / rewriting / any LLM step."""

    provider: str = MINIMAX_PROVIDER
    region: str = "global_en"
    base_url: str = "https://api.minimax.io/v1"
    model: str = "MiniMax-M3.1-Flash"
    api_key_env: str = "WECLONE_API_KEY"
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: float = 60.0
    max_concurrency: int = 8
    max_retries: int = 5
    # schema: response_format=json_schema -> object: json_object -> none: plain text
    json_mode: Literal["schema", "object", "none"] = "object"
    # DeepSeek 思考模式默认开启，会拖慢打分并吃掉 max_tokens；auto = 命中 deepseek 时关闭
    thinking: Literal["auto", "enabled", "disabled"] = "auto"
    # 原样并入请求体，用于各家网关的私有参数（会覆盖 thinking 的结果）
    extra_body: dict[str, Any] = Field(default_factory=dict)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    # 计价用（仅用于成本估算）：DeepSeek 以人民币/百万 token 计
    price_input_per_million: float | None = None
    price_output_per_million: float | None = None
    price_currency: str = "CNY"

    @model_validator(mode="before")
    @classmethod
    def _apply_provider_defaults(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "provider" not in data and ("base_url" in data or "model" in data):
            data["provider"] = "custom"
            return data
        provider = str(data.get("provider") or MINIMAX_PROVIDER)
        preset = API_PROVIDER_REGISTRY.get(provider)
        if preset is None:
            return data
        region = str(data.get("region") or preset["default_region"])
        endpoint = preset["endpoints"].get(region)
        if endpoint is None:
            supported = ", ".join(sorted(preset["endpoints"]))
            raise ValueError(f"Unsupported {provider} region {region!r}; choose one of: {supported}")
        data.setdefault("provider", provider)
        data.setdefault("region", region)
        data.setdefault("base_url", endpoint)
        data.setdefault("model", preset["default_model"])

        model = preset["models"].get(data["model"])
        if model is not None:
            pricing = model["pricing_usd_per_million_tokens"]
            data.setdefault("price_input_per_million", pricing["input"])
            data.setdefault("price_output_per_million", pricing["output"])
            data.setdefault("price_currency", "USD")
            effort = model.get("reasoning_effort_default")
            if effort:
                extra_body = dict(data.get("extra_body") or {})
                extra_body.setdefault("reasoning_effort", effort)
                data["extra_body"] = extra_body
        return data

    @field_validator("base_url")
    @classmethod
    def _strip_slash(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value:
            raise ValueError("api.base_url 不能为空")
        return value

    @property
    def resolved_key(self) -> str:
        key = (os.environ.get(self.api_key_env) or self.api_key or "").strip()
        if not key:
            raise ConfigError(
                f"缺少 API Key：请设置环境变量 {self.api_key_env}"
                f'（PowerShell: $env:{self.api_key_env}="sk-..."）。'
            )
        return key

    @property
    def key_source(self) -> str:
        if (os.environ.get(self.api_key_env) or "").strip():
            return f"env:{self.api_key_env}"
        return "config:api_key" if self.api_key else "missing"

    @property
    def resolved_thinking(self) -> str:
        """``enabled`` / ``disabled`` / ``omit``（网关不认识该字段时不要发）。"""
        if self.thinking != "auto":
            return self.thinking
        return "disabled" if "deepseek" in self.base_url.lower() else "omit"

    def estimate_cost(self, usage: dict[str, Any]) -> float | None:
        """按配置单价估算单次调用成本；没配价格则返回 None。"""
        inp = usage.get("prompt_tokens")
        out = usage.get("completion_tokens")
        if not isinstance(inp, int) or not isinstance(out, int):
            return None
        if self.price_input_per_million is None and self.price_output_per_million is None:
            return None
        cost = 0.0
        if self.price_input_per_million is not None:
            cost += inp * self.price_input_per_million / 1_000_000
        if self.price_output_per_million is not None:
            cost += out * self.price_output_per_million / 1_000_000
        return round(cost, 6)


class ScoringSettings(_Section):
    min_score: int = 2
    # 单条对话送入打分的最大字符数，超出后保留头尾
    max_chars: int = 6000
    cache_dir: str = ".cache/weclone-cloud"
    use_cache: bool = True
    # 每次请求之间的全局最小间隔（秒），应对免费额度限流
    request_interval: float = 0.0

    @field_validator("min_score")
    @classmethod
    def _score_range(cls, value: int) -> int:
        if not 1 <= value <= 5:
            raise ValueError("scoring.min_score 必须在 1..5")
        return value


class FinetuneSettings(_Section):
    """Cloud fine-tuning endpoint.  ``generic`` only exports JSONL for console upload."""

    provider: Literal["openai", "generic"] = "generic"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    base_model: str = "gpt-4o-mini-2024-07-18"
    suffix: str = "weclone"
    # legacy: hyperparameters at top level; method: method.supervised.hyperparameters
    api_style: Literal["legacy", "method"] = "method"
    epochs: int = 3
    batch_size: int | Literal["auto"] = "auto"
    learning_rate_multiplier: float | Literal["auto"] = "auto"
    timeout: float = 180.0
    poll_interval: float = 20.0
    state_file: str = "finetune_state.json"

    @field_validator("base_url")
    @classmethod
    def _strip_slash(cls, value: str) -> str:
        return value.strip().rstrip("/")

    @property
    def resolved_key(self) -> str:
        key = (os.environ.get(self.api_key_env) or self.api_key or "").strip()
        if not key:
            raise ConfigError(f"缺少微调 API Key：请设置环境变量 {self.api_key_env}。")
        return key


class ServeSettings(_Section):
    host: str = "127.0.0.1"
    port: int = 8010
    # 留空则从 finetune_state.json 的 model 字段读取
    model: str | None = None
    system_prompt: str = "请你扮演一名人类，不要说自己是人工智能。"
    # 指向 weclone-cloud persona 生成的文件；存在时优先于 system_prompt
    system_prompt_file: str | None = None
    # replace: 丢掉客户端自带的 system，强制用人设（推荐，LangBot 等框架会塞自己的提示词）
    # prepend: 保留客户端 system，把人设放在最前面
    # if_missing: 只有客户端没给 system 时才注入
    system_mode: Literal["replace", "prepend", "if_missing"] = "replace"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    # 回复前随机停顿（秒）。微信等平台短时间内连续发消息容易触发风控，
    # 给一个像人的思考/打字间隔能显著降低风险；0 表示不延迟。
    reply_delay_min: float = 1.5
    reply_delay_max: float = 4.0
    # 单个回复拆成多条消息时，每条之间再等多久（由调用方决定是否拆分，这里只提供节奏参考）
    reply_delay_per_segment: float = 0.0
    # 对外暴露时必须开启，防止别人白嫖你的微调额度
    require_api_key: bool = True
    api_key_env: str = "WECLONE_SERVE_KEY"
    api_key: str | None = None

    @property
    def resolved_key(self) -> str | None:
        if not self.require_api_key:
            return None
        key = (os.environ.get(self.api_key_env) or self.api_key or "").strip()
        if not key:
            raise ConfigError(
                f"serve.require_api_key=true 但未设置 {self.api_key_env}；本地自用可改为 false。"
            )
        return key


class CloudConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api: ApiSettings
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    finetune: FinetuneSettings = Field(default_factory=FinetuneSettings)
    serve: ServeSettings = Field(default_factory=ServeSettings)
    source_path: str | None = Field(default=None, exclude=True)

    def redacted(self) -> dict[str, Any]:
        """Dump for logs / doctor output without secret values."""
        data = self.model_dump(mode="json")
        for section, key in (("api", "api_key"), ("finetune", "api_key"), ("serve", "api_key")):
            if data.get(section, {}).get(key):
                data[section][key] = "***"
        return data


def find_config(explicit: str | Path | None = None) -> Path:
    candidate = explicit or os.environ.get("WECLONE_CLOUD_CONFIG") or DEFAULT_CONFIG_NAME
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        raise ConfigError(f"找不到配置文件 {path}。请复制 cloud.example.jsonc 为 cloud.jsonc 后再运行。")
    return path


def load_config(explicit: str | Path | None = None) -> CloudConfig:
    path = find_config(explicit)
    try:
        raw = pyjson5.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"解析 {path} 失败：{exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} 顶层必须是对象")
    try:
        config = CloudConfig(**raw)
    except Exception as exc:  # pydantic ValidationError 及其子类
        raise ConfigError(f"{path} 字段校验失败：{exc}") from exc
    config.source_path = str(path)
    return config
