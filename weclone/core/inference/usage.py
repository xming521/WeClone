"""Tokens the model calls of this process used, as the services report them, and their cost when the model has a
price. Every attempt counts, retries and failed validation included, since a service bills every answer it sends."""

import threading
from datetime import datetime, timezone
from typing import Any, NamedTuple

from .model_prices import Price

TOTALS = (
    "requests",
    "unmetered",
    "input_tokens",
    "cached_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cost",
)


class UsageLimitReached(RuntimeError):
    """The service refused a request because the account's usage limit is reached, as Codex does when the plan's
    quota is used up. Its message is the service's own words."""


class Usage(NamedTuple):
    input_tokens: int  # cache hits included
    cached_tokens: int
    output_tokens: int  # thinking included
    reasoning_tokens: int


def _number(raw: dict, *paths: str) -> int | None:
    for path in paths:
        value: Any = raw
        for key in path.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def read_usage(raw: Any) -> Usage | None:
    """The usage of a chat completion, of DeepSeek's and Kimi's variants of it, or of a Codex turn. A service that
    reports no cache hits is taken to have had none."""
    if not isinstance(raw, dict):
        return None
    input_tokens = _number(raw, "prompt_tokens", "input_tokens")
    output_tokens = _number(raw, "completion_tokens", "output_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    cached = _number(
        raw,
        "prompt_tokens_details.cached_tokens",
        "prompt_cache_hit_tokens",
        "cached_tokens",
        "cached_input_tokens",
        "input_tokens_details.cached_tokens",
    )
    reasoning = _number(
        raw,
        "completion_tokens_details.reasoning_tokens",
        "reasoning_output_tokens",
        "output_tokens_details.reasoning_tokens",
    )
    return Usage(input_tokens, cached or 0, output_tokens, reasoning or 0)


def empty() -> dict[str, Any]:
    return dict.fromkeys(TOTALS, 0) | {"cost": 0.0}


def subtract(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    """What was used between two readings of the totals."""
    return {
        key: None if after[key] is None or before[key] is None else after[key] - before[key] for key in TOTALS
    }


class Meter:
    """`price` prices each attempt at the rates of the moment it ends. Without one, `totals` gives no cost. `limit` is
    what the service said when a request last met the account's usage limit."""

    def __init__(self):
        self.lock = threading.Lock()
        self.price: Price | None = None
        self.limit: str | None = None
        self._totals = empty()

    def record(self, raw_usage: Any, at: datetime | None = None) -> None:
        usage = read_usage(raw_usage)
        with self.lock:
            self._totals["requests"] += 1
            if usage is None:
                self._totals["unmetered"] += 1
                return
            for key, value in usage._asdict().items():
                self._totals[key] += value
            if self.price is not None:
                off_peak = self.price.off_peak_at(at or datetime.now(timezone.utc))
                self._totals["cost"] += self.price.cost(
                    usage.input_tokens, usage.cached_tokens, usage.output_tokens, off_peak=off_peak
                )

    def totals(self) -> dict[str, Any]:
        with self.lock:
            totals = dict(self._totals)
        if self.price is None:
            totals["cost"] = None
        return totals


METER = Meter()
