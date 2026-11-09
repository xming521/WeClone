"""Model prices from the bundled snapshot of LiteLLM's community table, which scripts/update_model_prices.py refreshes.

A model is found by its id after the last "/", lowercased. A service whose host is in OFFICIAL is priced by its own
entry. Any other address, such as a relay or a self-hosted server, gets the median of what the providers listing the
same model charge. A model no provider lists has no price."""

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PRICES_PATH = Path(__file__).with_name("model_prices.json")
# LiteLLM's provider, and the service's key in VENDORS of web/src/settings/modelSettings.ts, for each host of the
# services the model settings offer by name. SiliconFlow and OpenRouter have no entries of their own in the table.
OFFICIAL = {
    "api.deepseek.com": ("deepseek", "deepseek"),
    "dashscope.aliyuncs.com": ("dashscope", "qwen"),
    "dashscope-intl.aliyuncs.com": ("dashscope", "qwen"),
    "api.moonshot.cn": ("moonshot", "kimi"),
    "api.moonshot.ai": ("moonshot", "kimi"),
    "open.bigmodel.cn": ("bigmodel", "glm"),
    "api.z.ai": ("zai", "glm"),
    "api.openai.com": ("openai", "openai"),
}


@cache
def table() -> dict[str, Any]:
    return json.loads(PRICES_PATH.read_text(encoding="utf-8"))


def _minutes(clock: str) -> int:
    hours, minutes = clock.split(":")
    return int(hours) * 60 + int(minutes)


def in_windows(windows: list[dict], at: datetime) -> bool:
    """Whether `at` falls in LiteLLM time windows: UTC spans such as "10:00-00:00" on ISO weekdays, where an end not
    after the start runs past midnight into the next day."""
    at = at.astimezone(timezone.utc)
    minute = at.hour * 60 + at.minute
    today, yesterday = at.isoweekday(), (at - timedelta(days=1)).isoweekday()
    for window in windows:
        days = window.get("weekdays", range(1, 8))
        spans = window.get("hours_utc") or "00:00-00:00"
        for span in [spans] if isinstance(spans, str) else spans:
            start, end = (_minutes(part) for part in span.split("-"))
            if end <= start:
                end += 24 * 60
            # Past midnight, a span belongs to the day it started on.
            if (today in days and start <= minute < end) or (
                yesterday in days and start <= minute + 24 * 60 < end
            ):
                return True
    return False


@dataclass(frozen=True)
class Price:
    """US dollars per token. `source` is "official" for the service's own entry, named by `vendor`, and "market" for
    the median of all providers; `off_peak` holds the cheaper rates and their time windows, for the services that have
    them."""

    input: float
    output: float
    cache_read: float
    source: str
    vendor: str | None
    date: str
    off_peak: dict[str, Any] | None = None

    def off_peak_at(self, at: datetime) -> bool:
        return self.off_peak is not None and in_windows(self.off_peak["windows"], at)

    def cost(
        self, input_tokens: int, cached_tokens: int, output_tokens: int, *, off_peak: bool = False
    ) -> float:
        """Cache hits at the cache price, the rest of the input at the input price, the output with its thinking at
        the output price."""
        rates = {"input": self.input, "output": self.output, "cache_read": self.cache_read}
        if off_peak and self.off_peak is not None:
            rates.update((key, value) for key, value in self.off_peak.items() if key in rates)
        cached = min(cached_tokens, input_tokens)
        return (
            cached * rates["cache_read"]
            + (input_tokens - cached) * rates["input"]
            + output_tokens * rates["output"]
        )


def lookup(base_url: str | None, model: str | None) -> Price | None:
    if not model or not model.strip():
        return None
    snapshot = table()
    entries = snapshot["models"].get(model.strip().lower().rsplit("/", 1)[-1])
    if not entries:
        return None
    provider, vendor = OFFICIAL.get((urlparse((base_url or "").strip()).hostname or "").lower(), (None, None))
    own = next((entry for entry in entries if provider and entry["provider"] == provider), None)
    if own is not None:
        return Price(
            own["input"],
            own["output"],
            own.get("cache_read", own["input"]),
            "official",
            vendor,
            snapshot["date"],
            own.get("off_peak"),
        )

    def median(key: str) -> float:
        return statistics.median(entry.get(key, entry["input"]) for entry in entries)

    return Price(median("input"), median("output"), median("cache_read"), "market", None, snapshot["date"])
