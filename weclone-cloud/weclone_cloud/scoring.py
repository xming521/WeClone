"""Score chat samples through an OpenAI-compatible API.

The score is written back into each record (``score`` / ``score_reason`` /
``score_source``), so the result stays compatible with WeClone's own cleaning
step, which keeps rows with ``score >= accept_score``.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .api import ChatClient
from .prompts import SCORE_SCHEMA, build_score_messages


@dataclass
class ScoreOutcome:
    index: int
    record_id: str
    score: int | None = None
    reason: str = ""
    error: str = ""
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.score is not None


class _Pacer:
    """Global minimum spacing between requests (protects free-tier quotas)."""

    def __init__(self, interval: float) -> None:
        self.interval = max(0.0, interval)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait_for = self._next_at - now
            self._next_at = max(now, self._next_at) + self.interval
        if wait_for > 0:
            time.sleep(wait_for)


def _record_id(record: dict[str, Any], index: int) -> str:
    for key in ("id", "sample_id"):
        value = record.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return str(index)


def _coerce_score(payload: dict[str, Any] | None) -> tuple[int | None, str]:
    if not isinstance(payload, dict):
        return None, ""
    raw = payload.get("score")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None, str(payload.get("reason") or "")
    if not 1 <= value <= 5:
        return None, str(payload.get("reason") or "")
    return value, str(payload.get("reason") or "")


def score_records(
    records: Sequence[dict[str, Any]],
    client: ChatClient,
    *,
    max_chars: int = 6000,
    max_workers: int = 8,
    request_interval: float = 0.0,
    rescore: bool = False,
    limit: int | None = None,
    use_cache: bool = True,
    progress: Callable[[Iterable[Any], int], Iterable[Any]] | None = None,
) -> list[ScoreOutcome]:
    """Score every record; unchanged records are skipped unless ``rescore``."""
    targets: list[int] = []
    for index, record in enumerate(records):
        if limit is not None and len(targets) >= limit:
            break
        if not rescore and isinstance(record.get("score"), int) and record.get("score_source") == "cloud":
            continue
        targets.append(index)

    outcomes: dict[int, ScoreOutcome] = {
        index: ScoreOutcome(
            index=index,
            record_id=_record_id(records[index], index),
            score=int(records[index]["score"]),  # type: ignore[arg-type]
            reason=str(records[index].get("score_reason") or ""),
        )
        for index in range(len(records))
        if isinstance(records[index].get("score"), int) and records[index].get("score_source") == "cloud"
    }

    if not targets:
        return [outcomes[i] for i in range(len(records))]

    pacer = _Pacer(request_interval)
    writer_lock = threading.Lock()

    def work(index: int) -> ScoreOutcome:
        record = records[index]
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            return ScoreOutcome(index=index, record_id=_record_id(record, index), error="messages 为空")
        pacer.wait()
        try:
            completion = client.chat(
                build_score_messages(messages, max_chars=max_chars),
                json_schema=SCORE_SCHEMA,
                use_cache=use_cache,
            )
        except Exception as exc:  # 网络/鉴权/限流，逐条记录不中断整批
            return ScoreOutcome(index=index, record_id=_record_id(record, index), error=str(exc))
        score, reason = _coerce_score(completion.data)
        if score is None:
            return ScoreOutcome(
                index=index,
                record_id=_record_id(record, index),
                error=f"无法解析评分：{completion.text[:200]!r}",
                cached=completion.cached,
            )
        return ScoreOutcome(
            index=index,
            record_id=_record_id(record, index),
            score=score,
            reason=reason,
            cached=completion.cached,
        )

    iterator: Iterable[Any]
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = [pool.submit(work, index) for index in targets]
        iterator = as_completed(futures)
        if progress is not None:
            iterator = progress(iterator, len(futures))
        for future in iterator:
            outcome = future.result()
            with writer_lock:
                outcomes[outcome.index] = outcome

    return [outcomes[i] for i in sorted(outcomes)]


def apply_outcomes(records: Sequence[dict[str, Any]], outcomes: Sequence[ScoreOutcome]) -> tuple[int, int]:
    """Write scores back; returns ``(applied, failed)``.  Keeps any pre-existing score."""
    applied = failed = 0
    for outcome in outcomes:
        record = records[outcome.index]
        if outcome.ok:
            if "original_score" not in record:
                record["original_score"] = record.get("score")
            record["score"] = outcome.score
            record["score_reason"] = outcome.reason
            record["score_source"] = "cloud"
            applied += 1
        else:
            failed += 1
    return applied, failed


def filter_records(records: Sequence[dict[str, Any]], *, min_score: int) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if isinstance(record.get("score"), int) and int(record["score"]) >= min_score
    ]


def build_report(
    records: Sequence[dict[str, Any]],
    outcomes: Sequence[ScoreOutcome],
    *,
    min_score: int,
    model: str,
) -> dict[str, Any]:
    # Only count rows this tool scored. WeClone initialises ``score`` to 0 for
    # every row it produces, so including unscored rows would drag the mean to
    # ~0 and count them as failures.
    scored_records = [
        record
        for record in records
        if record.get("score_source") == "cloud" and isinstance(record.get("score"), int)
    ]
    scores = [int(record["score"]) for record in scored_records]
    distribution = Counter(scores)
    failed = [o for o in outcomes if not o.ok]
    cache_hits = sum(1 for o in outcomes if o.cached)
    return {
        "model": model,
        "records": len(records),
        "scored": len(scores),
        "unscored": len(records) - len(scores),
        "failed": len(failed),
        "min_score": min_score,
        "kept": sum(1 for score in scores if score >= min_score),
        "mean_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "distribution": {str(score): distribution.get(score, 0) for score in range(1, 6)},
        "cache_hits": cache_hits,
        "errors": [{"id": o.record_id, "error": o.error} for o in failed][:50],
    }


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
