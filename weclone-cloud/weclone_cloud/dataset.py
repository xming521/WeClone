"""Turn WeClone SFT records into OpenAI fine-tune JSONL.

WeClone stores one JSON array of ``{id, time, score, messages, system, images}``
objects (see ``weclone/data/models.py:QaPair``).  Cloud fine-tuning wants
``{"messages": [{"role": ..., "content": ...}, ...]}`` per line, one sample per
line, ending with an ``assistant`` message.

Encrypted stores (WeClone ``storage_mode=encrypted``) start with ``WCENC`` and
cannot be read here; the error message explains how to decrypt.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .errors import DatasetError

ENCRYPTED_MAGIC = b"WCENC"
DEFAULT_SYSTEM = "请你扮演一名人类，不要说自己是人工智能"
VALID_ROLES = {"system", "user", "assistant"}


# --------------------------------------------------------------------------- io
def is_encrypted(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(8).startswith(ENCRYPTED_MAGIC)
    except OSError as exc:
        raise DatasetError(f"无法读取 {path}: {exc}") from exc


def read_records(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        raise DatasetError(f"数据集不存在：{target}")
    if is_encrypted(target):
        raise DatasetError(
            f"{target} 是 WeClone 加密文件（WCENC），本工具需要明文。\n"
            "  二选一：\n"
            '  1) 在 settings.jsonc 里把 security_args.storage_mode 改为 "plaintext"，'
            "重新跑一次 weclone-cli make-dataset；\n"
            "  2) 用 weclone-cli decrypt --input <加密文件> --output <明文文件> 导出后再传 --input。"
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetError(f"{target} 不是合法 JSON：{exc}") from exc
    if not isinstance(payload, list):
        raise DatasetError(f"{target} 顶层应为数组，实际是 {type(payload).__name__}")
    records = [item for item in payload if isinstance(item, dict)]
    if len(records) != len(payload):
        raise DatasetError(f"{target} 中有 {len(payload) - len(records)} 条记录不是对象")
    return records


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return target


# ------------------------------------------------------------------- conversion
def normalize_messages(
    messages: Sequence[dict[str, Any]],
    *,
    system: str | None = None,
) -> list[dict[str, str]]:
    """Keep valid roles, drop empties, enforce ``system`` first when supplied."""
    cleaned: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        content = message.get("content")
        if role not in VALID_ROLES or not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        if role == "system":
            continue  # 统一由参数注入，避免每条样本里出现不同 system
        cleaned.append({"role": role, "content": content})
    if system:
        cleaned.insert(0, {"role": "system", "content": system})
    return cleaned


def record_to_row(
    record: dict[str, Any],
    *,
    system: str | None,
    min_score: int,
) -> tuple[dict[str, Any] | None, str]:
    """Return ``(row, "")`` or ``(None, drop_reason)``."""
    score = record.get("score")
    if isinstance(score, int) and score < min_score:
        return None, "low_score"
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return None, "no_messages"
    if record.get("images"):
        return None, "image_sample"
    cleaned = normalize_messages(messages, system=system)
    dialogue = [m for m in cleaned if m["role"] != "system"]
    if len(dialogue) < 2:
        return None, "too_short"
    if dialogue[-1]["role"] != "assistant":
        return None, "no_assistant_tail"
    if not any(m["role"] == "user" for m in dialogue):
        return None, "no_user_turn"
    return {"messages": cleaned}, ""


# ---------------------------------------------------------------------- tokens
_ENCODER: Any = None
_ENCODER_READY = False


def _encoder() -> Any:
    global _ENCODER, _ENCODER_READY
    if not _ENCODER_READY:
        _ENCODER_READY = True
        try:  # 可选依赖；无网络时 get_encoding 会抛错
            import tiktoken

            _ENCODER = tiktoken.get_encoding("o200k_base")
        except Exception:
            _ENCODER = None
    return _ENCODER


def count_tokens(text: str) -> int:
    encoder = _encoder()
    if encoder is not None:
        return len(encoder.encode(text, disallowed_special=()))
    # 中文约 1 token / 1.5 字，英文约 1 token / 4 字符；取保守值
    return max(1, int(len(text) / 1.6))


def row_tokens(row: dict[str, Any]) -> int:
    return sum(count_tokens(message["content"]) + 4 for message in row["messages"])


# ---------------------------------------------------------------------- export
def export_dataset(
    records: Sequence[dict[str, Any]],
    out_dir: str | Path,
    *,
    system: str | None = DEFAULT_SYSTEM,
    min_score: int = 2,
    val_ratio: float = 0.05,
    seed: int = 42,
    max_tokens_per_row: int | None = None,
    base_model: str = "",
) -> dict[str, Any]:
    """Write ``train.jsonl``/``val.jsonl`` plus ``manifest.json``."""
    rows: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    seen: set[str] = set()

    for record in records:
        row, reason = record_to_row(record, system=system, min_score=min_score)
        if row is None:
            dropped[reason] += 1
            continue
        digest = hashlib.sha256(
            json.dumps(row["messages"], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if digest in seen:
            dropped["duplicate"] += 1
            continue
        if max_tokens_per_row is not None and row_tokens(row) > max_tokens_per_row:
            dropped["too_long"] += 1
            continue
        seen.add(digest)
        rows.append(row)

    if not rows:
        raise DatasetError(
            f"没有可导出的样本（来源 {len(records)} 条，丢弃原因：{dict(dropped)}）。"
            "请检查 --min-score 是否过高，或先运行 score。"
        )

    random.Random(seed).shuffle(rows)
    val_count = 0
    if val_ratio > 0 and len(rows) >= 20:
        # 微调平台通常要求验证集 >= 10 条，不够就干脆不切分
        val_count = min(max(10, int(len(rows) * min(val_ratio, 0.5))), len(rows) // 2)
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    train_path = target / "train.jsonl"
    val_path = target / "val.jsonl"
    _write_jsonl(train_path, train_rows)
    if val_rows:
        _write_jsonl(val_path, val_rows)

    tokens = [row_tokens(row) for row in rows]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model": base_model,
        "system": system,
        "min_score": min_score,
        "source_records": len(records),
        "exported": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "dropped": dict(dropped),
        "tokens": {
            "total": sum(tokens),
            "mean": round(sum(tokens) / len(tokens), 1),
            "p95": sorted(tokens)[int(len(tokens) * 0.95) - 1] if tokens else 0,
            "max": max(tokens) if tokens else 0,
        },
        "files": {
            "train": {"path": str(train_path), "sha256": _sha256(train_path)},
            **({"val": {"path": str(val_path), "sha256": _sha256(val_path)}} if val_rows else {}),
        },
    }
    write_json(target / "manifest.json", manifest)
    return manifest


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimate_cost(manifest: dict[str, Any], *, price_per_million: float, epochs: int) -> float:
    """Rough training-cost estimate: tokens * epochs * price / 1e6."""
    return round(manifest["tokens"]["total"] * max(1, epochs) * price_per_million / 1_000_000, 4)
