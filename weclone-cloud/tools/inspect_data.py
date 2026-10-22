#!/usr/bin/env python3
"""识别聊天数据形态，并给出下一步该跑的命令。

用法：
    python tools/inspect_data.py "D:\\导出\\某个文件或目录"
    python tools/inspect_data.py "app\\dataset\telegram" --json

只读，不修改任何文件；不需要第三方依赖。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ENCRYPTED_MAGIC = b"WCENC"
SQLITE_MAGIC = b"SQLite format 3\x00"

WECHAT_COLUMNS = {"is_sender", "msg", "CreateTime", "type_name"}


def fail(message: str) -> None:
    print(f"[ERR] {message}")
    raise SystemExit(1)


def read_prefix(path: Path, size: int = 16) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def describe_messages(messages: Any) -> tuple[int, bool, bool]:
    """返回 (轮数, 是否有 user, 是否有 assistant)。"""
    if not isinstance(messages, list):
        return 0, False, False
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    return len(messages), "user" in roles, "assistant" in roles


def inspect_json_file(path: Path) -> dict[str, Any] | None:
    payload = load_json(path)
    if payload is None:
        return None

    # Telegram 导出：{"messages": [...], "name": ..., "type": ...}
    if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        first = payload["messages"][0] if payload["messages"] else {}
        keys = set(first) if isinstance(first, dict) else set()
        if {"from_id", "date_unixtime"} & keys or {"from", "date"} & keys:
            return {
                "kind": "telegram",
                "detail": f"Telegram result.json，{len(payload['messages'])} 条消息，"
                f"chat name={payload.get('name') or '?'}",
            }

    if isinstance(payload, list):
        records = [item for item in payload if isinstance(item, dict)]
        if not records:
            return {"kind": "empty", "detail": "JSON 数组为空"}
        first = records[0]
        has_messages = isinstance(first.get("messages"), list)
        if has_messages:
            turns = [describe_messages(r.get("messages"))[0] for r in records]
            scored = sum(
                1 for r in records if isinstance(r.get("score"), int) and r.get("score_source") == "cloud"
            )
            any_score = sum(1 for r in records if isinstance(r.get("score"), int))
            images = sum(1 for r in records if r.get("images"))
            with_system = sum(1 for r in records if r.get("system"))
            kind = "weclone-sft" if "system" in first else "openai-messages"
            return {
                "kind": kind,
                "detail": (
                    f"{len(records)} 条样本；字段={sorted(first)[:8]}；"
                    f"平均 {sum(turns) / max(1, len(turns)):.1f} 轮；"
                    f"带 system {with_system}；带图片 {images}；"
                    f"有分数 {any_score}（其中云端打分 {scored}）"
                ),
                "records": len(records),
                "scored": any_score,
            }
        return {"kind": "unknown-json", "detail": f"JSON 数组元素字段={sorted(first)[:10]}"}

    if isinstance(payload, dict):
        return {"kind": "unknown-json", "detail": f"JSON 对象字段={sorted(payload)[:10]}"}
    return {"kind": "unknown-json", "detail": f"JSON 标量：{type(payload).__name__}"}


def inspect_jsonl(path: Path) -> dict[str, Any]:
    total = 0
    first_record: dict[str, Any] | None = None
    bad = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            if first_record is None:
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict):
                        first_record = parsed
                except ValueError:
                    bad += 1
    if first_record is None:
        return {"kind": "unknown-jsonl", "detail": f"{total} 行，首行无法解析"}
    return {
        "kind": "openai-jsonl",
        "detail": f"{total} 行（坏行 {bad}）；字段={sorted(first_record)[:8]}",
        "records": total,
    }


def inspect_csv_folder(folder: Path) -> dict[str, Any] | None:
    csv_files = sorted(folder.rglob("*.csv"))
    if not csv_files:
        return None
    sample = csv_files[0]
    try:
        header = sample.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return None
    columns = {part.strip() for part in header.split(",")}
    rows = sum(1 for _ in sample.open(encoding="utf-8", errors="replace")) - 1
    parents = {path.parent.name for path in csv_files}
    return {
        "kind": "wechat-csv",
        "detail": (
            f"{len(csv_files)} 个 CSV，{len(parents)} 个联系人目录；"
            f"样例 {sample.name}，约 {rows} 行；列={'OK' if WECHAT_COLUMNS <= columns else '[ERR] ' + str(sorted(columns)[:8])}"
        ),
        "csv_ok": WECHAT_COLUMNS <= columns,
    }


def inspect(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"路径不存在：{path}")

    if path.is_dir():
        result = inspect_csv_folder(path)
        if result:
            result["path"] = str(path)
            return result
        for name in ("result.json",):
            candidate = path / name
            if candidate.is_file():
                found = inspect_json_file(candidate)
                if found:
                    found["path"] = str(candidate)
                    return found
        # 目录里可能有 WeClone 产物
        for pattern in ("sft-*.json", "*.scored.json", "train.jsonl"):
            for candidate in path.rglob(pattern):
                if candidate.is_file():
                    found = (
                        inspect_json_file(candidate)
                        if candidate.suffix == ".json"
                        else inspect_jsonl(candidate)
                    )
                    if found:
                        found["path"] = str(candidate)
                        return found
        fail(f"{path} 里没找到可识别的聊天数据（CSV 目录 / result.json / sft-*.json / *.jsonl）")

    if path.suffix.lower() == ".db" or read_prefix(path, 16) == SQLITE_MAGIC:
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
                tables = [
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                    )
                ]
        except sqlite3.Error as exc:
            return {"path": str(path), "kind": "sqlite-unreadable", "detail": str(exc)}
        return {
            "path": str(path),
            "kind": "sqlite-db",
            "detail": f"SQLite 数据库，表：{tables[:10]}（{len(tables)} 张）",
        }

    if path.suffix.lower() in {".json", ".jsonc"}:
        found = inspect_json_file(path)
        if found:
            found["path"] = str(path)
            return found
        fail(f"{path} 不是合法 JSON")

    if path.suffix.lower() == ".jsonl":
        found = inspect_jsonl(path)
        found["path"] = str(path)
        return found

    prefix = read_prefix(path, 8)
    if prefix.startswith(ENCRYPTED_MAGIC):
        return {
            "path": str(path),
            "kind": "encrypted",
            "detail": "WeClone 加密文件（WCENC）",
        }
    fail(f"不认识的文件类型：{path.suffix or '无扩展名'}")


def next_steps(result: dict[str, Any]) -> list[str]:
    kind = result["kind"]
    path = result.get("path", "<你的文件>")
    if kind == "encrypted":
        return [
            '在 app\\settings.jsonc 把 security_args.storage_mode 改为 "plaintext" 后重跑 weclone-cli make-dataset',
            f"或导出明文：weclone-cli decrypt --input {path} --output plain.json",
        ]
    if kind == "weclone-sft":
        if result.get("scored"):
            return [
                f"weclone-cloud export --input {path} --min-score 3 --out-dir finetune_data",
                f"weclone-cloud persona --input {path} --examples 6",
            ]
        return [
            f"weclone-cloud score --input {path} --limit 20   # 先试 20 条",
            f"weclone-cloud score --input {path} --min-score 3   # 全量",
        ]
    if kind == "openai-messages":
        return [
            f"weclone-cloud score --input {path}   # 补分数（样本本身没有 score）",
            f"weclone-cloud export --input {path} --min-score 0 --out-dir finetune_data   # 不打分直接导出",
        ]
    if kind == "openai-jsonl":
        return [
            "已符合微调格式，可直接用（无需 score/export）",
            f"weclone-cloud finetune --train {path}   # provider=openai 时",
        ]
    if kind == "telegram":
        return [
            "把整个 ChatExport_* 文件夹放到 app\\dataset\\telegram\\",
            'app\\settings.jsonc 里设 "platform": "telegram"，并把 telegram_args.my_id 改成你消息的 from_id',
            "cd app && weclone-cli make-dataset",
        ]
    if kind == "wechat-csv":
        if not result.get("csv_ok"):
            return ["CSV 缺少 is_sender/msg/CreateTime/type_name 列，确认是完整导出（不是精简版）"]
        return [
            "把联系人目录放到 app\\dataset\\csv\\<联系人>\\*.csv（保持一层子目录）",
            '确认 app\\settings.jsonc 的 make_dataset_args.platform = "chat"',
            "cd app && weclone-cli make-dataset",
        ]
    if kind.startswith("sqlite"):
        return [
            "WeClone 需要 CSV。先导出：chatlog（Go 版）或 WeChatMsg 的 CSV 导出功能",
            "导出后按微信 CSV 分支走：app\\dataset\\csv\\<联系人>\\*.csv → weclone-cli make-dataset",
        ]
    return ["把上面的字段信息发给我，我告诉你下一步"]


KIND_NAMES = {
    "weclone-sft": "WeClone SFT 数据集",
    "openai-messages": "OpenAI 风格问答数组（无分数）",
    "openai-jsonl": "OpenAI 微调 JSONL",
    "telegram": "Telegram 导出",
    "wechat-csv": "微信导出 CSV",
    "sqlite-db": "SQLite 数据库",
    "sqlite-unreadable": "SQLite（读取失败）",
    "encrypted": "WeClone 加密文件",
    "empty": "空数据",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args()

    result = inspect(args.path)
    steps = next_steps(result)
    if args.json:
        print(json.dumps({"result": result, "next": steps}, ensure_ascii=False, indent=2))
        return

    print(f"路径  {result['path']}")
    print(f"形态  {KIND_NAMES.get(result['kind'], result['kind'])}")
    print(f"详情  {result['detail']}")
    print()
    print("下一步：")
    for index, step in enumerate(steps, 1):
        print(f"  {index}. {step}")


if __name__ == "__main__":
    sys.exit(main())
