#!/usr/bin/env python3
"""把 Alpha 导出的微信记录转成 WeClone 能吃的 CSV。

Alpha 导出结构：
    Alpha导出-20261020-123117/
      1-刘肖昕 大象电竞网管/
         刘肖昕 大象电竞网管.csv
         media-manifest.json
         media/image/*.gif ...
      2-木目心/...

Alpha 的 CSV 列与 WeClone 期望的一致：
    id,MsgSvrID,type_name,is_sender,talker,msg,src,CreateTime

但有三个必须处理的差异，否则会污染训练数据：
1. `type_name=sticker` 的 msg 是一整段 `<msg><emoji .../></msg>` XML，
   WeClone 只按中文类型名过滤，会把 XML 当成正文塞进样本；
2. `image` 的 msg 是 `[图片]`，且 `src` 指向未导入的媒体，WeClone 会保留成正文；
3. 文件带 BOM 时 WeClone 的 `encoding="utf-8"` 会把首列读成 `\\ufeffid`。

因此默认只保留 `type_name=text` 的行，并写成 UTF-8 无 BOM。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

COLUMNS = ["id", "MsgSvrID", "type_name", "is_sender", "talker", "msg", "src", "CreateTime"]
KEEP_TYPE = "text"
_SAFE = re.compile(r'[<>:"/\\|?*]')


def safe_name(name: str) -> str:
    cleaned = _SAFE.sub("_", name).strip().strip(".")
    return cleaned or "contact"


def find_contacts(root: Path) -> list[tuple[str, Path]]:
    """返回 [(联系人名, csv 路径)]。"""
    if root.is_file() and root.suffix.lower() == ".csv":
        return [(root.parent.name, root)]
    if not root.is_dir():
        return []
    contacts: list[tuple[str, Path]] = []
    direct = sorted(root.glob("*.csv"))
    if direct:  # 传进来的就是一个联系人目录
        return [(root.name, direct[0])]
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        csv_files = sorted(child.glob("*.csv"))
        if not csv_files:
            continue
        # 目录名形如 "1-何明"，取 "何明"；没有前缀就用目录名
        name = re.sub(r"^\d+[-_ ]*", "", child.name) or child.name
        contacts.append((safe_name(name), csv_files[0]))
    return contacts


def convert(source: Path, target: Path, *, dry_run: bool = False) -> dict:
    stats: dict = {"kept": 0, "dropped": Counter(), "first": "", "last": "", "chars": 0}
    rows = []
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{source} 没有表头")
        missing = [
            column
            for column in ("type_name", "is_sender", "msg", "CreateTime")
            if column not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"{source} 缺少列：{missing}")
        for row in reader:
            kind = (row.get("type_name") or "").strip().lower()
            content = (row.get("msg") or "").strip()
            if kind != KEEP_TYPE:
                stats["dropped"][kind or "empty"] += 1
                continue
            if not content:
                stats["dropped"]["empty_text"] += 1
                continue
            rows.append(row)
            stats["kept"] += 1
            stats["chars"] += len(content)
            stamp = (row.get("CreateTime") or "").strip()
            if stamp:
                stats["first"] = min(stats["first"], stamp) if stats["first"] else stamp
                stats["last"] = max(stats["last"], stamp) if stats["last"] else stamp

    if dry_run:
        return stats

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, type=Path, help="Alpha 导出根目录或单个联系人目录")
    parser.add_argument("--out", type=Path, default=Path("app/dataset/csv"), help="WeClone 的 csv 目录")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    contacts = find_contacts(args.input)
    if not contacts:
        print(f"[ERR] 在 {args.input} 下没找到联系人 CSV")
        raise SystemExit(1)

    print(f"发现 {len(contacts)} 个联系人，输出到 {args.out}\n")
    total = 0
    for name, csv_path in contacts:
        stats = convert(csv_path, args.out / name / f"{name}.csv", dry_run=args.dry_run)
        total += stats["kept"]
        dropped = ", ".join(f"{k}={v}" for k, v in stats["dropped"].most_common())
        print(
            f"  {name:<28} 保留 {stats['kept']:>6} 行  {stats['chars']:>8} 字  {stats['first'][:10]} → {stats['last'][:10]}"
        )
        if dropped:
            print(f"  {'':<28} 丢弃 {dropped}")

    print(f"\n合计保留 {total} 行文本消息。")
    if args.dry_run:
        print("（dry-run：未写入文件）")
    else:
        print("\n下一步：\n  cd app && weclone-cli make-dataset")


if __name__ == "__main__":
    sys.exit(main())
