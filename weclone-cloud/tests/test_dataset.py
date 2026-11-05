from __future__ import annotations

import json

import pytest
from weclone_cloud.dataset import (
    DatasetError,
    export_dataset,
    is_encrypted,
    normalize_messages,
    read_records,
)
from weclone_cloud.errors import DatasetError as DatasetErrorAlias

from .conftest import record


def test_normalize_messages_drops_system_and_empty():
    messages = [
        {"role": "system", "content": "旧系统提示"},
        {"role": "user", "content": "  "},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "在的"},
        {"role": "tool", "content": "ignore"},
    ]
    cleaned = normalize_messages(messages, system="新系统提示")
    assert cleaned == [
        {"role": "system", "content": "新系统提示"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "在的"},
    ]


def test_export_writes_jsonl_and_manifest(tmp_path):
    records = [
        record(0, score=5),
        record(1, score=1),  # 低分被过滤
        record(2, score=4, messages=[{"role": "assistant", "content": "只有一句"}]),
        {**record(3, score=5), "images": ["a.jpg"]},
    ]
    manifest = export_dataset(records, tmp_path, system="你是 B", min_score=3, val_ratio=0.0)

    assert manifest["exported"] == 1
    assert manifest["dropped"]["low_score"] == 1
    assert manifest["dropped"]["too_short"] == 1
    assert manifest["dropped"]["image_sample"] == 1

    lines = (tmp_path / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["messages"][0] == {"role": "system", "content": "你是 B"}
    assert row["messages"][-1]["role"] == "assistant"
    assert manifest["files"]["train"]["sha256"]


def test_export_rejects_all_dropped(tmp_path):
    with pytest.raises(DatasetError):
        export_dataset([record(0, score=1)], tmp_path, min_score=3)


def test_validation_split_keeps_minimum(tmp_path):
    records = [record(i, score=5) for i in range(40)]
    manifest = export_dataset(records, tmp_path, val_ratio=0.05)
    assert manifest["val"] >= 10
    assert manifest["train"] + manifest["val"] == 40
    assert (tmp_path / "val.jsonl").is_file()


def test_read_records_rejects_encrypted(tmp_path):
    target = tmp_path / "sft-my.json"
    target.write_bytes(b"WCENC\x01\r\n" + b"\x00" * 32)
    assert is_encrypted(target)
    with pytest.raises(DatasetError) as excinfo:
        read_records(target)
    assert "加密" in str(excinfo.value)


def test_dataset_error_is_exported():
    assert DatasetError is DatasetErrorAlias
