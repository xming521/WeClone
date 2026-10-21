#!/usr/bin/env python3
"""Generate ``app/settings.jsonc`` from the WeClone template, without regex surgery.

Why: Windows PowerShell 5.1 mangles UTF-8 when editing the file as text (it even
produced "Json5EOF: Unclosed string").  Parsing with pyjson5 and writing plain
JSON is deterministic and keeps the Chinese system prompt intact.

Modes
-----
default : template as-is, minus ``security_args`` (no shared password prompt)
cpu     : tuned for CPU LoRA on a small model (1.5B class):
          cutoff 1024, batch 1 x grad-accum 16, fp16 off, sdpa attention
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyjson5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["default", "cpu"], default="default")
    parser.add_argument("--model", default="", help="model folder name under app/models/")
    args = parser.parse_args()

    if not args.template.is_file():
        print(f"template not found: {args.template}")
        return 1

    data = pyjson5.loads(args.template.read_text(encoding="utf-8"))
    data.pop("security_args", None)

    if args.mode == "cpu":
        model = args.model or "DeepSeek-R1-Distill-Qwen-1.5B"
        common = data.setdefault("common_args", {})
        common["model_name_or_path"] = f"./models/{model}"
        common["template"] = "qwen"
        train = data.setdefault("train_sft_args", {})
        train.update(
            {
                "cutoff_len": 1024,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "gradient_checkpointing": True,
                "num_train_epochs": 2,
                "learning_rate": 1e-4,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "fp16": False,
                "bf16": False,
                "plot_loss": False,
                "flash_attn": "disabled",
                "preprocessing_num_workers": 1,
                "dataloader_num_workers": 0,
                "logging_steps": 10,
                "save_steps": 200,
                "lora_rank": 8,
                "lora_dropout": 0.1,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out} (mode={args.mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
