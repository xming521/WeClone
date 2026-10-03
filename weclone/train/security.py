"""Adapters for third-party training tools that require filesystem paths."""

import json
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable

from weclone.utils.secure_storage import (
    is_encrypted_file,
    is_encrypted_mode,
    persist_tree,
    resolve_path,
    sensitive_runtime,
)


def parsed_cli_arguments(parser: Callable[..., tuple[Any, ...]]) -> dict[str, Any]:
    """Keep LlamaFactory's argument parsing while allowing secure path replacement."""
    from llamafactory.hparams import read_args

    original = read_args()
    if isinstance(original, dict):
        config = dict(original)
    else:
        config = {
            field.name: getattr(group, field.name)
            for group in parser(original)
            for field in fields(group)
            if field.init and not field.name.startswith("_")
        }
    adapters = config.get("adapter_name_or_path")
    if isinstance(adapters, list):
        config["adapter_name_or_path"] = ",".join(adapters)
    return config


def _encrypted_checkpoint(path: Path) -> bool:
    if not path.is_dir() or path.is_symlink() or not re.fullmatch(r"checkpoint-\d+", path.name):
        return False
    state = resolve_path(path / "trainer_state.json")
    return (
        state.is_file()
        and is_encrypted_file(state)
        and all(
            not item.is_symlink()
            and (not item.is_file() or item.name.endswith(".lock") or is_encrypted_file(item))
            for item in path.rglob("*")
        )
    )


def _complete_checkpoints(directory: Path) -> set[str]:
    completed = set()
    for path in directory.glob("checkpoint-*"):
        if not path.is_dir() or not re.fullmatch(r"checkpoint-\d+", path.name):
            continue
        try:
            state = json.loads((path / "trainer_state.json").read_bytes())
        except (OSError, ValueError):
            continue
        if isinstance(state, dict) and state.get("global_step") == int(path.name.removeprefix("checkpoint-")):
            completed.add(path.name)
    return completed


@contextmanager
def model_runtime(config: dict[str, Any], *, evaluation: bool = False):
    output_keys = ("save_dir", "logging_dir") if evaluation else ("export_dir",)
    with sensitive_runtime(
        config,
        dataset_keys=("dataset_dir", "media_dir", "task_dir"),
        model_keys=("adapter_name_or_path", "model_name_or_path"),
        output_keys=output_keys,
    ) as runtime:
        yield runtime


def run_training(config: dict[str, Any], trainer: Callable[..., Any]) -> Any:
    """Persist completed checkpoints immediately, including interrupted jobs."""
    if not is_encrypted_mode():
        return trainer(config)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 or os.environ.get("RANK") not in {None, "-1"}:
        raise ValueError(
            "安全模式暂不支持 torchrun/DeepSpeed 多进程启动：需要共享安全运行目录与运行密钥。"
            "请使用单进程训练命令；不会自动减少 GPU、batch size 或并发数。"
        )
    if config.get("use_ray"):
        raise ValueError("安全模式暂不支持 Ray 分布式训练：远端运行目录和密钥尚未协调。")

    from transformers import TrainerCallback

    original_output = config.get("output_dir")
    if not original_output:
        raise ValueError("Encrypted training requires an explicit output_dir")
    report_to = config.get("report_to")
    enabled_reporters = [report_to] if isinstance(report_to, str) else (report_to or [])
    if any(reporter not in {"none", "tensorboard"} for reporter in enabled_reporters):
        raise ValueError("Encrypted training supports only local tensorboard or report_to='none'")
    secured = dict(config)
    if report_to is None:
        secured["report_to"] = "none"
    if secured.get("use_swanlab") or secured.get("push_to_hub"):
        raise ValueError("Encrypted training cannot upload private checkpoints or training data")

    with sensitive_runtime(
        secured,
        dataset_keys=("dataset_dir", "media_dir"),
        model_keys=("adapter_name_or_path", "model_name_or_path", "resume_from_checkpoint"),
        output_keys=("output_dir", "logging_dir", "tokenized_path"),
    ) as runtime:
        # Keep a custom/default logging directory within the temporary output.
        if not runtime.get("logging_dir"):
            runtime["logging_dir"] = str(Path(runtime["output_dir"]) / "runs")
        durable_output = Path(original_output)
        runtime_output = Path(runtime["output_dir"])
        known_checkpoints = {
            path.name for path in durable_output.glob("checkpoint-*") if _encrypted_checkpoint(path)
        }

        def synchronize_checkpoints():
            complete = _complete_checkpoints(runtime_output)
            # The storage layer skips half-written checkpoint directories. Copy
            # completed checkpoints before honoring the Trainer's own retention.
            persist_tree(runtime_output, durable_output)
            for name in known_checkpoints - complete:
                target = durable_output / name
                if not (runtime_output / name).exists() and _encrypted_checkpoint(target):
                    shutil.rmtree(target)
            known_checkpoints.update(complete)

        class EncryptedCheckpointCallback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                if getattr(args, "should_save", True):
                    synchronize_checkpoints()
                return control

        return trainer(runtime, callbacks=[EncryptedCheckpointCallback()])
