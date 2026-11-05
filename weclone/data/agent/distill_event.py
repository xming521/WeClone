"""Event memories from chat samples, extracted and cached the same way as profile memories (see distill_profile)."""

from pathlib import Path
from typing import Any

from weclone.data.agent.distill_profile import FileStats, process_task_file, run_task

TASK = "event"


def process_file(source_path: Path, **options: Any) -> FileStats:
    return process_task_file(source_path, task=TASK, **options)


def main(
    *,
    input_dir: Path,
    output_dir: Path,
    config_path: Path | None = None,
    confirmed: bool = False,
    dry_run: bool = False,
) -> FileStats:
    return run_task(
        TASK,
        input_dir=input_dir,
        output_dir=output_dir,
        config_path=config_path,
        confirmed=confirmed,
        dry_run=dry_run,
    )
