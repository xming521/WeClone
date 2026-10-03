"""Build a profile from chat data, keeping each run's intermediate results separate."""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter

import click

from weclone.utils import secure_storage


@contextmanager
def _stage(label: str):
    started = perf_counter()
    click.echo(f"[WeClone] I | {datetime.now():%H:%M:%S} | {label}：开始")
    try:
        yield
    except BaseException as exc:
        click.echo(
            f"[WeClone] E | {datetime.now():%H:%M:%S} | {label}：失败（{type(exc).__name__}），"
            f"耗时 {perf_counter() - started:.1f} 秒"
        )
        raise
    else:
        click.echo(
            f"[WeClone] I | {datetime.now():%H:%M:%S} | {label}：完成，耗时 {perf_counter() - started:.1f} 秒"
        )


def check_distillation(checkpoint: Path) -> None:
    state = secure_storage.read_json(checkpoint)
    failed = [entry for entry in state["entries"].values() if entry.get("status") != "done"]
    if failed:
        raise RuntimeError(f"记忆抽取有 {len(failed)} 个样本未完成，停止流水线。详情：{checkpoint}")


def run(
    *,
    with_events: bool,
    output_dir: Path | None,
    config_path: Path,
    max_context_tokens: int | None = None,
) -> Path:
    from weclone.data.agent import distill_profile

    secure_storage.configure(config_path)
    args = distill_profile.default_args()
    args.config_path = config_path
    args = distill_profile.resolve_llm_args(args)
    args.max_context_tokens = max_context_tokens
    if output_dir is None:
        output_dir = Path("dataset/res_csv/agent/profile_runs") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"结果目录非空，请指定新的目录：{output_dir}")
    people_dir = output_dir / "people"
    distill_dir = output_dir / "distill"
    organization_dir = output_dir / "memory_organization"
    hierarchy_dir = output_dir / "profile_hierarchy"
    args.input_dir, args.output_dir = people_dir, distill_dir
    click.echo(f"本次结果目录：{output_dir}")
    click.echo(f"抽取内容：{'画像和事件' if with_events else '仅画像'}")
    task = "用户画像和事件记忆" if with_events else "用户画像"
    if not distill_profile.confirm_distillation(args, task=task, file_count=None):
        raise click.Abort()

    with _stage("启动：加载抽取与整理模块"):
        from weclone.data.agent import distill_event, llm_profile_hierarchy, organize_memories

    with _stage("启动：读取模型上下文"):
        max_context_tokens = organize_memories.resolve_context_window(args)
    click.echo(f"输入 Token 上限：{max_context_tokens}")

    with _stage("[1/4] 准备聊天数据"):
        from weclone.data.qa_generator import DataProcessor

        DataProcessor().main(
            dataset_output_dir=output_dir / "sft",
            stage2_output_dir=people_dir,
            strict_stage2=True,
            calculate_cutoff_len=False,
        )

    source_files = list(distill_profile.iter_chat_files(people_dir))
    if not source_files:
        raise FileNotFoundError(f"没有生成聊天样本：{people_dir}")
    click.echo(f"聊天样本准备完成：{len(source_files)} 个聊天文件。")

    with _stage("[2/4] 抽取画像记忆"):
        distill_profile.main(
            input_dir=people_dir, output_dir=distill_dir, config_path=config_path, confirmed=True
        )
        check_distillation(distill_profile.default_state_path(distill_dir))
    if with_events:
        with _stage("[2/4] 抽取事件记忆"):
            distill_event.main(
                input_dir=people_dir, output_dir=distill_dir, config_path=config_path, confirmed=True
            )
            check_distillation(distill_event.default_event_state_path(distill_dir))

    with _stage("[3/4] 分类、分组和归纳记忆"):
        organize_memories.main(
            [
                "--stage",
                "all",
                "--input-dir",
                str(distill_dir),
                "--output-dir",
                str(organization_dir),
                "--config-path",
                str(config_path),
                "--max-context-tokens",
                str(max_context_tokens),
            ]
        )

    with _stage("[4/4] 生成画像层级"):
        llm_profile_hierarchy.main(
            [
                "--input-path",
                str(organization_dir / "organized_memories.json"),
                "--output-dir",
                str(hierarchy_dir),
                "--config-path",
                str(config_path),
                "--max-context-tokens",
                str(max_context_tokens),
            ]
        )
        result = secure_storage.resolve_path(hierarchy_dir / "profile_hierarchy.json")
        if not result.is_file():
            raise FileNotFoundError(f"没有生成画像文件：{result}")
    return result
