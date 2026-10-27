"""Build a profile from chat data, keeping each run's intermediate results separate."""

from contextlib import contextmanager
from datetime import datetime
from importlib.util import find_spec
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


def run(
    *,
    with_style: bool,
    output_dir: Path | None,
    config_path: Path,
    max_context_tokens: int | None = None,
) -> Path:
    from weclone.data.agent import distill_profile
    from weclone.data.agent.distill_style import run as distill_style

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
    style_dir = output_dir / "style"
    style_argv = ["--input-dir", str(people_dir), "--output", str(style_dir), "--settings", str(config_path)]
    if with_style:
        # Style needs the local LTP parser, which the profile itself does not.
        ltp_model = distill_style.parse_args(style_argv).ltp_model
        if find_spec("ltp") is None or not ltp_model.is_dir():
            raise RuntimeError(f"做分身要抽取说话风格，请先安装 ltp 包，并把 LTP 模型放在 {ltp_model}")
    stages = 6 if with_style else 5
    click.echo(f"本次结果目录：{output_dir}")
    click.echo(f"抽取内容：{'画像、事件和说话风格' if with_style else '画像和事件'}")
    task = "用户画像、事件记忆和说话风格" if with_style else "用户画像和事件记忆"
    if not distill_profile.confirm_distillation(args, task=task, file_count=None):
        raise click.Abort()

    with _stage("启动：加载抽取与整理模块"):
        from weclone.data.agent import distill_event, llm_profile_hierarchy, organize_memories

    with _stage("启动：读取模型上下文"):
        max_context_tokens = organize_memories.resolve_context_window(args)
    click.echo(f"输入 Token 上限：{max_context_tokens}")

    with _stage(f"[1/{stages}] 准备聊天数据"):
        from weclone.data.agent import chat_snapshot

        chat_snapshot.build_snapshot(people_dir, chat_snapshot.load_chat_args(config_path))

    source_files = list(distill_profile.iter_chat_files(people_dir))
    if not source_files:
        raise FileNotFoundError(f"没有生成聊天样本：{people_dir}")
    click.echo(f"聊天样本准备完成：{len(source_files)} 个聊天文件。")

    with _stage(f"[2/{stages}] 抽取画像记忆"):
        # Raises when samples failed. Rerunning build-profile into a new directory asks only for those.
        distill_profile.main(
            input_dir=people_dir, output_dir=distill_dir, config_path=config_path, confirmed=True
        )
    with _stage(f"[2/{stages}] 抽取事件记忆"):
        distill_event.main(
            input_dir=people_dir, output_dir=distill_dir, config_path=config_path, confirmed=True
        )

    with _stage(f"[3/{stages}] 归纳领域、分类、分主题和汇总记忆"):
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

    with _stage(f"[4/{stages}] 生成画像层级"):
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

    # The profile is done by now, so domains left without a summary only get a note.
    with _stage(f"[5/{stages}] 生成领域总结"):
        from weclone.server.profile_summary import build_summaries

        missing = build_summaries(hierarchy_dir)
    if missing:
        click.echo(f"领域总结：{missing}")

    if with_style:
        with _stage("[6/6] 抽取说话风格"):
            distill_style.main(style_argv)
    return result
