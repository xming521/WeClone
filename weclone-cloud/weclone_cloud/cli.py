"""``weclone-cloud`` command line."""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from typing import Any, Callable

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

from . import __version__
from .api import ChatClient, CompletionCache
from .config import CloudConfig, load_config
from .dataset import (
    DEFAULT_SYSTEM,
    DatasetError,
    count_tokens,
    estimate_cost,
    export_dataset,
    read_records,
    write_json,
)
from .errors import CloudError, ConfigError
from .finetune import FineTuneClient, JobState, load_state, save_state
from .scoring import (
    apply_outcomes,
    build_report,
    filter_records,
    score_records,
    write_report,
)

# Chinese Windows consoles default to GBK; rich writes box-drawing and symbols
# that GBK cannot encode. Force UTF-8 (with replacement) so a plain console
# never dies with UnicodeEncodeError while printing an error message.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

console = Console()
DEFAULT_SCORED_SUFFIX = ".scored.json"


def handle_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (CloudError, DatasetError, ConfigError) as exc:
            console.print(f"[bold red]ERR {exc}[/bold red]")
            raise SystemExit(1) from None
        except KeyboardInterrupt:
            console.print("\n[yellow]已中断（已完成的评分已落盘）[/yellow]")
            raise SystemExit(130) from None

    return wrapper


def _config(ctx: click.Context) -> CloudConfig:
    return load_config(ctx.obj["config_path"])


def _usage_line(config: CloudConfig, usage: dict[str, int]) -> str:
    line = (
        f"token：入 {usage.get('prompt_tokens', 0):,} · "
        f"出 {usage.get('completion_tokens', 0):,} · 请求 {usage.get('requests', 0)}"
    )
    cost = config.api.estimate_cost(usage)
    if cost is not None:
        line += f" · 估算 {cost:.4f} {config.api.price_currency}"
    return line


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    "config_path",
    default=None,
    envvar="WECLONE_CLOUD_CONFIG",
    help="配置文件路径（默认 ./cloud.jsonc）",
)
@click.version_option(__version__, prog_name="weclone-cloud")
@click.pass_context
def main(ctx: click.Context, config_path: str | None) -> None:
    """用 API 处理聊天记录、云端微调，并把分身暴露成 OpenAI 兼容接口。"""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


# --------------------------------------------------------------------------- doctor
@main.command()
@click.pass_context
@handle_errors
def doctor(ctx: click.Context) -> None:
    """检查配置、密钥、接口连通性和数据集路径。"""
    config = _config(ctx)
    table = Table(title="weclone-cloud doctor", show_lines=False)
    table.add_column("项目", style="cyan", no_wrap=True)
    table.add_column("状态")
    table.add_column("说明", overflow="fold")

    table.add_row("配置文件", "OK", config.source_path or "")

    try:
        key_state = f"OK {config.api.key_source}"
        config.api.resolved_key
    except ConfigError as exc:
        key_state, note = "ERR", str(exc)
        table.add_row("API Key", key_state, note)
    else:
        table.add_row("API Key", key_state, config.api.base_url)

    table.add_row("打分模型", "·", config.api.model)
    table.add_row(
        "思考模式",
        "·",
        {"enabled": "开启（更贵更慢）", "disabled": "关闭（打分推荐）", "omit": "不发送该参数"}[
            config.api.resolved_thinking
        ],
    )
    table.add_row(
        "计价",
        "·" if config.api.price_input_per_million is None else "OK",
        f"{config.api.price_currency} {config.api.price_input_per_million}/{config.api.price_output_per_million} 每百万"
        if config.api.price_input_per_million is not None
        else "未配置（不显示成本估算）",
    )
    table.add_row("微调服务", "·", f"{config.finetune.provider} @ {config.finetune.base_url}")
    table.add_row("微调基座", "·", config.finetune.base_model)

    try:
        config.finetune.resolved_key
        table.add_row("微调 Key", "OK", f"env:{config.finetune.api_key_env}")
    except ConfigError as exc:
        table.add_row("微调 Key", "ERR", str(exc))

    try:
        serve_key = config.serve.resolved_key
        table.add_row(
            "服务鉴权",
            "OK" if serve_key else "·",
            f"Bearer {config.serve.api_key_env}" if serve_key else "未启用（本地自用）",
        )
    except ConfigError as exc:
        table.add_row("服务鉴权", "ERR", str(exc))

    state = load_state(config.finetune.state_file)
    table.add_row(
        "微调任务",
        "·" if not state.job_id else "OK",
        f"{state.job_id} → {state.status}" + (f" / {state.model}" if state.model else ""),
    )

    try:
        import tiktoken  # noqa: F401

        table.add_row("token 统计", "OK", "tiktoken 可用")
    except Exception:
        table.add_row("token 统计", "·", "未安装 tiktoken，使用字符估算（pip install '.[tokens]'）")

    console.print(table)

    if "deepseek" in config.api.base_url.lower() and config.finetune.provider == "openai":
        console.print(
            "[yellow]! DeepSeek 官方 API 目前没有微调接口。[/yellow]\n"
            '  打分/数据处理可以直接用；训练请改 finetune.provider = "generic" 导出 JSONL，\n'
            "  或把 base_url 指向支持 fine-tune 的平台（OpenAI / 百炼等）。"
        )

    # 连通性：GET /models 是最轻量的探测点
    try:
        import httpx

        response = httpx.get(
            f"{config.api.base_url}/models",
            headers={"Authorization": f"Bearer {config.api.resolved_key}"},
            timeout=15.0,
        )
        if response.status_code < 400:
            console.print(f"[green]OK 接口连通[/green] {config.api.base_url}")
        else:
            console.print(f"[yellow]! 接口返回 {response.status_code}[/yellow] {response.text[:200]}")
    except Exception as exc:
        console.print(f"[yellow]! 连通性检查失败[/yellow] {exc}")

    # DeepSeek 专有的余额接口；别的网关 404 就静默跳过
    try:
        import httpx

        balance = httpx.get(
            f"{config.api.base_url}/user/balance",
            headers={"Authorization": f"Bearer {config.api.resolved_key}"},
            timeout=10.0,
        )
        if balance.status_code == 200:
            payload = balance.json()
            infos = payload.get("balance_infos") or []
            parts = [
                f"{item.get('currency', '')} {item.get('total_balance', '')}"
                for item in infos
                if isinstance(item, dict) and item.get("total_balance")
            ]
            state = "可用" if payload.get("is_available") else "不可用"
            if parts:
                console.print(f"[green]余额（{state}）[/green] " + " · ".join(parts))
            else:
                console.print(f"[yellow]余额（{state}）[/yellow] 响应里没有金额字段")
    except Exception:
        pass


# ---------------------------------------------------------------------------- score
@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=None)
@click.option("--min-score", type=int, default=None, help="保留阈值（默认取 scoring.min_score）")
@click.option("--rescore", "rescore", is_flag=True, help="忽略已有云端分数，全部重打")
@click.option("--limit", type=int, default=None, help="只打分前 N 条（试跑用）")
@click.option("--workers", type=int, default=None, help="并发数（默认 api.max_concurrency）")
@click.option("--no-cache", is_flag=True, help="不读也不写本地缓存")
@click.option("--dry-run", is_flag=True, help="只统计条数与 token，不发请求")
@click.pass_context
@handle_errors
def score(
    ctx: click.Context,
    input_path: Path,
    output_path: Path | None,
    min_score: int | None,
    rescore: bool,
    limit: int | None,
    workers: int | None,
    no_cache: bool,
    dry_run: bool,
) -> None:
    """用 API 给每条对话打 1-5 分（可断点续跑）。"""
    config = _config(ctx)
    records = read_records(input_path)
    threshold = config.scoring.min_score if min_score is None else min_score
    target = output_path or input_path.with_suffix(
        input_path.suffix + DEFAULT_SCORED_SUFFIX if input_path.suffix else DEFAULT_SCORED_SUFFIX
    )

    if dry_run:
        chars = sum(
            sum(len(str(m.get("content") or "")) for m in record.get("messages", []) if isinstance(m, dict))
            for record in records
        )
        estimated = sum(
            count_tokens(str(m.get("content") or ""))
            for record in records
            for m in record.get("messages", [])
            if isinstance(m, dict)
        )
        console.print(
            Panel.fit(
                f"样本：{len(records)} 条\n"
                f"正文字符：{chars:,}\n"
                f"估算输入 token：{estimated:,}\n"
                f"预计请求数：{len(records) if limit is None else min(limit, len(records))}",
                title="dry-run",
            )
        )
        return

    cache = None
    if not no_cache and config.scoring.use_cache:
        cache = CompletionCache(config.scoring.cache_dir)

    worker_count = workers or config.api.max_concurrency
    console.print(
        f"[cyan]打分中[/cyan] {input_path.name} · {len(records)} 条 · 并发 {worker_count} · "
        f"模型 {config.api.model}"
    )

    with ChatClient(config.api, cache=cache) as client:
        outcomes = score_records(
            records,
            client,
            max_chars=config.scoring.max_chars,
            max_workers=worker_count,
            request_interval=config.scoring.request_interval,
            rescore=rescore,
            limit=limit,
            use_cache=not no_cache,
            progress=lambda it, total: tqdm(it, total=total, unit="条", desc="scoring"),
        )

    applied, failed = apply_outcomes(records, outcomes)
    report = build_report(records, outcomes, min_score=threshold, model=config.api.model)
    report["failed_to_apply"] = failed
    write_json(target, records)
    report_path = write_report(target.with_suffix(".report.json"), report)

    console.print(
        Panel.fit(
            f"写入：{target}\n"
            f"成功 {applied} · 失败 {failed} · 缓存命中 {report['cache_hits']}\n"
            f"平均分 {report['mean_score']} · ≥{threshold} 分保留 {report['kept']} 条\n"
            f"分布 {report['distribution']}\n"
            f"{_usage_line(config, client.usage_total)}\n"
            f"报告：{report_path}",
            title="score 完成",
        )
    )
    if failed:
        console.print("[yellow]部分样本失败，重跑同一条命令即可续跑（成功的会命中缓存）。[/yellow]")


# --------------------------------------------------------------------------- export
@main.command("export")
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("finetune_data"))
@click.option("--system", default=DEFAULT_SYSTEM, help="统一注入的 system prompt")
@click.option("--no-system", is_flag=True, help="不注入 system")
@click.option("--min-score", type=int, default=None)
@click.option("--val-ratio", type=float, default=0.05, help="验证集比例（0 表示不切分）")
@click.option("--seed", type=int, default=42)
@click.option("--max-tokens", "max_tokens", type=int, default=None, help="单条样本 token 上限，超出丢弃")
@click.option("--price-per-million", type=float, default=None, help="训练单价（$/1M tokens），用于估算")
@click.option("--epochs", type=int, default=None)
@click.pass_context
@handle_errors
def export_cmd(
    ctx: click.Context,
    input_path: Path,
    out_dir: Path,
    system: str,
    no_system: bool,
    min_score: int | None,
    val_ratio: float,
    seed: int,
    max_tokens: int | None,
    price_per_million: float | None,
    epochs: int | None,
) -> None:
    """把带分数的样本导出成 OpenAI 微调 JSONL。"""
    config = _config(ctx)
    threshold = config.scoring.min_score if min_score is None else min_score
    records = read_records(input_path)
    manifest = export_dataset(
        records,
        out_dir,
        system=None if no_system else system,
        min_score=threshold,
        val_ratio=val_ratio,
        seed=seed,
        max_tokens_per_row=max_tokens,
        base_model=config.finetune.base_model,
    )
    console.print(
        Panel.fit(
            f"导出 {manifest['exported']} 条（train {manifest['train']} / val {manifest['val']}）\n"
            f"丢弃：{manifest['dropped'] or '无'}\n"
            f"token：均值 {manifest['tokens']['mean']} · p95 {manifest['tokens']['p95']} · "
            f"合计 {manifest['tokens']['total']:,}\n"
            f"目录：{out_dir}",
            title="export 完成",
        )
    )
    if price_per_million is not None:
        cost = estimate_cost(
            manifest, price_per_million=price_per_million, epochs=epochs or config.finetune.epochs
        )
        console.print(f"[cyan]训练费用粗估[/cyan] 约 ${cost}（{epochs or config.finetune.epochs} epochs）")
    console.print(
        "[dim]下一步：weclone-cloud finetune --train "
        f"{out_dir / 'train.jsonl'} --val {out_dir / 'val.jsonl'}[/dim]"
    )


# ------------------------------------------------------------------------- finetune
@main.command()
@click.option("--train", "train_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--val", "val_path", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--model", "base_model", default=None, help="基座模型（默认 finetune.base_model）")
@click.option("--epochs", type=int, default=None)
@click.option("--suffix", default=None, help="微调模型名后缀")
@click.option("--style", type=click.Choice(["legacy", "method"]), default=None)
@click.option("--state", "state_path", type=click.Path(path_type=Path), default=None)
@click.option("--watch/--no-watch", default=True, help="是否轮询直到结束")
@click.option("--dry-run", is_flag=True, help="只打印将发送的请求体")
@click.pass_context
@handle_errors
def finetune(
    ctx: click.Context,
    train_path: Path,
    val_path: Path | None,
    base_model: str | None,
    epochs: int | None,
    suffix: str | None,
    style: str | None,
    state_path: Path | None,
    watch: bool,
    dry_run: bool,
) -> None:
    """上传 JSONL 并创建云端微调任务。"""
    config = _config(ctx)
    settings = config.finetune
    if style:
        settings.api_style = style  # type: ignore[assignment]
    state_file = state_path or Path(settings.state_file)
    state = load_state(state_file)

    if settings.provider == "generic":
        console.print(
            Panel.fit(
                "当前 finetune.provider = generic：本工具不调用微调接口。\n"
                f"请把 {train_path} 上传到云平台控制台，拿到模型名后写入 {state_file}\n"
                f'（格式：{{"model": "你的模型名", "status": "succeeded"}}），再运行 weclone-cloud serve。',
                title="手动模式",
            )
        )
        return

    body_preview = {
        "model": base_model or settings.base_model,
        "training_file": "<upload>",
        "validation_file": "<upload>" if val_path else None,
        "suffix": suffix or settings.suffix,
        "api_style": settings.api_style,
    }
    if dry_run:
        console.print_json(data=body_preview)
        return

    with FineTuneClient(settings) as client:
        if not state.file_id:
            state.file_id = client.upload(train_path)
            console.print(f"[green]OK[/green] 训练文件已上传：{state.file_id}")
        if val_path and not state.validation_file_id:
            state.validation_file_id = client.upload(val_path)
            console.print(f"[green]OK[/green] 验证文件已上传：{state.validation_file_id}")

        if not state.job_id:
            job = client.create(
                state.file_id,
                validation_file=state.validation_file_id or None,
                model=base_model,
                epochs=epochs,
                suffix=suffix,
            )
            state.job_id = str(job.get("id") or "")
            state.status = str(job.get("status") or "")
            state.base_model = base_model or settings.base_model
            state.epochs = settings.epochs if epochs is None else epochs
            save_state(state_file, state)
            console.print(f"[green]OK[/green] 任务已创建：{state.job_id}（{state.status}）")
        else:
            console.print(f"[cyan]复用已有任务[/cyan] {state.job_id}（{state.status}）")

        if watch:

            def on_poll(job: dict[str, Any]) -> None:
                update_state_from_job(state, job)
                save_state(state_file, state)
                hyper = job.get("hyperparameters") or {}
                console.print(
                    f"  [dim]{job.get('status')}[/dim] "
                    f"trained_tokens={job.get('trained_tokens') or '-'} "
                    f"epochs={hyper.get('n_epochs', '?')}"
                )

            job = client.wait(state.job_id, on_poll=on_poll)
            update_state_from_job(state, job)
            save_state(state_file, state)
            console.print(
                Panel.fit(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), title="微调完成")
            )


def update_state_from_job(state: JobState, job: dict[str, Any]) -> None:
    state.status = str(job.get("status") or state.status)
    state.model = str(job.get("fine_tuned_model") or state.model or "")
    state.error = str(job.get("error") or "")
    state.updated_at = str(job.get("finished_at") or "")


# --------------------------------------------------------------------------- status
@main.command()
@click.option("--job-id", default=None)
@click.option("--watch", is_flag=True)
@click.option("--state", "state_path", type=click.Path(path_type=Path), default=None)
@click.pass_context
@handle_errors
def status(ctx: click.Context, job_id: str | None, watch: bool, state_path: Path | None) -> None:
    """查看微调任务状态与事件。"""
    config = _config(ctx)
    state_file = state_path or Path(config.finetune.state_file)
    state = load_state(state_file)
    job_id = job_id or state.job_id
    if not job_id:
        raise CloudError(f"{state_file} 里没有任务，请先运行 weclone-cloud finetune。")
    with FineTuneClient(config.finetune) as client:
        if watch:

            def on_poll(job: dict[str, Any]) -> None:
                console.print(f"  [dim]{job.get('status')}[/dim]")

            job = client.wait(job_id, on_poll=on_poll)
        else:
            job = client.retrieve(job_id)
        console.print_json(data=job)
        if job.get("status") in {"failed", "cancelled"}:
            for event in client.events(job_id)[-10:]:
                console.print(f"  [red]{event.get('level')}[/red] {event.get('message')}")


# --------------------------------------------------------------------------- cancel
@main.command()
@click.option("--state", "state_path", type=click.Path(path_type=Path), default=None)
@click.pass_context
@handle_errors
def cancel(ctx: click.Context, state_path: Path | None) -> None:
    """取消当前微调任务。"""
    config = _config(ctx)
    state_file = state_path or Path(config.finetune.state_file)
    state = load_state(state_file)
    if not state.job_id:
        raise CloudError("没有可取消的任务。")
    with FineTuneClient(config.finetune) as client:
        job = client.cancel(state.job_id)
    console.print(f"[yellow]已请求取消[/yellow] {state.job_id} → {job.get('status')}")


DEFAULT_STYLE = """- 句子短、口语化，不用书面语，不要分点罗列，不要长篇解释
- 保留原文的语气词、口头禅、标点习惯（包括省略号、重复、错别字）
- 被问到不知道的事，用一句短话带过或反问，不要编造大段内容
- 不要提到“作为 AI”“语言模型”“示例”等字眼"""

PERSONA_TEMPLATE = """你在扮演一个真实的人，别人不知道你是 AI。下面是「我」的真实聊天片段，请严格模仿其说话方式。

风格要求：
{style}

【真实聊天示例】
{examples}

【输出要求】
只输出「我」会说的那句话，不要旁白、不要解释、不要加引号。"""


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=Path("persona_system.md"))
@click.option("--examples", type=int, default=6, help="少样本示例条数")
@click.option("--min-score", type=int, default=4)
@click.option("--max-chars", type=int, default=600, help="每条示例最多字符")
@click.option("--summarize/--no-summarize", default=True, help="用 DeepSeek 先总结说话风格")
@click.pass_context
@handle_errors
def persona(
    ctx: click.Context,
    input_path: Path,
    out_path: Path,
    examples: int,
    min_score: int,
    max_chars: int,
    summarize: bool,
) -> None:
    """用高分样本生成人设提示词（DeepSeek 不微调时的替代方案）。"""
    from .prompts import STYLE_SUMMARY_SYSTEM, STYLE_SUMMARY_USER, render_conversation

    records = read_records(input_path)

    def pick(threshold: int) -> list[str]:
        chosen: list[str] = []
        seen: set[str] = set()
        for record in sorted(records, key=lambda row: -(row.get("score") or 0)):
            if len(chosen) >= examples:
                break
            score = record.get("score")
            if not isinstance(score, int) or score < threshold or record.get("images"):
                continue
            dialogue = [
                message
                for message in record.get("messages", [])
                if isinstance(message, dict)
                and message.get("role") in {"user", "assistant"}
                and str(message.get("content") or "").strip()
            ]
            if not 2 <= len(dialogue) <= 8:
                continue
            digest = "|".join(str(message.get("content")) for message in dialogue)
            if digest in seen:
                continue
            seen.add(digest)
            chosen.append(render_conversation(dialogue, max_chars=max_chars))
        return chosen

    picked = pick(min_score)
    if not picked:
        # 样本少或打分偏严时自动降阈值，别让人设生成直接失败
        for threshold in range(min_score - 1, 1, -1):
            picked = pick(threshold)
            if picked:
                console.print(f"[yellow]! {min_score} 分以上样本不足，自动退到 ≥{threshold} 分[/yellow]")
                break
    if not picked:
        raise CloudError(
            "没有满足条件的样本（2-8 轮对话）。先运行 weclone-cloud score，或检查数据集是否为空。"
        )

    style = ""
    if summarize:
        config = _config(ctx)
        try:
            with ChatClient(config.api, cache=CompletionCache(config.scoring.cache_dir)) as client:
                completion = client.chat(
                    [
                        {"role": "system", "content": STYLE_SUMMARY_SYSTEM},
                        {
                            "role": "user",
                            "content": STYLE_SUMMARY_USER.format(conversation="\n\n".join(picked)),
                        },
                    ],
                    max_tokens=600,
                )
            style = completion.text.strip()
        except Exception as exc:  # 网络/限额/解析失败都不该阻断人设生成
            console.print(f"[yellow]! 风格总结失败，退回通用规则：{exc}[/yellow]")

    text = PERSONA_TEMPLATE.format(style=style or DEFAULT_STYLE, examples="\n\n".join(picked))
    out_path.write_text(text, encoding="utf-8")
    console.print(
        Panel.fit(
            f"已生成：{out_path}\n"
            f"示例：{len(picked)} 条 · 风格要点：{'AI 总结 ' + str(len(style.splitlines())) + ' 条' if style else '通用规则'}\n"
            f"约 {len(text)} 字（每次请求会占用相应 token）\n\n"
            f"启动方式：weclone-cloud serve --persona {out_path}",
            title="persona 完成",
        )
    )
    console.print(
        "[dim]这是「不微调」方案：DeepSeek 基座 + AI 总结的风格规则 + 你的真实对话示例。"
        "能对齐句长/语气词/口头禅，但不改变模型权重。[/dim]"
    )


# --------------------------------------------------------------------------- serve
@main.command()
@click.option("--host", default=None)
@click.option("--port", type=int, default=None)
@click.option("--model", default=None, help="模型名（默认：微调模型 → api.model）")
@click.option("--persona", type=click.Path(path_type=Path), default=None, help="persona_system.md 路径")
@click.pass_context
@handle_errors
def serve(
    ctx: click.Context,
    host: str | None,
    port: int | None,
    model: str | None,
    persona: Path | None,
) -> None:
    """启动 OpenAI 兼容的数字分身接口（含内置网页）。"""
    import uvicorn

    from .serve import create_app, resolve_model_id

    config = _config(ctx)
    if model:
        config.serve.model = model
    if persona:
        config.serve.system_prompt_file = str(persona)
    resolved = resolve_model_id(config)
    fine_tuned = bool(model) or bool(load_state(config.finetune.state_file).model)
    app = create_app(config, model_id=resolved)
    host = host or config.serve.host
    port = port or config.serve.port
    console.print(
        Panel.fit(
            f"接口：http://{host}:{port}/v1/chat/completions\n"
            f"网页：http://{host}:{port}/\n"
            f"模型：{resolved}（{'微调模型' if fine_tuned else '基座模型 + 人设提示词'}）\n"
            f"人设：{config.serve.system_prompt_file or '内置 system_prompt'}\n"
            f"鉴权：{'Bearer ' + config.serve.api_key_env if config.serve.require_api_key else '关闭'}",
            title="weclone-cloud serve",
        )
    )
    uvicorn.run(app, host=host, port=port)


# ------------------------------------------------------------------------- pipeline
@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("build"))
@click.option("--min-score", type=int, default=None)
@click.option("--system", default=DEFAULT_SYSTEM)
@click.option("--val-ratio", type=float, default=0.05)
@click.option("--rescore", is_flag=True)
@click.option("--limit", type=int, default=None)
@click.pass_context
@handle_errors
def pipeline(
    ctx: click.Context,
    input_path: Path,
    out_dir: Path,
    min_score: int | None,
    system: str,
    val_ratio: float,
    rescore: bool,
    limit: int | None,
) -> None:
    """score → export 一条龙。"""
    config = _config(ctx)
    threshold = config.scoring.min_score if min_score is None else min_score
    records = read_records(input_path)
    cache = CompletionCache(config.scoring.cache_dir) if config.scoring.use_cache else None

    with ChatClient(config.api, cache=cache) as client:
        outcomes = score_records(
            records,
            client,
            max_chars=config.scoring.max_chars,
            max_workers=config.api.max_concurrency,
            request_interval=config.scoring.request_interval,
            rescore=rescore,
            limit=limit,
            progress=lambda it, total: tqdm(it, total=total, unit="条", desc="scoring"),
        )

    apply_outcomes(records, outcomes)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "scored.json", records)
    report = build_report(records, outcomes, min_score=threshold, model=config.api.model)
    write_report(out_dir / "score_report.json", report)

    kept = filter_records(records, min_score=threshold)
    manifest = export_dataset(
        kept,
        out_dir / "finetune",
        system=system,
        min_score=threshold,
        val_ratio=val_ratio,
        base_model=config.finetune.base_model,
    )
    console.print(
        Panel.fit(
            f"打分：{report['scored']} 条，平均 {report['mean_score']}，保留 {report['kept']}\n"
            f"{_usage_line(config, client.usage_total)}\n"
            f"导出：train {manifest['train']} / val {manifest['val']}，"
            f"合计 {manifest['tokens']['total']:,} tokens\n"
            f"产物：{out_dir}\n\n"
            f"下一步：weclone-cloud finetune --train {out_dir / 'finetune' / 'train.jsonl'}",
            title="pipeline 完成",
        )
    )


if __name__ == "__main__":
    main()
