import functools
import os
import sys
from pathlib import Path
from typing import cast

import click
import pyjson5
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from weclone.utils.config import load_config
from weclone.utils.config_models import CliArgs
from weclone.utils.log import capture_output, configure_log_level_from_config, logger

cli_config: CliArgs | None = None

try:
    import tomllib  # type: ignore Python 3.11+
except ImportError:
    import tomli as tomllib


def clear_argv(func):
    """
    Decorator: Clear sys.argv before calling the decorated function, keeping only the script name. Restore original sys.argv after calling.
    Used to prevent arguments from being parsed by Hugging Face HfArgumentParser causing ValueError.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        original_argv = sys.argv.copy()
        sys.argv = [original_argv[0]]  # Keep only script name
        try:
            return func(*args, **kwargs)
        finally:
            sys.argv = original_argv  # Restore original sys.argv

    return wrapper


def with_community_info(func):
    """
    Decorator: Show community info before executing the command
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        show_community_info()
        return func(*args, **kwargs)

    return wrapper


def apply_common_decorators(capture_output_enabled=False):
    """
    A unified decorator for applications
    """

    def decorator(original_cmd_func):
        @functools.wraps(original_cmd_func)
        def new_runtime_wrapper(*args, **kwargs):
            if cli_config and cli_config.full_log:
                return capture_output(original_cmd_func)(*args, **kwargs)
            else:
                return original_cmd_func(*args, **kwargs)

        func_with_clear_argv = clear_argv(new_runtime_wrapper)

        return functools.wraps(original_cmd_func)(func_with_clear_argv)

    return decorator


@click.group(invoke_without_command=True)
@click.option(
    "--config-path",
    default=None,
    help="Specify config file path, or set WECLONE_CONFIG_PATH environment variable",
)
@click.pass_context
def cli(ctx, config_path):
    """WeClone: Build a portrait of your life. Assistants that know you. Clones that talk like you."""
    # Only show community info when no subcommand is invoked
    if ctx.invoked_subcommand is None:
        show_community_info()
        click.echo(ctx.get_help())
        return

    if config_path:
        os.environ["WECLONE_CONFIG_PATH"] = config_path
        logger.info(f"Config file path set to: {config_path}")

    _check_project_root()
    from weclone.utils import secure_storage as secure

    secure.configure(config_path)
    ctx.call_on_close(secure.lock)
    if ctx.invoked_subcommand in {
        "dev",
        "server",
        "server-reset-password",
        "security-init",
        "security-change-password",
        "decrypt",
    }:
        return
    _check_versions()
    global cli_config
    cli_config = cast(CliArgs, load_config(arg_type="cli_args"))

    configure_log_level_from_config()
    if ctx.invoked_subcommand not in {"version", "test-model"}:
        try:
            if secure.configure()["explicit"]:
                secure.ensure_initialized()
            secure.ensure_unlocked()
        except secure.SecurityError as exc:
            raise click.ClickException(str(exc)) from None


@cli.command("make-dataset", help="Process chat history CSV files to generate Q&A pair datasets.")
@with_community_info
@apply_common_decorators()
def qa_generator():
    """Process chat history CSV files to generate Q&A pair datasets."""
    from weclone.data.qa_generator import DataProcessor

    processor = DataProcessor()
    processor.main()


@cli.command(
    "distill-style",
    help="Extract speaking style from a chat snapshot, such as the people directory of a build-profile run.",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    add_help_option=False,
)
@click.pass_context
def distill_style(ctx: click.Context):
    from weclone.data.agent.distill_style.run import main

    config_path = ctx.parent.params.get("config_path") if ctx.parent else None
    argv = ["--settings", str(config_path)] if config_path else []
    main(argv + ctx.args)


@cli.command(
    "build-profile", help="从聊天数据依次完成数据准备、记忆抽取、整理和画像生成，做分身时另外抽取语言风格。"
)
@click.option(
    "--purpose",
    type=click.Choice(["assistant", "clone"]),
    help="assistant：给个人助理用，抽取画像和事件。clone：做分身，另外抽取语言风格，需要 ltp 包和 LTP 模型。未指定时交互选择。",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    help="本次流水线的结果目录；默认在 dataset/res_csv/agent/profile_runs 下新建时间戳目录。",
)
@click.option(
    "--max-context-tokens",
    type=click.IntRange(min=1),
    help="记忆整理和画像生成的输入 token 上限；默认读取 Codex 上下文长度，API 模型需要显式指定。",
)
@apply_common_decorators()
def build_profile(purpose: str | None, output_dir: Path | None, max_context_tokens: int | None):
    from weclone.data.agent.profile_pipeline import run

    if purpose is None:
        click.echo("1. 给个人助理用：抽取画像和事件\n2. 做分身：另外抽取语言风格，需要 ltp 包和 LTP 模型")
        choice = click.prompt("请选择用途", type=click.Choice(["1", "2"]), default="1")
        purpose = "clone" if choice == "2" else "assistant"
    config_path = Path(os.environ.get("WECLONE_CONFIG_PATH", "settings.jsonc"))
    try:
        result = run(
            with_style=purpose == "clone",
            output_dir=output_dir,
            config_path=config_path,
            max_context_tokens=max_context_tokens,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"画像生成完成：{result}")


@cli.command("train-sft", help="Fine-tune the model using prepared datasets.")
@apply_common_decorators()
def train_sft():
    """Fine-tune the model using prepared datasets."""
    from weclone.train.train_sft import main as train_sft_main

    train_sft_main()


@cli.command("train-pt", help="Continue pre-training the model using prepared text datasets.")
@apply_common_decorators()
def train_pt():
    """Continue pre-training the model using prepared text datasets."""
    from weclone.train.train_pt import main as train_pt_main

    train_pt_main()


@cli.command("webchat-demo", help="Launch Web UI for interactive testing with fine-tuned model.")
@apply_common_decorators()
def web_demo():
    """Launch Web UI for interactive testing with fine-tuned model."""
    from weclone.eval.web_demo import main as web_demo_main

    web_demo_main()


# TODO Add evaluation functionality @cli.command("eval-model", help="Evaluate using validation set split from training data.")
@apply_common_decorators()
def eval_model():
    """Evaluate using validation set split from training data."""
    from weclone.eval.eval_model import main as evaluate_main

    evaluate_main()


@cli.command("test-model", help="Test model with common chat questions.")
@apply_common_decorators()
def test_model():
    """Test model with common chat questions."""
    from weclone.eval.test_model import main as test_main

    test_main()


@cli.command("server-reset-password", help="重新设置共用密码；旧密文失效，必须从原始聊天重新生成全部数据。")
def server_reset_password():
    from weclone.utils import secure_storage as secure

    click.confirm("重置后旧密文无法用新密码读取，必须重新处理全部聊天数据。继续？", abort=True)
    password = click.prompt("设置新的共用密码", hide_input=True, confirmation_prompt=True)
    try:
        secure.reset_password(password)
    except secure.SecurityError as exc:
        raise click.ClickException(str(exc)) from None
    click.echo("密码已重置。旧文件保留；请使用新输出目录从原始聊天完整重新生成。")


@cli.command("security-init", help="按 settings.jsonc 固定存储模式并设置网页、文件读写共用密码。")
def security_init():
    from weclone.utils import secure_storage as secure

    if secure.is_initialized():
        raise click.ClickException("已经初始化，不能切换模式；改密或重置请使用对应命令。")
    try:
        secure.ensure_initialized()
    except secure.SecurityError as exc:
        raise click.ClickException(str(exc)) from None
    click.echo(f"初始化完成：{secure.configure()['storage_mode']}。密码未明文保存。")


@cli.command("security-change-password", help="使用旧密码修改共用密码，保留已有加密数据。")
def security_change_password():
    from weclone.utils import secure_storage as secure

    old_password = click.prompt("输入当前密码", hide_input=True)
    new_password = click.prompt("设置新密码", hide_input=True, confirmation_prompt=True)
    try:
        secure.change_password(old_password, new_password)
    except secure.SecurityError as exc:
        raise click.ClickException(str(exc)) from None
    click.echo("密码已修改，已有数据保留，旧网页会话失效。")


@cli.command("decrypt", help="输入共用密码，将加密文件导出到新的明文文件。")
@click.option(
    "--input", "input_path", required=True, type=click.Path(path_type=Path, exists=True, dir_okay=False)
)
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path, dir_okay=False))
def decrypt(input_path: Path, output_path: Path):
    from weclone.utils import secure_storage as secure

    try:
        secure.decrypt_file(input_path, output_path)
    except (secure.SecurityError, OSError) as exc:
        raise click.ClickException(str(exc)) from None
    click.echo(f"已导出明文文件：{output_path}，请妥善保管。")


@cli.command("server", help="Start the WeClone server, optionally with model inference.")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=click.IntRange(1, 65535), default=5175, envvar="API_PORT", show_default=True)
@click.option(
    "--run",
    "run_dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="build-profile result directory to serve; defaults to the newest one.",
)
@click.option(
    "--inference", is_flag=True, help="Load the configured model and enable /v1 inference endpoints."
)
@click.option(
    "--demo",
    type=click.Choice(["zh", "en"]),
    is_flag=False,
    flag_value="zh",
    help="Serve the synthetic test profile (Chinese, or English with en) in a temporary workspace removed on exit. "
    "Password: demo.",
)
@clear_argv
def server(host: str, port: int, run_dir: Path | None, inference: bool, demo: str | None):
    from weclone.server.app import serve

    if demo and run_dir:
        raise click.UsageError("--demo 和 --run 不能同时使用。")
    serve(host=host, port=port, run=run_dir, inference=inference, demo=demo)


@cli.command("dev", help="Start the backend and frontend development server.")
@click.option(
    "--demo",
    type=click.Choice(["zh", "en"]),
    is_flag=False,
    flag_value="zh",
    help="Have the backend serve the synthetic test profile, as server --demo does.",
)
@click.pass_context
def dev(ctx: click.Context, demo: str | None):
    import signal
    import subprocess
    import time

    processes = []
    shutdown_signal = None

    def interrupt(signum, _frame):
        nonlocal shutdown_signal
        shutdown_signal = signum

    def signal_group(process, signum):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    handlers = {
        sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        server = [sys.executable, "-m", "weclone.cli", "server", "--port", "5175"]
        commands = [server + (["--demo", demo] if demo else []), ["pnpm", "--dir", "web", "dev"]]
        for command in commands:
            if shutdown_signal is not None:
                ctx.exit(128 + shutdown_signal)
            processes.append(subprocess.Popen(command, start_new_session=True))
        click.echo("前端 dev: http://127.0.0.1:5174\n后端: http://127.0.0.1:5175\n按 Ctrl+C 关闭全部服务。")
        while shutdown_signal is None:
            for process in processes:
                code = process.poll()
                if code is not None:
                    ctx.exit(code if code >= 0 else 128 - code)
            time.sleep(0.1)
        ctx.exit(128 + shutdown_signal)
    except OSError as exc:
        raise click.ClickException(str(exc)) from None
    finally:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            for process in processes:
                signal_group(process, signal.SIGTERM)
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                signal_group(process, signal.SIGKILL)
                process.wait()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


@cli.command(
    "organize-memories",
    help="Induce life domains, classify, group into topics, and summarize extracted memories.",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    add_help_option=False,
)
@click.pass_context
def organize_memories(ctx: click.Context):
    """Pass stage options to the memory organization CLI."""
    from weclone.data.agent import organize_memories as organizer

    argv = list(ctx.args)
    config_path = ctx.parent.params.get("config_path") if ctx.parent else None
    if config_path and not any(arg == "--config-path" or arg.startswith("--config-path=") for arg in argv):
        argv.extend(("--config-path", config_path))
    organizer.main(argv)


@cli.command("version", help="Show WeClone version information.")
@with_community_info
def version():
    """Show WeClone version information."""
    pass


def show_community_info():
    console = Console()
    content = Text()
    content.append("📱 Official group\n", style="bold green")
    content.append("   • Telegram: ", style="bold cyan")
    content.append("https://t.me/+JEdak4m0XEQ3NGNl\n", style="bright_blue")
    content.append("   • QQ群: ", style="bold cyan")
    content.append("708067078\n\n", style="bright_green")
    content.append("🌐 Social media\n", style="bold magenta")
    content.append("   • Twitter: ", style="bold cyan")
    content.append("https://x.com/weclone567\n", style="bright_blue")
    content.append("   • 小红书: ", style="bold cyan")
    content.append("🔍 搜索WeClone\n\n", style="bright_blue")
    content.append("📚 Official resources\n", style="bold red")
    content.append("   • Repository: ", style="bold cyan")
    content.append("https://github.com/xming521/WeClone\n", style="bright_blue")
    content.append("   • Homepage: ", style="bold cyan")
    content.append("https://www.weclone.love/\n", style="bright_blue")
    content.append("   • Document: ", style="bold cyan")
    content.append("https://docs.weclone.love/\n\n", style="bright_blue")
    content.append("💡 感谢您的关注和支持！Thank you for your support!", style="bold bright_green")
    panel = Panel(
        content,
        title="🌟 Community & Social Media",
        title_align="center",
        border_style="bright_cyan",
        padding=(1, 2),
    )
    console.print(panel)


def _check_project_root():
    """Check if current directory is project root and verify project name."""
    project_root_marker = "pyproject.toml"
    current_dir = Path(os.getcwd())
    pyproject_path = current_dir / project_root_marker

    if not pyproject_path.is_file():
        logger.error(f"{project_root_marker} file not found in current directory.")
        logger.error("Please ensure you are running this command in the WeClone project root directory.")
        sys.exit(1)

    try:
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        project_name = pyproject_data.get("project", {}).get("name")
        if project_name != "WeClone":
            logger.error("Please ensure you are running in the correct WeClone project root directory.")
            sys.exit(1)
    except tomllib.TOMLDecodeError as e:
        logger.error(f"Error: Unable to parse {pyproject_path} file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error occurred while reading or processing {pyproject_path}: {e}")
        sys.exit(1)


def _check_versions():
    """Compare local settings.jsonc version with config file guide version in pyproject.toml"""
    if tomllib is None:  # Skip check if toml parser failed to import
        return

    ROOT_DIR = Path(__file__).parent.parent
    SETTINGS_PATH = ROOT_DIR / "settings.jsonc"
    PYPROJECT_PATH = ROOT_DIR / "pyproject.toml"

    settings_version = None
    config_guide_version = None
    project_version = None

    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                content = f.read()
                settings_data = pyjson5.loads(content)
                settings_version = settings_data.get("version")
        except Exception as e:
            logger.error(f"Error: Unable to read or parse {SETTINGS_PATH}: {e}")
            logger.error("Please ensure settings.jsonc file exists and is properly formatted.")
            sys.exit(1)
    else:
        logger.error(f"Error: Config file {SETTINGS_PATH} not found.")
        logger.error("Please ensure settings.jsonc file is located in the project root directory.")
        sys.exit(1)

    if PYPROJECT_PATH.exists():
        try:
            with open(PYPROJECT_PATH, "rb") as f:  # tomllib requires binary mode
                pyproject_data = tomllib.load(f)
                weclone_tool_data = pyproject_data.get("tool", {}).get("weclone", {})
                config_guide_version = weclone_tool_data.get("config_version")
                project_version = pyproject_data.get("project", {}).get("version")
        except Exception as e:
            logger.warning(
                f"Warning: Unable to read or parse {PYPROJECT_PATH}: {e}. Cannot check if config file is up to date."
            )
    else:
        logger.warning(
            f"Warning: File {PYPROJECT_PATH} not found. Cannot check if config file is up to date."
        )

    if not settings_version:
        logger.error(f"Error: 'version' field not found in {SETTINGS_PATH}.")
        logger.error("Please copy from settings.template.jsonc or update your settings.jsonc file.")
        sys.exit(1)

    if config_guide_version:
        if settings_version != config_guide_version:
            logger.warning(
                f"Warning: Your settings.jsonc file version ({settings_version}) does not match the project's recommended config version ({config_guide_version})."
            )
            logger.warning(
                "This may cause unexpected behavior or errors. Please copy from settings.template.jsonc or update your settings.jsonc file."
            )

        logger.info(f"📦 Project Version: {project_version}")
        logger.info(f"⚙️  Config Version: {settings_version}")
    elif PYPROJECT_PATH.exists():  # If file exists but version not found
        logger.warning(
            f"Warning: 'config_version' field not found under [tool.weclone] in {PYPROJECT_PATH}. "
            "Cannot confirm if your settings.jsonc is the latest config version."
        )


if __name__ == "__main__":
    cli()
