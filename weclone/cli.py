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
    """WeClone: One-stop solution for creating digital avatars from chat history"""
    # Only show community info when no subcommand is invoked
    if ctx.invoked_subcommand is None:
        show_community_info()
        click.echo(ctx.get_help())
        return

    if config_path:
        os.environ["WECLONE_CONFIG_PATH"] = config_path
        logger.info(f"Config file path set to: {config_path}")

    _check_project_root()
    _check_versions()
    global cli_config
    cli_config = cast(CliArgs, load_config(arg_type="cli_args"))

    configure_log_level_from_config()


@cli.command("make-dataset", help="Process chat history CSV files to generate Q&A pair datasets.")
@with_community_info
@apply_common_decorators()
def qa_generator():
    """Process chat history CSV files to generate Q&A pair datasets."""
    from weclone.data.qa_generator import DataProcessor

    processor = DataProcessor()
    processor.main()


@cli.command("train-sft", help="Fine-tune the model using prepared datasets.")
@apply_common_decorators()
def train_sft():
    """Fine-tune the model using prepared datasets."""
    from weclone.train.train_sft import main as train_sft_main

    train_sft_main()


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


@cli.command("server", help="Start API service providing model inference interface.")
@apply_common_decorators()
def server():
    """Start API service providing model inference interface."""
    from weclone.server.api_service import main as server_main

    server_main()


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
    config_changelog = None
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
                config_changelog = weclone_tool_data.get("config_changelog", "N/A")
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
        logger.error("Please copy from settings.template.json or update your settings.jsonc file.")
        sys.exit(1)

    if config_guide_version:
        if settings_version != config_guide_version:
            logger.warning(
                f"Warning: Your settings.jsonc file version ({settings_version}) does not match the project's recommended config version ({config_guide_version})."
            )
            logger.warning(
                "This may cause unexpected behavior or errors. Please copy from settings.template.json or update your settings.jsonc file."
            )
            # TODO Print update log based on version number
            logger.warning(f"Config file changelog:\n{config_changelog}")

        logger.info(f"📦 Project Version: {project_version}")
        logger.info(f"⚙️  Config Version: {settings_version}")
    elif PYPROJECT_PATH.exists():  # If file exists but version not found
        logger.warning(
            f"Warning: 'config_version' field not found under [tool.weclone] in {PYPROJECT_PATH}. "
            "Cannot confirm if your settings.jsonc is the latest config version."
        )


if __name__ == "__main__":
    cli()
