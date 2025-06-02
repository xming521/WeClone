import click
import commentjson
from pathlib import Path
import os
import sys
import functools

from weclone.utils.log import logger, capture_output
from weclone.utils.config import load_config

cli_config: dict | None = None

try:
    import tomllib  # type: ignore Python 3.11+
except ImportError:
    import tomli as tomllib


def clear_argv(func):
    """
    装饰器：在调用被装饰函数前，清理 sys.argv，只保留脚本名。调用后恢复原始 sys.argv。
    用于防止参数被 Hugging Face HfArgumentParser 解析造成 ValueError。
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        original_argv = sys.argv.copy()
        sys.argv = [original_argv[0]]  # 只保留脚本名
        try:
            return func(*args, **kwargs)
        finally:
            sys.argv = original_argv  # 恢复原始 sys.argv

    return wrapper


def apply_common_decorators(capture_output_enabled=False):
    """
    A unified decorator for applications
    """

    def decorator(original_cmd_func):
        @functools.wraps(original_cmd_func)
        def new_runtime_wrapper(*args, **kwargs):
            if cli_config and cli_config.get("full_log", False):
                return capture_output(original_cmd_func)(*args, **kwargs)
            else:
                return original_cmd_func(*args, **kwargs)

        func_with_clear_argv = clear_argv(new_runtime_wrapper)

        return functools.wraps(original_cmd_func)(func_with_clear_argv)

    return decorator


@click.group()
def cli():
    """WeClone: 从聊天记录创造数字分身的一站式解决方案"""
    _check_project_root()
    _check_versions()
    global cli_config
    cli_config = load_config(arg_type="cli_args")


@cli.command("make-dataset", help="处理聊天记录CSV文件，生成问答对数据集。")
@apply_common_decorators()
def qa_generator():
    """处理聊天记录CSV文件，生成问答对数据集。"""
    config = load_config(arg_type="make_dataset")

    if "image" in config.get("include_type", []):
        from weclone.data.qa_generatorV2 import DataProcessor

        logger.info("检测到配置包含image类型，使用qa_generatorV2")
    else:
        from weclone.data.qa_generator import DataProcessor

        logger.info("使用标准qa_generator")

    processor = DataProcessor()
    processor.main()


@cli.command("train-sft", help="使用准备好的数据集对模型进行微调。")
@apply_common_decorators()
def train_sft():
    """使用准备好的数据集对模型进行微调。"""
    from weclone.train.train_sft import main as train_sft_main

    train_sft_main()


@cli.command(
    "webchat-demo", help="启动 Web UI 与微调后的模型进行交互测试。"
)  # 命令名修改为 web-demo
@apply_common_decorators()
def web_demo():
    """启动 Web UI 与微调后的模型进行交互测试。"""
    from weclone.eval.web_demo import main as web_demo_main

    web_demo_main()


# TODO 添加评估功能 @cli.command("eval-model", help="使用从训练数据中划分出来的验证集评估。")
@apply_common_decorators()
def eval_model():
    """使用从训练数据中划分出来的验证集评估。"""
    from weclone.eval.eval_model import main as evaluate_main

    evaluate_main()


@cli.command("test-model", help="使用常见聊天问题测试模型。")
@apply_common_decorators()
def test_model():
    """测试"""
    from weclone.eval.test_model import main as test_main

    test_main()


@cli.command("server", help="启动API服务，提供模型推理接口。")
@apply_common_decorators()
def server():
    """启动API服务，提供模型推理接口。"""
    from weclone.server.api_service import main as server_main

    server_main()


def _check_project_root():
    """检查当前目录是否为项目根目录，并验证项目名称。"""
    project_root_marker = "pyproject.toml"
    current_dir = Path(os.getcwd())
    pyproject_path = current_dir / project_root_marker

    if not pyproject_path.is_file():
        logger.error(f"未在当前目录找到 {project_root_marker} 文件。")
        logger.error("请确保在WeClone项目根目录下运行此命令。")
        sys.exit(1)

    try:
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        project_name = pyproject_data.get("project", {}).get("name")
        if project_name != "WeClone":
            logger.error("请确保在正确的 WeClone 项目根目录下运行。")
            sys.exit(1)
    except tomllib.TOMLDecodeError as e:
        logger.error(f"错误：无法解析 {pyproject_path} 文件: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"读取或处理 {pyproject_path} 时发生意外错误: {e}")
        sys.exit(1)


def _check_versions():
    """比较本地 settings.jsonc 版本和 pyproject.toml 中的配置文件指南版本"""
    if tomllib is None:  # Skip check if toml parser failed to import
        return

    ROOT_DIR = Path(__file__).parent.parent
    SETTINGS_PATH = ROOT_DIR / "settings.jsonc"
    PYPROJECT_PATH = ROOT_DIR / "pyproject.toml"

    settings_version = None
    config_guide_version = None
    config_changelog = None

    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                settings_data = commentjson.load(f)
                settings_version = settings_data.get("version")
        except Exception as e:
            logger.error(f"错误：无法读取或解析 {SETTINGS_PATH}: {e}")
            logger.error("请确保 settings.jsonc 文件存在且格式正确。")
            sys.exit(1)
    else:
        logger.error(f"错误：未找到配置文件 {SETTINGS_PATH}。")
        logger.error("请确保 settings.jsonc 文件位于项目根目录。")
        sys.exit(1)

    if PYPROJECT_PATH.exists():
        try:
            with open(PYPROJECT_PATH, "rb") as f:  # tomllib 需要二进制模式
                pyproject_data = tomllib.load(f)
                weclone_tool_data = pyproject_data.get("tool", {}).get("weclone", {})
                config_guide_version = weclone_tool_data.get("config_version")
                config_changelog = weclone_tool_data.get("config_changelog", "N/A")
        except Exception as e:
            logger.warning(
                f"警告：无法读取或解析 {PYPROJECT_PATH}: {e}。无法检查配置文件是否为最新。"
            )
    else:
        logger.warning(
            f"警告：未找到文件 {PYPROJECT_PATH}。无法检查配置文件是否为最新。"
        )

    if not settings_version:
        logger.error(f"错误：在 {SETTINGS_PATH} 中未找到 'version' 字段。")
        logger.error("请从 settings.template.json 复制或更新您的 settings.jsonc 文件。")
        sys.exit(1)

    if config_guide_version:
        if settings_version != config_guide_version:
            logger.warning(
                f"警告：您的 settings.jsonc 文件版本 ({settings_version}) 与项目建议的配置版本 ({config_guide_version}) 不一致。"
            )
            logger.warning(
                "这可能导致意外行为或错误。请从 settings.template.json 复制或更新您的 settings.jsonc 文件。"
            )
            # TODO 根据版本号打印更新日志
            logger.warning(f"配置文件更新日志：\n{config_changelog}")
    elif PYPROJECT_PATH.exists():  # 如果文件存在但未读到版本
        logger.warning(
            f"警告：在 {PYPROJECT_PATH} 的 [tool.weclone] 下未找到 'config_version' 字段。"
            "无法确认您的 settings.jsonc 是否为最新配置版本。"
        )


if __name__ == "__main__":
    cli()
