import functools
import os
import shutil
import subprocess
import sys
import time
from typing import Optional, Union, cast
from unittest import mock

import pytest

from weclone.utils.config import load_config
from weclone.utils.config_models import DataModality, WCMakeDatasetConfig
from weclone.utils.log import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_CSV_DIR = os.path.join(PROJECT_ROOT, "dataset", "csv")
TESTS_DIR = os.path.dirname(__file__)
TEST_DATA_PERSON_DIR = os.path.join(TESTS_DIR, "tests_data", "test_person")

test_logger = logger.bind()
test_logger.remove()
test_logger.add(
    sys.stderr,
    format="<yellow><b>{message}</b></yellow>",
    colorize=True,
    level="INFO",
)

def get_config_files():
    """获取所有配置文件"""
    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    config_files = []
    for file in os.listdir(configs_dir):
        if file.endswith('.jsonc'):
            config_files.append(f"tests/configs/{file}")
    return config_files

def print_test_header(test_name: str, config_file: str = ""):
    line_length = 100
    test_logger.info("\n" + "─" * line_length)
    if config_file:
        title = f"  Testing Phase: {test_name} | Config: {os.path.basename(config_file)}  "
    else:
        title = f"  Testing Phase: {test_name}  "
    padding_total = line_length - len(title)
    padding_left = padding_total // 2
    padding_right = padding_total - padding_left
    test_logger.info(" " * padding_left + title + " " * padding_right)
    test_logger.info("─" * line_length)

def print_config_header(config_file: str):
    """打印配置文件开始测试的头部"""
    line_length = 120
    test_logger.info("\n" + "═" * line_length)
    title = f"  开始测试配置文件: {os.path.basename(config_file)}  "
    padding_total = line_length - len(title)
    padding_left = padding_total // 2
    padding_right = padding_total - padding_left
    test_logger.info(" " * padding_left + title + " " * padding_right)
    test_logger.info("═" * line_length)

def set_test_env():
    if os.path.exists("model_output"): 
        shutil.rmtree("model_output")
    if os.path.exists(DATASET_CSV_DIR):
        shutil.rmtree(DATASET_CSV_DIR)
    os.makedirs(DATASET_CSV_DIR)
    
    test_person_csv_dir = os.path.join(DATASET_CSV_DIR, "test_person")
    os.makedirs(test_person_csv_dir)

    for item_name in os.listdir(TEST_DATA_PERSON_DIR):
        source_item_path = os.path.join(TEST_DATA_PERSON_DIR, item_name)
        if os.path.isfile(source_item_path) and item_name.lower().endswith('.csv'):
            destination_item_path = os.path.join(test_person_csv_dir, item_name)
            shutil.copy2(source_item_path, destination_item_path)
        

def run_cli_command(command: list[str], config_path: str, timeout: int | None = None, background: bool = False) -> Union[subprocess.CompletedProcess, subprocess.Popen]:
    """Execute a CLI command and return the result.
    
    Args:
        command: List of commands to execute.
        config_path: Path to the configuration file.
        timeout: Timeout in seconds.
        background: Whether to run in the background.
        
    Returns:
        If background=True, returns a Popen object; otherwise, returns a CompletedProcess object.
    """
    env = os.environ.copy()
    env["WECLONE_CONFIG_PATH"] = config_path # Set environment variable

    if background:
        process = subprocess.Popen(
            [sys.executable, "-m", "weclone.cli"] + command,
            stderr=None,
            stdout=None,
            text=True,
            cwd=PROJECT_ROOT_DIR,
            env=env
        )
        time.sleep(2)
        return process
    else:
        process = subprocess.run(
            [sys.executable, "-m", "weclone.cli"] + command,
            stderr=None,
            stdout=None,
            text=True,
            cwd=PROJECT_ROOT_DIR,  # Execute in the project root directory
            timeout=timeout,
            env=env  # Pass the modified environment variables
        )
        return process

def load_config_with_path(config_file: str, config_section: str):
    """临时设置环境变量并加载配置"""
    original_env = os.environ.get("WECLONE_CONFIG_PATH")
    os.environ["WECLONE_CONFIG_PATH"] = config_file
    
    try:
        return load_config(config_section)
    finally:
        # 恢复原始环境变量
        if original_env is not None:
            os.environ["WECLONE_CONFIG_PATH"] = original_env
        elif "WECLONE_CONFIG_PATH" in os.environ:
            del os.environ["WECLONE_CONFIG_PATH"]

def run_make_dataset_test(config_file: str):
    """执行 make-dataset 测试"""
    print_test_header("make-dataset", config_file)
    
    config: WCMakeDatasetConfig = cast(WCMakeDatasetConfig, load_config_with_path(config_file, "make_dataset"))
    if DataModality.IMAGE in config.include_type:
        #复制图片到media_dir/iamges
        os.makedirs(config.media_dir, exist_ok=True)
        os.makedirs(os.path.join(config.media_dir, "images"), exist_ok=True)
        for file in os.listdir(os.path.join(PROJECT_ROOT_DIR, "tests", "tests_data", "images")):
            shutil.copy(os.path.join(PROJECT_ROOT_DIR, "tests", "tests_data", "images", file), os.path.join(config.media_dir, "images", file))

    result = run_cli_command(["make-dataset"], config_file)
    assert result.returncode == 0, f"make-dataset command execution failed for config {config_file}"

def run_train_sft_test(config_file: str):
    """执行 train-sft 测试"""
    print_test_header("train-sft", config_file)
   
    try:
        result = run_cli_command(["train-sft"], config_file) 
        assert result.returncode == 0, f"train-sft command failed or did not fail fast as expected for config {config_file}"
    except subprocess.TimeoutExpired:
        test_logger.info(f"train-sft command terminated due to timeout for config {config_file}, which is acceptable in testing, indicating the command has started execution.")
        pass
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during train-sft command execution for config {config_file}: {e}")

def run_webchat_demo_test(config_file: str):
    """执行 webchat-demo 测试"""
    print_test_header("webchat-demo", config_file)
    
    with mock.patch("weclone.eval.web_demo.main") as mock_main:
        mock_main.return_value = None
        try:
            result = run_cli_command(["webchat-demo"], config_file, timeout=20)
            assert result.returncode == 0, f"webchat-demo command execution failed for config {config_file}"
        except subprocess.TimeoutExpired:
            pass

def run_server_test(config_file: str) -> subprocess.Popen:
    """执行 server 测试，返回进程对象"""
    print_test_header("server (background)", config_file)
    server_process = cast(subprocess.Popen, run_cli_command(["server"], config_file, background=True))
    test_logger.info("等待服务器启动，20秒后检查状态...")
    time.sleep(20)
    assert server_process.poll() is None, f"Server startup failed for config {config_file}"
    test_logger.info(f"使用配置 {config_file} 的服务器已在后台启动")
    return server_process

def run_test_model_test(config_file: str, server_process: subprocess.Popen):
    """执行 test-model 测试并关闭服务器"""
    print_test_header("test-model", config_file)
    try:
        result = run_cli_command(["test-model"], config_file)
        assert result.returncode == 0, f"test-model command execution failed for config {config_file}"
    finally:
        if server_process is not None and server_process.poll() is None:
            test_logger.info(f"测试完成，正在关闭使用配置 {config_file} 的服务器...")
            server_process.terminate()
            server_process.wait(timeout=5)
            if server_process.poll() is None:
                server_process.kill()  # Force kill if the process hasn't terminated
            test_logger.info("服务器已关闭")

@pytest.mark.parametrize("config_file", get_config_files())
def test_full_pipeline_for_config(config_file):
    """为每个配置文件完整执行所有测试步骤"""
    print_config_header(config_file)
    
    set_test_env()
    
    server_process = None
    try:
        # 按顺序执行所有测试步骤
        run_make_dataset_test(config_file)
        run_train_sft_test(config_file)
        run_webchat_demo_test(config_file)
        server_process = run_server_test(config_file)
        run_test_model_test(config_file, server_process)
        
        test_logger.info(f"✅ 配置文件 {os.path.basename(config_file)} 的所有测试已完成")
        
    except Exception as e:
        test_logger.error(f"❌ 配置文件 {os.path.basename(config_file)} 测试失败: {e}")
        if server_process is not None and server_process.poll() is None:
            server_process.terminate()
            server_process.wait(timeout=5)
            if server_process.poll() is None:
                server_process.kill()
        raise

if __name__ == "__main__":
    set_test_env()
