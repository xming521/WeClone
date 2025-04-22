import subprocess
import sys
import os
import time
import shutil
import threading # 导入 threading
from typing import Optional, Union, IO # 导入 IO
import torch
from loguru import logger
from subprocess import Popen

# 配置 Loguru
logger.remove() # 移除默认处理器
current_time = time.strftime('%Y%m%d_%H%M%S')
log_file_path = os.path.join(os.path.dirname(__file__), f"pipeline_test_{current_time}.log") # 日志文件名包含执行时间
logger.add(log_file_path, rotation="10 MB", encoding='utf-8', level="DEBUG", enqueue=True) # 文件记录 DEBUG 级别
logger.add(sys.stdout, colorize=True, format="[test] <green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level.name[0]}</level> | <level>{message}</level>", level="INFO", enqueue=True) # 控制台保持 INFO 级别

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger.info(f"项目根目录: {project_root}")

qa_script = "weclone/data/qa_generator.py"
train_script = "weclone/train/train_sft.py"
api_service_script = "weclone/server/api_service.py"
eval_script = "weclone/eval/test_model.py"
web_demo_script = "weclone/eval/web_demo.py"

DEFAULT_TIMEOUT: Optional[Union[int, float]] = 45
API_STARTUP_WAIT = 20
API_TERMINATE_WAIT = 15
WEB_DEMO_STARTUP_WAIT = 20
WEB_DEMO_TERMINATE_WAIT = 15

STEP_QA = "QA 数据生成"
STEP_TRAIN = "SFT 训练"
STEP_COPY_CKPT = "Checkpoint 复制"
STEP_API_START = "API 服务启动"
STEP_EVAL = "模型评估"
STEP_WEB_DEMO = "Web Demo 启动"

# Mapping from identifiers (script paths or custom keys) to step names
step_identifiers = {
    qa_script: STEP_QA,
    train_script: STEP_TRAIN,
    "copy_checkpoint": STEP_COPY_CKPT, # Custom key for non-script step
    api_service_script: STEP_API_START, # Script associated with starting API
    eval_script: STEP_EVAL,
    web_demo_script: STEP_WEB_DEMO, # Script associated with starting Web Demo
}
# Order for fallback logic
step_order = [STEP_QA, STEP_TRAIN, STEP_COPY_CKPT, STEP_API_START, STEP_EVAL, STEP_WEB_DEMO]

#todo 需要测试前替换成测试的settings.json 测试完再替换回来

class PipelineStepError(Exception):
    """自定义异常类，用于表示 Pipeline 步骤执行失败。"""
    pass

# --- 辅助函数：用于在线程中读取和记录流 ---
def log_stream(stream: Optional[IO[str]], log_func):
    """读取流并使用指定的 log 函数记录每一行。"""
    if stream is None:
        return
    try:
        for line in iter(stream.readline, ''):
            if line:
                log_func(line.strip()) # 去除末尾换行符
    except ValueError:
        # 当 Popen 的 stream 在另一线程中被关闭时，readline 可能会抛出 ValueError
        logger.warning("日志流在读取时似乎已被关闭。")
    except Exception as e:
        # 捕获其他潜在的读取错误
        logger.warning(f"日志流读取时发生未预料的错误: {e}")
    finally:
        if stream:
            try:
                stream.close() # 确保流被关闭
            except Exception as close_e:
                logger.warning(f"关闭日志流时发生错误: {close_e}")

# --- 新增：启动日志流线程的辅助函数 ---
def _start_stream_logging_threads(process: Popen, stdout_log_func=logger.info, stderr_log_func=logger.error) -> tuple[threading.Thread, threading.Thread]:
    """为给定的进程启动 stdout 和 stderr 的日志记录线程。"""
    stdout_thread = threading.Thread(
        target=log_stream,
        args=(process.stdout, stdout_log_func),
        daemon=True
    )
    stderr_thread = threading.Thread(
        target=log_stream,
        args=(process.stderr, stderr_log_func),
        daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    return stdout_thread, stderr_thread


def run_script(script_relative_path: str, timeout: Optional[Union[int, float]] = DEFAULT_TIMEOUT, ignore_timeout_error: bool = False, env: Optional[dict] = None):
    """使用 Popen 执行脚本，通过线程实时记录 stdout/stderr 到 loguru。"""
    script_full_path = os.path.join(project_root, script_relative_path)
    timeout_str = '无限制' if timeout is None else f'{timeout}s'
    env_str = f" (环境变量: {env})" if env else ""
    logger.info(f"--- 开始执行 (流式): {script_relative_path} (超时: {timeout_str}){env_str} ---")
    if not os.path.exists(script_full_path):
        error_msg = f"脚本文件不存在 {script_full_path}"
        logger.error(error_msg)
        raise PipelineStepError(error_msg)

    process: Optional[Popen] = None
    stdout_thread: Optional[threading.Thread] = None
    stderr_thread: Optional[threading.Thread] = None

    # 准备环境变量
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    try:
        process = Popen(
            [sys.executable, script_full_path],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            bufsize=1, # 行缓冲
            env=run_env # 传递环境变量
        )

        # 使用辅助函数启动日志线程
        stdout_thread, stderr_thread = _start_stream_logging_threads(process, logger.debug, logger.debug) # stdout/stderr 都用 debug

        # 等待子进程完成或超时
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            warn_msg = f"{script_relative_path} 执行超时 ({timeout}s)。"
            logger.warning(warn_msg)
            # 尝试优雅地关闭流（可能已被 log_stream 关闭）
            if process.stdout: process.stdout.close()
            if process.stderr: process.stderr.close()
            process.kill() # 强制终止超时进程
            logger.warning(f"已强制终止进程 {process.pid}")
            # 等待 I/O 线程完成（即使进程被 kill，也要尝试读取剩余输出）
            if stdout_thread: stdout_thread.join(timeout=5)
            if stderr_thread: stderr_thread.join(timeout=5)
            if not ignore_timeout_error:
                error_msg = f"{script_relative_path} 执行超时 ({timeout}s) 且未忽略。"
                logger.error(error_msg)
                raise PipelineStepError(error_msg)
            else:
                logger.info("--- 根据设置，超时不视为错误，继续执行后续步骤。 ---")
                return # 忽略超时，函数正常返回

        # 等待日志线程完成（确保所有输出都被记录）
        if stdout_thread: stdout_thread.join()
        if stderr_thread: stderr_thread.join()

        # 检查返回码
        if return_code != 0:
            error_msg = f"{script_relative_path} 执行失败，返回码 {return_code}"
            logger.error(error_msg)
            raise PipelineStepError(error_msg)
        else:
            logger.success(f"--- {script_relative_path} 执行成功 ---")

    except FileNotFoundError:
        error_msg = f"Python 解释器 '{sys.executable}' 或脚本 '{script_full_path}' 未找到。"
        logger.error(error_msg)
        raise PipelineStepError(error_msg)
    except Exception as e:
        # 捕获其他潜在错误 (例如 Popen 本身失败)
        error_msg = f"执行 {script_relative_path} 时发生意外错误: {e}"
        logger.error(error_msg)
        # 尝试确保进程和线程被清理
        if process and process.poll() is None:
            try:
                if process.stdout: process.stdout.close()
                if process.stderr: process.stderr.close()
                process.kill()
                logger.warning(f"因异常 {e}，强制终止进程 {process.pid}")
            except Exception as kill_e:
                logger.error(f"清理过程中强制终止进程失败: {kill_e}")
        if stdout_thread and stdout_thread.is_alive(): stdout_thread.join(timeout=1)
        if stderr_thread and stderr_thread.is_alive(): stderr_thread.join(timeout=1)
        raise PipelineStepError(error_msg)


def start_api_service_background() -> Popen:
    """在后台启动 API 服务脚本，实时记录启动日志，失败时抛出 PipelineStepError。"""
    script_full_path = os.path.join(project_root, api_service_script)
    logger.info(f"--- 尝试在后台启动: {api_service_script} ---")
    if not os.path.exists(script_full_path):
        error_msg = f"脚本文件不存在 {script_full_path}"
        logger.error(error_msg)
        raise PipelineStepError(error_msg)

    process: Optional[Popen] = None
    stdout_thread: Optional[threading.Thread] = None
    stderr_thread: Optional[threading.Thread] = None
    try:
        logger.info(f"启动命令: {[sys.executable, script_full_path]}")
        process = Popen(
            [sys.executable, script_full_path],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            bufsize=1 # 行缓冲
        )

        # 使用辅助函数启动日志线程
        stdout_thread, stderr_thread = _start_stream_logging_threads(process, logger.debug, logger.debug) # stdout/stderr 都用 debug

        logger.info(f"等待 {API_STARTUP_WAIT} 秒让服务初步启动 (日志将实时显示)...")
        time.sleep(API_STARTUP_WAIT)

        # 检查进程是否仍在运行
        if process.poll() is None:
            logger.success(f"--- {api_service_script} 似乎已在后台启动 (进程 PID: {process.pid}) ---")
            # 注意：不 join 日志线程，让它们继续运行
            return process
        else:
            # 进程过早退出
            logger.error(f"{api_service_script} 启动后在 {API_STARTUP_WAIT} 秒内过早退出，返回码 {process.returncode}")
            # 尝试等待日志线程结束以捕获最后输出
            if stdout_thread: stdout_thread.join(timeout=2)
            if stderr_thread: stderr_thread.join(timeout=2)
            # 读取 communicate 获取可能遗漏的最终输出 (虽然理论上线程应该读完了)
            try:
                # 设置短超时，因为进程已退出，communicate 应该立即返回
                stdout, stderr = process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                logger.warning("等待 communicate 超时，可能没有更多输出了。")
                stdout, stderr = "", "" # 假设没有更多输出
            except Exception as comm_e:
                 logger.warning(f"调用 communicate 获取最后输出时出错: {comm_e}")
                 stdout, stderr = "", ""

            error_message = f'''--- EARLY EXIT STDOUT ---
                                {stdout}
                                --- EARLY EXIT STDERR ---
                                {stderr}'''
            logger.error(error_message)
            raise PipelineStepError(f"{api_service_script} 启动失败并过早退出。")

    except FileNotFoundError:
        error_msg = f"Python 解释器 '{sys.executable}' 或脚本 '{script_full_path}' 未找到。"
        logger.error(error_msg)
        raise PipelineStepError(error_msg)
    except Exception as e:
        # 捕获其他启动错误
        error_msg = f"启动 {api_service_script} 时发生意外错误: {e}"
        logger.error(error_msg)
        if process and process.poll() is None:
             logger.warning("捕获到异常，尝试强制终止进程...")
             try:
                 if process.stdout: process.stdout.close()
                 if process.stderr: process.stderr.close()
                 process.kill()
             except Exception as kill_e: logger.error(f"强制终止进程时出错: {kill_e}")
        # 尝试join线程
        if stdout_thread and stdout_thread.is_alive(): stdout_thread.join(timeout=1)
        if stderr_thread and stderr_thread.is_alive(): stderr_thread.join(timeout=1)
        raise PipelineStepError(error_msg)

def stop_api_service(process: Optional[Popen]):
    """停止指定的 API 服务进程，采用更健壮的终止和清理逻辑。"""
    if process and process.poll() is None:
        pid = process.pid # Get PID for logging
        logger.info(f"--- 尝试停止 API 服务 (PID: {pid}) ---")
        try:
            logger.info(f"发送 SIGTERM 信号到进程 {pid}...")
            process.terminate()
            try:
                logger.info(f"等待最多 {API_TERMINATE_WAIT} 秒让进程 {pid} 优雅终止...")
                process.wait(timeout=API_TERMINATE_WAIT)
                logger.info(f"API 服务进程 {pid} 已优雅终止，返回码: {process.returncode}")
                # 进程已终止，尝试获取最终输出
                try:
                    stdout, stderr = process.communicate(timeout=2)
                    if stdout: logger.debug(f"进程 {pid} 最终 STDOUT:\n{stdout.strip()}")
                    if stderr: logger.debug(f"进程 {pid} 最终 STDERR:\n{stderr.strip()}")
                except subprocess.TimeoutExpired:
                    logger.warning(f"获取进程 {pid} 最终输出时超时。")
                except Exception as comm_e:
                    logger.warning(f"获取进程 {pid} 最终输出时出错: {comm_e}")

            except subprocess.TimeoutExpired:
                logger.warning(f"进程 {pid} 优雅终止超时 ({API_TERMINATE_WAIT}s)，发送 SIGKILL 信号...")
                process.kill()
                logger.info(f"等待进程 {pid} 被强制终止...")
                # 在 kill 后等待，应该很快返回。增加安全超时。
                try:
                    process.wait(timeout=5)
                    logger.info(f"API 服务进程 {pid} 已被强制终止。")
                except subprocess.TimeoutExpired:
                     logger.error(f"进程 {pid} 在发送 SIGKILL 后仍然没有终止！")
                except Exception as wait_kill_e:
                     logger.error(f"等待强制终止进程 {pid} 时发生错误: {wait_kill_e}")

                # 尝试在 kill 后获取输出
                try:
                    # 在 kill 后也使用 communicate，它隐式处理等待
                    stdout, stderr = process.communicate(timeout=2)
                    if stdout: logger.warning(f"来自进程 {pid} 的 Kill 后输出 (STDOUT):\n{stdout.strip()}")
                    if stderr: logger.warning(f"来自进程 {pid} 的 Kill 后输出 (STDERR):\n{stderr.strip()}")
                except Exception as comm_e:
                    logger.warning(f"获取进程 {pid} (强制终止后) 输出时出错: {comm_e}")

        except Exception as e:
            logger.error(f"停止 API 服务 (PID: {pid if process else '未知'}) 时发生意外错误: {e}")
            # 如果进程仍然存活，尝试最后一次强制 kill
            if process and process.poll() is None:
                logger.warning(f"最终尝试强制终止进程 {pid}...")
                try:
                    process.kill()
                    process.wait(timeout=5)
                except Exception as final_kill_e:
                    logger.error(f"最终强制终止进程 {pid} 时出错: {final_kill_e}")

    elif process:
        logger.info(f"--- API 服务进程 (PID: {process.pid}) 在尝试停止前已经退出。 ---")
    else:
        logger.debug("--- 无需停止 API 服务 (进程不存在或已为 None) ---")

def start_web_demo_background() -> Popen:
    """在后台启动 Web Demo 脚本，实时记录启动日志，失败时抛出 PipelineStepError。"""
    script_full_path = os.path.join(project_root, web_demo_script)
    logger.info(f"--- 尝试在后台启动: {web_demo_script} ---")
    if not os.path.exists(script_full_path):
        error_msg = f"脚本文件不存在 {script_full_path}"
        logger.error(error_msg)
        raise PipelineStepError(error_msg)

    process: Optional[Popen] = None
    stdout_thread: Optional[threading.Thread] = None
    stderr_thread: Optional[threading.Thread] = None
    try:
        logger.info(f"启动命令: {[sys.executable, script_full_path]}")
        process = Popen(
            [sys.executable, script_full_path],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            bufsize=1 # 行缓冲
        )

        # 使用辅助函数启动日志线程 (stdout/stderr 都用 info)
        stdout_thread, stderr_thread = _start_stream_logging_threads(process, logger.debug, logger.debug) # stdout/stderr 都用 debug


        logger.info(f"等待 {WEB_DEMO_STARTUP_WAIT} 秒让 Web Demo 初步启动 (日志将实时显示)...")
        time.sleep(WEB_DEMO_STARTUP_WAIT)

        # 检查进程是否仍在运行
        if process.poll() is None:
            logger.success(f"--- {web_demo_script} 似乎已在后台启动 (进程 PID: {process.pid}) ---")
            # 注意：不 join 日志线程
            return process
        else:
            # 进程过早退出
            logger.error(f"{web_demo_script} 启动后在 {WEB_DEMO_STARTUP_WAIT} 秒内过早退出，返回码 {process.returncode}")
            if stdout_thread: stdout_thread.join(timeout=2)
            if stderr_thread: stderr_thread.join(timeout=2)
            try:
                stdout, stderr = process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                 logger.warning("等待 communicate 超时，可能没有更多输出了。")
                 stdout, stderr = "", ""
            except Exception as comm_e:
                 logger.warning(f"调用 communicate 获取最后输出时出错: {comm_e}")
                 stdout, stderr = "", ""
            error_message = f'''--- EARLY EXIT STDOUT ---
                                {stdout}
                                --- EARLY EXIT STDERR ---
                                {stderr}'''
            logger.error(error_message)
            raise PipelineStepError(f"{web_demo_script} 启动失败并过早退出。")

    except FileNotFoundError:
        error_msg = f"Python 解释器 '{sys.executable}' 或脚本 '{script_full_path}' 未找到。"
        logger.error(error_msg)
        raise PipelineStepError(error_msg)
    except Exception as e:
        error_msg = f"启动 {web_demo_script} 时发生意外错误: {e}"
        logger.error(error_msg)
        if process and process.poll() is None:
             logger.warning("捕获到异常，尝试强制终止进程...")
             try:
                 if process.stdout: process.stdout.close()
                 if process.stderr: process.stderr.close()
                 process.kill()
             except Exception as kill_e: logger.error(f"强制终止进程时出错: {kill_e}")
        if stdout_thread and stdout_thread.is_alive(): stdout_thread.join(timeout=1)
        if stderr_thread and stderr_thread.is_alive(): stderr_thread.join(timeout=1)
        raise PipelineStepError(error_msg)

def stop_web_demo(process: Optional[Popen]):
    """停止指定的 Web Demo 进程，采用更健壮的终止和清理逻辑。"""
    if process and process.poll() is None:
        pid = process.pid # Get PID for logging
        logger.info(f"--- 尝试停止 Web Demo 服务 (PID: {pid}) ---")
        try:
            logger.info(f"发送 SIGTERM 信号到进程 {pid}...")
            process.terminate()
            try:
                logger.info(f"等待最多 {WEB_DEMO_TERMINATE_WAIT} 秒让进程 {pid} 优雅终止...")
                process.wait(timeout=WEB_DEMO_TERMINATE_WAIT)
                logger.info(f"Web Demo 服务进程 {pid} 已优雅终止，返回码: {process.returncode}")
                # 进程已终止，尝试获取最终输出
                try:
                    stdout, stderr = process.communicate(timeout=2)
                    if stdout: logger.debug(f"进程 {pid} 最终 STDOUT:\n{stdout.strip()}")
                    if stderr: logger.debug(f"进程 {pid} 最终 STDERR:\n{stderr.strip()}")
                except subprocess.TimeoutExpired:
                    logger.warning(f"获取进程 {pid} 最终输出时超时。")
                except Exception as comm_e:
                    logger.warning(f"获取进程 {pid} 最终输出时出错: {comm_e}")

            except subprocess.TimeoutExpired:
                logger.warning(f"进程 {pid} 优雅终止超时 ({WEB_DEMO_TERMINATE_WAIT}s)，发送 SIGKILL 信号...")
                process.kill()
                logger.info(f"等待进程 {pid} 被强制终止...")
                try:
                    process.wait(timeout=5)
                    logger.info(f"Web Demo 服务进程 {pid} 已被强制终止。")
                except subprocess.TimeoutExpired:
                     logger.error(f"进程 {pid} 在发送 SIGKILL 后仍然没有终止！")
                except Exception as wait_kill_e:
                     logger.error(f"等待强制终止进程 {pid} 时发生错误: {wait_kill_e}")

                # 尝试在 kill 后获取输出
                try:
                    stdout, stderr = process.communicate(timeout=2)
                    if stdout: logger.warning(f"来自进程 {pid} 的 Kill 后输出 (STDOUT):\n{stdout.strip()}")
                    if stderr: logger.warning(f"来自进程 {pid} 的 Kill 后输出 (STDERR):\n{stderr.strip()}")
                except Exception as comm_e:
                    logger.warning(f"获取进程 {pid} (强制终止后) 输出时出错: {comm_e}")

        except Exception as e:
            logger.error(f"停止 Web Demo 服务 (PID: {pid if process else '未知'}) 时发生意外错误: {e}")
            # 如果进程仍然存活，尝试最后一次强制 kill
            if process and process.poll() is None:
                logger.warning(f"最终尝试强制终止进程 {pid}...")
                try:
                    process.kill()
                    process.wait(timeout=5)
                except Exception as final_kill_e:
                    logger.error(f"最终强制终止进程 {pid} 时出错: {final_kill_e}")

    elif process:
        logger.info(f"--- Web Demo 服务进程 (PID: {process.pid}) 在尝试停止前已经退出。 ---")
    else:
        logger.debug("--- 无需停止 Web Demo 服务 (进程不存在或已为 None) ---")


if __name__ == "__main__":
    logger.info("="*20 + " 开始执行 WeClone Pipeline 脚本 " + "="*20)

    is_cuda_available = torch.cuda.is_available()
    logger.info("--- CUDA 可用性检查 ---")
    if is_cuda_available:
        gpu_count = torch.cuda.device_count()
        logger.success(f"CUDA 可用 (找到 {gpu_count} 个 GPU)")
        for i in range(gpu_count):
            logger.info(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("CUDA 不可用，将使用 CPU (如果适用)。")
    logger.info("-" * 25)

    steps_completed = []
    api_process: Optional[Popen] = None
    web_demo_process: Optional[Popen] = None

    # 设置哪些步骤需要运行
    run_qa = True
    run_train = True
    run_copy_checkpoint = True # 依赖于 run_train
    run_api = True
    run_eval = True # 依赖于 run_api
    run_web_demo = True # 不依赖于 run_api

    try:
        # 步骤 1: QA Generator
        if run_qa:
            logger.info("-" * 10 + " 步骤 1: QA 数据生成 " + "-" * 10)
            run_script(qa_script)
            steps_completed.append(f"{STEP_QA}: 成功")
        else:
            logger.info(f"{STEP_QA}: 跳过 (配置)")
            steps_completed.append(f"{STEP_QA}: 跳过")

        # 步骤 2: Train SFT
        if run_train:
            logger.info("-" * 10 + " 步骤 2: SFT 训练 " + "-" * 10)

            # --- 开始：添加 Checkpoint 检查 ---
            model_output_dir = os.path.join(project_root, "model_output")
            checkpoint_exists = False
            if os.path.isdir(model_output_dir):
                logger.info(f"检查目录 {model_output_dir} 是否存在 checkpoint...")
                try:
                    for item in os.listdir(model_output_dir):
                        item_path = os.path.join(model_output_dir, item)
                        if os.path.isdir(item_path) and item.startswith("checkpoint"):
                            logger.warning(f"找到现有的 Checkpoint 目录: {item_path}，将跳过训练。")
                            checkpoint_exists = True
                            break
                    if not checkpoint_exists:
                        logger.info("未找到现有的 Checkpoint 目录。")
                except Exception as e:
                    logger.error(f"检查 Checkpoint 时出错: {e}")
                    # Treat check error as reason to skip
                    checkpoint_exists = True
                    logger.warning("由于检查 Checkpoint 时出错，将跳过训练。")
            else:
                logger.info(f"目录 {model_output_dir} 不存在，无需检查 Checkpoint。")

            if checkpoint_exists:
                steps_completed.append(f"{STEP_TRAIN}: 跳过 (存在 Checkpoint)")
                # 如果训练跳过，复制步骤也必须跳过
                logger.info(f"{STEP_COPY_CKPT}: 跳过 (训练未运行)")
                steps_completed.append(f"{STEP_COPY_CKPT}: 跳过 (训练未运行)")
            else:
                # --- 结束：添加 Checkpoint 检查 ---
                # 只有在 checkpoint 不存在时才执行以下操作
                logger.info("没有找到 Checkpoint，继续执行训练步骤。")
                # 删除 model_output 目录
                if os.path.exists(model_output_dir):
                    logger.info(f"删除现有的 model_output 目录: {model_output_dir}")
                    try:
                        shutil.rmtree(model_output_dir)
                        logger.success("成功删除 model_output 目录")
                    except Exception as e:
                        logger.error(f"删除 model_output 目录时出错: {e}")
                        # Let's log and raise, consistent with other errors.
                        raise PipelineStepError(f"删除 model_output 目录失败: {e}")

                # 尝试禁用 tqdm
                run_script(train_script, timeout=2000, ignore_timeout_error=True, env={'TQDM_DISABLE': '1'})
                steps_completed.append(f"{STEP_TRAIN}: 成功或超时跳过")

                # 步骤 2.1: 复制 Checkpoint (只有在训练运行后才可能执行)
                if run_copy_checkpoint:
                    logger.info("-" * 10 + " 步骤 2.1: 复制 Checkpoint 到 model_output " + "-" * 10)
                    source_dir = os.path.join(project_root, "model_output", "checkpoint-2") # Note: This assumes checkpoint-2 specifically.
                    dest_dir = os.path.join(project_root, "model_output")
                    if os.path.isdir(source_dir):
                        try:
                            logger.info(f"开始将 {source_dir} 的内容复制到 {dest_dir}...")
                            shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
                            logger.success(f"--- {STEP_COPY_CKPT} 成功 ---")
                            steps_completed.append(f"{STEP_COPY_CKPT}: 成功")
                        except Exception as e:
                            # Embed identifier in the error for the except block
                            error_msg = f"{STEP_COPY_CKPT} 时发生错误: {e}"
                            logger.error(error_msg)
                            # Add a unique marker to identify this step in the except block
                            raise PipelineStepError(f"{error_msg} ###step_id:copy_checkpoint###")
                    else:
                        logger.warning(f"源 Checkpoint 目录 {source_dir} 不存在或不是目录，跳过复制。")
                        steps_completed.append(f"{STEP_COPY_CKPT}: 跳过 (源不存在)")
                        # Consider if missing checkpoint-2 after training is an error
                        # raise PipelineStepError(f"必需的源 Checkpoint 目录 {source_dir} 不存在")
                else:
                    logger.info(f"{STEP_COPY_CKPT}: 跳过 (配置)")
                    steps_completed.append(f"{STEP_COPY_CKPT}: 跳过 (配置)")

        else:
            logger.info(f"{STEP_TRAIN}: 跳过 (配置)")
            steps_completed.append(f"{STEP_TRAIN}: 跳过 (配置)")
            logger.info(f"{STEP_COPY_CKPT}: 跳过 (训练未运行)")
            steps_completed.append(f"{STEP_COPY_CKPT}: 跳过 (训练未运行)")


        # 步骤 3: Start API Service
        if run_api:
            logger.info("-" * 10 + " 步骤 3: 启动 API 服务 " + "-" * 10)
            api_process = start_api_service_background()
            steps_completed.append(f"{STEP_API_START}: 成功")
        else:
            logger.info(f"{STEP_API_START}: 跳过 (配置)")
            steps_completed.append(f"{STEP_API_START}: 跳过 (配置)")


        # 步骤 4: Eval Model (依赖 API 服务)
        if run_eval:
            if not run_api:
                logger.info("-" * 10 + " 步骤 4: 模型评估 " + "-" * 10)
                logger.warning("--- 因 API 服务配置为不运行，跳过执行: weclone/eval/test_model.py ---")
                steps_completed.append(f"{STEP_EVAL}: 跳过 (API未配置运行)")
            elif api_process is None or api_process.poll() is not None: # 检查进程是否已退出
                 error_msg = "尝试运行评估，但 API 服务进程不存在或已退出。"
                 logger.error(error_msg)
                 raise PipelineStepError(error_msg)
            else:
                logger.info("-" * 10 + " 步骤 4: 模型评估 " + "-" * 10)
                # 在调用评估脚本时禁用 tqdm
                run_script(eval_script, timeout=9999, env={'TQDM_DISABLE': '1'})
                steps_completed.append(f"{STEP_EVAL}: 成功")
                stop_api_service(api_process) # 评估完成后停止API
                api_process = None # 标记为已停止
        else:
            logger.info(f"{STEP_EVAL}: 跳过 (配置)")
            steps_completed.append(f"{STEP_EVAL}: 跳过 (配置)")
            if api_process: # 如果API在运行但评估被跳过，也停止API
                logger.info("评估被跳过，停止 API 服务...")
                stop_api_service(api_process)
                api_process = None


        # 步骤 5: Start Web Demo (不依赖 API 服务)
        if run_web_demo:
            logger.info("-" * 10 + " 步骤 5: 启动 Web Demo " + "-" * 10)
            web_demo_process = start_web_demo_background()
            steps_completed.append(f"{STEP_WEB_DEMO}: 成功")
            logger.info("--- Web Demo 已启动，测试流程继续... ---")
        else:
            logger.info(f"{STEP_WEB_DEMO}: 跳过 (配置)")
            steps_completed.append(f"{STEP_WEB_DEMO}: 跳过 (配置)")

        # Pipeline 成功完成所有请求的步骤
        logger.info("="*20 + " Pipeline 执行摘要 " + "="*20)
        for step in steps_completed:
            logger.info(f"- {step}")
        logger.success("✅ 所有请求执行的 Pipeline 步骤均成功完成！")

        skipped_steps = [s for s in steps_completed if "跳过" in s]
        if skipped_steps:
            logger.warning("注意: 以下步骤被设置为跳过，如需执行请修改脚本顶部的 run_xxx 变量：")
            for skipped in skipped_steps:
                 logger.warning(f"  - {skipped.split(':')[0]}")


    except PipelineStepError as e:
        logger.error("="*20 + " Pipeline 执行失败 " + "="*20)

        failing_step = "未知步骤"
        error_details = str(e)
        cleaned_error_details = error_details # Store original/cleaned details for logging

        # Attempt 1: Check for explicit marker (e.g., from copy_checkpoint)
        marker_prefix = "###step_id:"
        marker_suffix = "###"
        marker_start = error_details.find(marker_prefix)
        if marker_start != -1:
            marker_end = error_details.find(marker_suffix, marker_start + len(marker_prefix))
            if marker_end != -1:
                step_id = error_details[marker_start + len(marker_prefix):marker_end]
                failing_step = step_identifiers.get(step_id, f"未知标记 ({step_id})")
                # Clean the marker from the displayed error details
                cleaned_error_details = error_details[:marker_start].strip()

        # Attempt 2: Check for known script paths in the error message if marker not found
        if failing_step == "未知步骤":
            found_script = False
            # Iterate through potential script paths stored as keys in step_identifiers
            for identifier, step_name in step_identifiers.items():
                # Check if the identifier looks like a path and is in the error message
                if isinstance(identifier, str) and ('/' in identifier or '\\\\' in identifier) and identifier in error_details:
                    failing_step = step_name
                    found_script = True
                    break # Found the most likely script

        # Attempt 3: Fallback based on last completed step (if still unknown)
        if failing_step == "未知步骤":
            if steps_completed:
                last_completed = steps_completed[-1].split(':')[0]
                try:
                    last_completed_index = step_order.index(last_completed)
                    if last_completed_index + 1 < len(step_order):
                        # Assume the next step in the defined order failed
                        failing_step = step_order[last_completed_index + 1] + " (推断)"
                    else:
                        failing_step = "Pipeline末尾或未知 (推断)" # Error after the last known step
                except ValueError:
                    # Last completed step name wasn't found in our defined order
                    failing_step = f"未知 (最后完成: {last_completed})"
            else:
                failing_step = "初始化期间" # No steps completed

        logger.error(f"错误发生在步骤: {failing_step}")
        logger.error(f"错误详情: {cleaned_error_details}") # Log the cleaned error details
        logger.info("--- 已完成步骤 ---")
        for step in steps_completed:
            logger.info(f"- {step}")
        logger.error("="*50)
        sys.exit(1) # 测试失败时退出码为 1

    finally:
        logger.info("--- Pipeline 结束，开始清理后台服务 ---")
        # 确保在 finally 块中总是尝试停止服务
        stop_web_demo(web_demo_process)
        stop_api_service(api_process) # 即使评估步骤停止了它，这里也尝试停止，无害
        logger.info("--- 后台服务清理完成 ---")

    # 如果 Pipeline 成功，确保退出码为 0
    sys.exit(0) 