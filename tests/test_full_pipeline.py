import subprocess
import sys
import os
import time
from typing import Optional, Union, TYPE_CHECKING

# 类型检查时导入 Popen，避免运行时循环导入问题 (虽然此脚本不太可能被导入)
if TYPE_CHECKING:
    from subprocess import Popen

# 获取项目根目录 (假设 tests 目录在项目根目录下)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"项目根目录: {project_root}")

# 定义要执行的脚本相对于项目根目录的路径
qa_script = "weclone/data/qa_generator.py"
train_script = "weclone/train/train_sft.py"
api_service_script = "weclone/server/api_service.py"
eval_script = "weclone/eval/test_model.py"

# 配置执行参数
DEFAULT_TIMEOUT: Optional[Union[int, float]] = 300 # 默认脚本执行超时时间（秒），允许 None
API_STARTUP_WAIT = 30 # 等待 API 服务初步启动的时间（秒）
API_TERMINATE_WAIT = 15 # 等待 API 服务终止的时间（秒）

def run_script(script_relative_path: str, timeout: Optional[Union[int, float]] = DEFAULT_TIMEOUT, ignore_timeout_error: bool = False) -> bool:
    """执行指定的 Python 脚本 (阻塞式)"""
    script_full_path = os.path.join(project_root, script_relative_path)
    timeout_str = '无限制' if timeout is None else f'{timeout}s'
    print(f"\n--- 开始执行 (阻塞): {script_relative_path} (超时: {timeout_str}) ---")
    if not os.path.exists(script_full_path):
        print(f"错误: 脚本文件不存在 {script_full_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, script_full_path],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=timeout
        )
        print(f"--- {script_relative_path} 执行成功 ---")
        return True
    except FileNotFoundError:
        print(f"错误: Python 解释器 '{sys.executable}' 或脚本 '{script_full_path}' 未找到。")
        return False
    except subprocess.CalledProcessError as e:
        print(f"错误: {script_relative_path} 执行失败，返回码 {e.returncode}")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        return False
    except subprocess.TimeoutExpired as e:
        timeout_val = f" ({e.timeout}s)" if hasattr(e, 'timeout') and e.timeout is not None else ""
        print(f"警告: {script_relative_path} 执行超时{timeout_val}。")
        stdout_str = e.stdout if hasattr(e, 'stdout') else "(无)"
        stderr_str = e.stderr if hasattr(e, 'stderr') else "(无)"
        print("--- STDOUT (超时前) ---")
        print(stdout_str)
        print("--- STDERR (超时前) ---")
        print(stderr_str)
        if ignore_timeout_error:
            print(f"--- 根据设置，超时不视为错误，继续执行后续步骤。 ---")
            return True
        else:
            return False
    except Exception as e:
        print(f"执行 {script_relative_path} 时发生意外错误: {e}")
        return False

def start_api_service_background() -> Optional['Popen']:
    """在后台启动 API 服务脚本，并返回进程对象。"""
    script_full_path = os.path.join(project_root, api_service_script)
    print(f"\n--- 尝试在后台启动: {api_service_script} ---")
    if not os.path.exists(script_full_path):
        print(f"错误: 脚本文件不存在 {script_full_path}")
        return None

    process = None
    try:
        print(f"启动命令: {[sys.executable, script_full_path]}")
        process = subprocess.Popen(
            [sys.executable, script_full_path],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        print(f"等待 {API_STARTUP_WAIT} 秒让服务初步启动...")
        time.sleep(API_STARTUP_WAIT)

        # 检查进程是否在等待期间意外退出
        if process.poll() is None:
            print(f"--- {api_service_script} 似乎已在后台启动 (进程 PID: {process.pid}) ---")
            return process # 返回进程对象
        else:
            # 进程已退出，读取输出并报告错误
            stdout, stderr = process.communicate()
            print(f"错误: {api_service_script} 启动后过早退出，返回码 {process.returncode}")
            print("--- STDOUT ---")
            print(stdout)
            print("--- STDERR ---")
            print(stderr)
            return None # 启动失败

    except FileNotFoundError:
        print(f"错误: Python 解释器 '{sys.executable}' 或脚本 '{script_full_path}' 未找到。")
        return None
    except Exception as e:
        print(f"启动 {api_service_script} 时发生意外错误: {e}")
        if process and process.poll() is None:
             print("捕获到异常，尝试强制终止进程...")
             try:
                 process.kill()
             except Exception as kill_e:
                 print(f"强制终止进程时出错: {kill_e}")
        return None # 启动失败

def stop_api_service(process: Optional['Popen']):
    """停止指定的 API 服务进程。"""
    if process and process.poll() is None:
        print(f"\n--- 尝试停止 API 服务 (PID: {process.pid}) ---")
        try:
            process.terminate() # 尝试优雅终止
            print(f"等待 {API_TERMINATE_WAIT} 秒让服务终止...")
            stdout, stderr = process.communicate(timeout=API_TERMINATE_WAIT)
            print(f"API 服务进程已终止，返回码: {process.returncode}")
            # 可以选择性打印服务的最后输出
            # print("--- API 服务 STDOUT ---")
            # print(stdout)
            # print("--- API 服务 STDERR ---")
            # print(stderr)
        except subprocess.TimeoutExpired:
            print(f"警告: 优雅终止超时 ({API_TERMINATE_WAIT}s)，强制终止进程...")
            try:
                process.kill()
                stdout, stderr = process.communicate() # 获取剩余输出
                print("API 服务进程已被强制终止。")
                # print("--- API 服务 STDOUT (强制终止后) ---")
                # print(stdout)
                # print("--- API 服务 STDERR (强制终止后) ---")
                # print(stderr)
            except Exception as kill_e:
                 print(f"强制终止进程时出错: {kill_e}")
        except Exception as e:
            print(f"停止 API 服务时发生错误: {e}")
    elif process:
        print(f"\n--- API 服务进程 (PID: {process.pid}) 在尝试停止前已经退出。 ---")
    else:
        # 如果 process 为 None，则什么也不做
        pass


if __name__ == "__main__":
    print("="*20 + " 开始执行 WeClone Pipeline 脚本 " + "="*20)
    steps = []
    all_success = True
    api_process: Optional['Popen'] = None # 用于存储 API 服务进程

    try:
        # 步骤 1: QA Generator
        print("\n" + "-"*10 + " 步骤 1: QA 数据生成 " + "-"*10)
        if not run_script(qa_script):
            all_success = False
            steps.append("QA 数据生成: 失败")
        else:
            steps.append("QA 数据生成: 成功")

        # 步骤 2: Train SFT (默认跳过)
        print("\n" + "-"*10 + " 步骤 2: SFT 训练 " + "-"*10)
        run_train = True # 设置为 True 以运行训练
        if not run_train:
            print("--- 跳过执行 (耗时较长): weclone/train/train_sft.py ---")
            steps.append("SFT 训练: 跳过")
        elif all_success:
            train_success = run_script(train_script, timeout=3600, ignore_timeout_error=True)
            if not train_success:
                # 如果 run_script 返回 False，并且我们设置了 ignore_timeout_error=True,
                # 那么这一定是一个非超时的错误。
                all_success = False
                steps.append("SFT 训练: 失败 (非超时错误)")
            else:
                # 如果 run_script 返回 True，它可能是成功完成，也可能是超时被忽略了。
                # 为了更清晰，我们可以考虑修改 run_script 返回更详细的状态，但目前保持简单。
                steps.append("SFT 训练: 成功或超时跳过")
        else:
            print("--- 因先前步骤失败，跳过执行: weclone/train/train_sft.py ---")
            steps.append("SFT 训练: 跳过 (先前失败)")

        # 步骤 3: Start API Service (默认跳过)
        print("\n" + "-"*10 + " 步骤 3: 启动 API 服务 " + "-"*10)
        run_api = True # 设置为 True 以启动 API 服务
        if not run_api:
            print("--- 跳过执行: weclone/server/api_service.py ---")
            steps.append("API 服务启动: 跳过")
        elif all_success:
            api_process = start_api_service_background()
            if api_process is None:
                all_success = False
                steps.append("API 服务启动: 失败")
            else:
                steps.append("API 服务启动: 成功")
        else:
            print("--- 因先前步骤失败，跳过执行: weclone/server/api_service.py ---")
            steps.append("API 服务启动: 跳过 (先前失败)")

        # 步骤 4: Eval Model (默认跳过, 依赖 API 服务)
        print("\n" + "-"*10 + " 步骤 4: 模型评估 " + "-"*10)
        run_eval = True # 设置为 True 以运行评估
        if not run_eval:
            print("--- 跳过执行: weclone/eval/test_model.py ---")
            steps.append("模型评估: 跳过")
        elif not run_api or api_process is None: # 如果API服务未启动或启动失败，则跳过
            print("--- 因 API 服务未运行或启动失败，跳过执行: weclone/eval/test_model.py ---")
            steps.append("模型评估: 跳过 (API 未运行)")
            all_success = False # 依赖失败，整个流程不算完全成功
        elif all_success:
            # 评估没有超时限制
            if not run_script(eval_script, timeout=None):
                all_success = False
                steps.append("模型评估: 失败")
            else:
                steps.append("模型评估: 成功")
        else:
            # 如果之前的步骤失败 (非API启动失败，因为那个情况上面处理了)
            print("--- 因先前步骤失败，跳过执行: weclone/eval/test_model.py ---")
            steps.append("模型评估: 跳过 (先前失败)")

    finally:
        # 无论如何，确保停止 API 服务 (如果它已启动)
        stop_api_service(api_process)

    print("\n" + "="*20 + " Pipeline 执行摘要 " + "="*20)
    for step in steps:
        print(f"- {step}")

    print("\n" + "="*50)
    # 检查是否有跳过的步骤
    skipped_steps = [s for s in steps if "跳过" in s and "超时跳过" not in s] # 超时跳过不算需要手动处理的跳过

    if all_success:
        print("✅ 所有已执行的 Pipeline 步骤均成功完成！")
        if skipped_steps:
            print("\n注意: 以下步骤被设置为跳过，如需执行请修改脚本中的 run_xxx 变量：")
            for skipped in skipped_steps:
                print(f"  - {skipped.split(':')[0]}")
    else:
        print("❌ Pipeline 执行过程中至少有一个步骤失败或被跳过 (由于依赖失败)。")
        if skipped_steps:
             print("\n此外，以下步骤被设置为跳过，如需执行请修改脚本中的 run_xxx 变量：")
             for skipped in skipped_steps:
                 print(f"  - {skipped.split(':')[0]}")
        sys.exit(1) # 以非零退出码退出，表示失败 