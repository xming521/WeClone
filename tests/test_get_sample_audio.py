import os
import subprocess
import sys
import pytest

# 获取 weclone-audio/src 目录的绝对路径
# 这假设 tests 目录和 weclone-audio 在同一个父目录下
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'weclone-audio', 'src'))
SCRIPT_PATH = os.path.join(SRC_DIR, 'get_sample_audio.py')

# --- 测试配置 ---
# 请将下面的路径替换为你的测试数据库文件的实际路径
# 最好放在 tests/data 目录下，并使用相对路径
TEST_DB_PATH = r"D:\projects\python projects\WeClone-data\wxdump_work\wxid_d6wwiru2zsmo22\merge_all.db"# <--- 修改这里
# 请将下面的 ID 替换为测试数据库中一个有效的音频消息的 MsgSvrID
TEST_MSG_SVR_ID = "3269716813078873653" # <--- 修改这里
# ----------------

@pytest.fixture(scope="module")
def setup_test_environment():
    """确保测试所需的文件和目录存在"""
    if not os.path.exists(TEST_DB_PATH):
        pytest.fail(f"测试数据库文件未找到: {TEST_DB_PATH}。请提供一个有效的测试数据库。")
    if not os.path.exists(SCRIPT_PATH):
         pytest.fail(f"待测试的脚本未找到: {SCRIPT_PATH}")
    # 可以添加其他设置，例如创建测试数据目录

def test_audio_extraction(tmp_path, setup_test_environment):
    """
    测试 get_sample_audio.py 是否能成功提取音频并保存为 wav 文件。
    """
    output_filename = "test_output.wav"
    output_path = tmp_path / output_filename # 使用 pytest 的 tmp_path fixture 创建临时输出路径

    # 构建命令行参数
    cmd = [
        sys.executable, # 使用当前的 Python 解释器
        SCRIPT_PATH,
        "--db-path", TEST_DB_PATH,
        "--MsgSvrID", TEST_MSG_SVR_ID,
        "--save-path", str(output_path),
        "--rate", "24000" # 可以根据需要调整
    ]

    # 运行脚本
    # 注意：脚本中的 'key' 可能需要根据实际情况调整，或者修改脚本以允许通过参数传递 key
    # 目前脚本中硬编码了 key="test1"
    result = subprocess.run(cmd, capture_output=True, text=True, check=False) # check=False 允许我们检查返回码

    # 打印输出以便调试 (如果测试失败)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # 断言脚本成功运行
    assert result.returncode == 0, f"脚本执行失败，错误信息: {result.stderr}"

    # 断言输出文件已创建
    assert output_path.exists(), f"输出文件 {output_path} 未被创建"

    # (可选) 断言文件大小大于 0
    assert output_path.stat().st_size > 0, f"输出文件 {output_path} 为空"

    # (可选) 更复杂的检查，例如使用 wave 库检查文件头或内容
    # import wave
    # try:
    #     with wave.open(str(output_path), 'rb') as wf:
    #         assert wf.getnchannels() == 1 # 假设是单声道
    #         assert wf.getframerate() == 24000 # 检查采样率
    # except wave.Error as e:
    #     pytest.fail(f"无法读取输出的 WAV 文件: {e}") 

def main_debug():
    """用于直接运行和调试的主要函数"""
    print("--- 开始调试运行 ---")

    # 检查基本环境
    if not os.path.exists(TEST_DB_PATH):
        print(f"错误: 测试数据库文件未找到: {TEST_DB_PATH}")
        return
    if not os.path.exists(SCRIPT_PATH):
        print(f"错误: 待测试的脚本未找到: {SCRIPT_PATH}")
        return
    if TEST_MSG_SVR_ID == "YOUR_TEST_MSG_SVR_ID":
         print(f"警告: TEST_MSG_SVR_ID 似乎未配置 ({TEST_MSG_SVR_ID})")
         # 可以选择在这里 return 或继续执行

    # 定义调试输出路径
    debug_output_dir = os.path.join(os.path.dirname(__file__), "debug_output")
    os.makedirs(debug_output_dir, exist_ok=True) # 创建输出目录（如果不存在）
    debug_output_path = os.path.join(debug_output_dir, "debug_sample.wav")

    print(f"脚本路径: {SCRIPT_PATH}")
    print(f"数据库路径: {TEST_DB_PATH}")
    print(f"消息 ID: {TEST_MSG_SVR_ID}")
    print(f"输出路径: {debug_output_path}")

    # 构建命令行参数
    cmd = [
        sys.executable,
        SCRIPT_PATH,
        "--db-path", TEST_DB_PATH,
        "--MsgSvrID", TEST_MSG_SVR_ID,
        "--save-path", debug_output_path,
        "--rate", "24000"
    ]

    print(f"执行命令: {' '.join(cmd)}")

    # 运行脚本
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30) # 添加超时
        print("\\n--- 脚本执行结果 ---")
        print("返回码:", result.returncode)
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)

        # 检查结果
        if result.returncode == 0:
            print("\\n--- 结果检查 ---")
            if os.path.exists(debug_output_path):
                print(f"[成功] 输出文件已创建: {debug_output_path}")
                if os.path.getsize(debug_output_path) > 0:
                    print(f"[成功] 输出文件大小 > 0 ({os.path.getsize(debug_output_path)} bytes)")
                else:
                    print(f"[失败] 输出文件为空: {debug_output_path}")
            else:
                print(f"[失败] 输出文件未找到: {debug_output_path}")
        else:
            print("\\n[失败] 脚本执行失败。")

    except subprocess.TimeoutExpired:
        print("\\n[失败] 脚本执行超时。")
    except Exception as e:
        print(f"\\n[失败] 执行命令时发生异常: {e}")

    print("\\n--- 调试运行结束 ---")


if __name__ == "__main__":
    # 确保在直接运行时正确设置了测试数据路径
    # 注意：这里仍然使用文件顶部的 TEST_DB_PATH 和 TEST_MSG_SVR_ID
    # 请确保它们已经被修改为有效值！
    if TEST_DB_PATH == "tests/data/your_test_db.sqlite" or TEST_MSG_SVR_ID == "YOUR_TEST_MSG_SVR_ID":
        print("*"*40)
        print("警告：请先在脚本顶部修改 TEST_DB_PATH 和 TEST_MSG_SVR_ID 为有效的测试值！")
        print("*"*40)
        # sys.exit(1) # 可以取消注释以强制退出，如果未配置

    main_debug() 