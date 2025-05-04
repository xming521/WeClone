import pytest
from unittest.mock import patch, MagicMock, call
from langchain_core.prompts import PromptTemplate
from datetime import datetime
import pandas as pd # 导入 pandas

# 确保可以正确导入被测试的模块和依赖项
# 可能需要根据你的项目结构调整导入路径
try:
    from weclone.data.clean.strategies import LLMCleaningStrategy
    from weclone.data.models import QaPair
    from weclone.prompts.clean_data import CLEAN_PROMPT
except ImportError:
    # 如果直接运行脚本时找不到模块，尝试添加项目根目录到 sys.path
    import sys
    import os
    # 获取当前脚本文件所在的目录 (tests/data/clean)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取 tests 目录
    tests_dir = os.path.dirname(os.path.dirname(current_dir))
    # 获取项目根目录 (weclone 的父目录)
    project_root = os.path.dirname(tests_dir)
    sys.path.insert(0, project_root)
    from weclone.data.clean.strategies import LLMCleaningStrategy
    from weclone.data.models import QaPair
    from weclone.prompts.clean_data import CLEAN_PROMPT



@pytest.fixture
def sample_qa_pairs():
    """提供一些测试用的 QaPair 数据"""
    # now = datetime.now() # 不再需要 datetime
    return [
        QaPair(id=1, instruction="问题1", output="答案1", system="", history=[], time=pd.Timestamp.now(), score=0), # 使用 pd.Timestamp
        QaPair(id=2, instruction="问题2", output="答案2", system="", history=[], time=pd.Timestamp.now(), score=0), # 使用 pd.Timestamp
    ]

@pytest.fixture
def mock_make_dataset_config():
    """提供模拟的 make_dataset_config"""
    return {
        "model_name_or_path": "mock_model",
        "template": "mock_template",
        # 可以根据需要添加其他配置
    }

@patch("weclone.data.clean.strategies.infer") # 模拟 infer 函数
def test_llm_cleaning_strategy_clean(mock_infer, sample_qa_pairs, mock_make_dataset_config):
    """测试 LLMCleaningStrategy.clean 方法"""
    # 1. 准备
    print("--- 开始测试 test_llm_cleaning_strategy_clean ---")
    strategy = LLMCleaningStrategy(make_dataset_config=mock_make_dataset_config)
    prompt_template = PromptTemplate.from_template(CLEAN_PROMPT)

    # 预期 infer 函数的输入
    expected_inputs = []
    for qa in sample_qa_pairs:
        expected_inputs.append(prompt_template.invoke({"id": qa.id, "Q": qa.instruction, "A": qa.output}))
    print(f"预期 infer 输入: {expected_inputs}")

    # 设置模拟 infer 函数的返回值
    mock_cleaned_outputs = ["cleaned_output_1", "cleaned_output_2"]
    mock_infer.return_value = mock_cleaned_outputs
    print(f"设置 mock infer 返回值: {mock_cleaned_outputs}")

    # 2. 执行
    print("调用 strategy.clean...")
    # 注意：原始的 clean 方法没有 return 语句。如果需要测试返回值，
    # 需要在 weclone/data/clean/strategies.py 中取消注释 'return cleaned_data'
    cleaned_data = strategy.clean(sample_qa_pairs)
    # strategy.clean(sample_qa_pairs) # 暂时只调用，不获取返回值
    print(f"获取的 cleaned_data: {cleaned_data}") # 如果有返回值，取消注释此行

    # 3. 断言
    print("执行断言...")
    # 验证 infer 函数是否以正确的参数被调用
    try:
        mock_infer.assert_called_once_with(
            expected_inputs,
            mock_make_dataset_config["model_name_or_path"],
            template=mock_make_dataset_config["template"],
            temperature=0,
        )
        print("infer 函数调用断言成功！")
    except AssertionError as e:
        print(f"infer 函数调用断言失败: {e}")
        raise # 重新抛出异常，以便 pytest 能捕获

    # 验证 clean 方法的返回值（基于假设）
    # 如果原始 clean 方法确实没有 return，可以移除这个断言或者修改 clean 方法添加 return
    assert cleaned_data == mock_cleaned_outputs
    print("返回值断言成功！")

    print("--- 测试 test_llm_cleaning_strategy_clean 结束 ---")


if __name__ == "__main__":
    print("直接运行测试脚本进行调试...")

    # 手动准备依赖项 (代替 pytest fixtures)
    # now_main = datetime.now() # 不再需要 datetime
    qa_pairs = [
        QaPair(id=101, instruction="调试问题1", output="调试答案1", system="", history=[], time=pd.Timestamp.now(), score=0), # 使用 pd.Timestamp
        QaPair(id=102, instruction="调试问题2", output="调试答案2", system="", history=[], time=pd.Timestamp.now(), score=0), # 使用 pd.Timestamp
    ]
    config = {
        "model_name_or_path": "debug_model",
        "template": "debug_template",
    }

    # 方案1：直接调用被测代码逻辑 (更简单)
    print("\n--- 方案1：直接调用被测代码逻辑 ---")
    try:
        from weclone.data.clean.strategies import infer # 需要导入 infer
    except ImportError:
         # 处理导入错误的代码已在文件顶部
         from weclone.data.clean.strategies import infer

    with patch("weclone.data.clean.strategies.infer") as mock_infer_main:
        strategy = LLMCleaningStrategy(make_dataset_config=config)
        prompt_template = PromptTemplate.from_template(CLEAN_PROMPT)
        inputs_main = []
        for qa in qa_pairs:
            inputs_main.append(prompt_template.invoke({"id": qa.id, "Q": qa.instruction, "A": qa.output}))
        
        mock_return = ["debug_cleaned_1", "debug_cleaned_2"]
        mock_infer_main.return_value = mock_return
        print(f"设置 main 中的 mock infer 返回值: {mock_return}")

        print("在 main 中调用 strategy.clean...")
        cleaned_result_main = strategy.clean(qa_pairs) # 如果 clean 有返回值
        # strategy.clean(qa_pairs) # 如果 clean 没有返回值
        print(f"Main 中获取的 cleaned_result: {cleaned_result_main}") # 如果有返回值

        print("在 main 中进行断言...")
        try:
            mock_infer_main.assert_called_once_with(
                inputs_main,
                config["model_name_or_path"],
                template=config["template"],
                temperature=0,
            )
            print("Main 中的 infer 函数调用断言成功！")
            if cleaned_result_main == mock_return: # 如果有返回值
                print("Main 中的返回值断言成功！")
            else:
                print(f"Main 中的返回值断言失败: 预期 {mock_return}, 得到 {cleaned_result_main}")

        except AssertionError as e:
            print(f"Main 中的 infer 函数调用断言失败: {e}")


    # # 方案2：手动调用测试函数（稍微复杂，需要手动创建 mock）
    # print("\\n--- 方案2：手动调用测试函数 ---")
    # # 创建一个 mock 对象手动传递
    # mock_infer_manual = MagicMock()
    # # 为手动创建的 mock 设置返回值 (如果需要)
    # mock_return_manual = ["debug_cleaned_1_manual", "debug_cleaned_2_manual"]
    # mock_infer_manual.return_value = mock_return_manual
    # print(f"设置 manual mock infer 返回值: {mock_return_manual}")

    # try:
    #     print("手动调用 test_llm_cleaning_strategy_clean...")
    #     # 注意：直接调用被 @patch 装饰的函数可能导致 TypeError
    #     # 因为装饰器期望由测试运行器（如 pytest）注入 mock 对象
    #     test_llm_cleaning_strategy_clean(mock_infer_manual, qa_pairs, config)
    #     print("手动调用测试函数完成。请检查上面的打印输出。")
    #     # 检查手动传入的 mock 是否被调用 (可能不会，因为 @patch 可能覆盖了它)
    #     print("检查 manual mock 调用次数:", mock_infer_manual.call_count)
    # except TypeError as e:
    #     print(f"\\n手动调用测试函数时捕获到 TypeError: {e}")
    #     print("这通常发生在直接运行脚本时，@patch 装饰器未能正确处理 mock 注入。")
    #     print("建议使用方案1（'with patch(...)' 上下文管理器）进行调试，因为它在 __main__ 块中更可靠。")

    print("\n调试脚本运行结束。") 