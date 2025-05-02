from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict
# from ..models import ChatMessage # 如果需要操作特定模型，取消注释并调整


@dataclass
class CleaningStrategy(ABC):
    """数据清洗策略的抽象基类"""

    config: Dict

    @abstractmethod
    def clean(self, data: Any) -> Any:
        """
        执行数据清洗操作。

        Args:
            data: 需要清洗的数据。

        Returns:
            清洗后的数据。
        """
        pass


@dataclass
class LLMCleaningStrategy(CleaningStrategy):
    """使用大模型进行数据清洗的策略"""

    # 这里可以添加LLM相关的配置，例如模型名称、API密钥等
    # model_name: str = "your_llm_model"

    def clean(self, data: Any) -> Any:
        """
        使用大模型清洗数据。
        具体的实现需要根据您选择的LLM API和清洗任务来定。
        """
        # 此处为调用LLM进行清洗的逻辑占位符
        print(f"使用LLM清洗数据: {data}")
        # 假设LLM返回了清洗后的数据
        cleaned_data = f"LLM cleaned: {data}"  # 示例返回值
        return cleaned_data
