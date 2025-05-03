from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict
from langchain_core.prompts import PromptTemplate
from weclone.prompts.clean_data import CLEAN_PROMPT


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

    def clean(self, data: Any) -> Any:
        prompt_template = PromptTemplate.from_template(CLEAN_PROMPT)

        prompt_template.invoke({"topic": "cats"})
        # prompt_template.
        # return cleaned_data
