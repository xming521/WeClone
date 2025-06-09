from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from .models import ChatMessage


@dataclass
class ConversationStrategy(ABC):
    """对话策略的抽象基类"""

    is_single_chat: bool

    @abstractmethod
    def is_same_conversation(self, history_msg: List[ChatMessage], current_msg: ChatMessage) -> bool:
        """判断两条消息是否属于同一个对话"""
        pass


@dataclass
class TimeWindowStrategy(ConversationStrategy):
    """基于时间窗口的判断策略"""

    time_window: int  # 时间窗口（分钟）

    def is_same_conversation(self, history_msg: List[ChatMessage], current_msg: ChatMessage) -> bool:
        time_diff = abs((current_msg.CreateTime - history_msg[-1].CreateTime)).total_seconds()
        return time_diff <= self.time_window


@dataclass
class LLMStrategy(ConversationStrategy):
    """基于大模型判断策略"""

    def is_same_conversation(self, history_msg: List[ChatMessage], current_msg: ChatMessage) -> bool:
        # 修复user_id错误，使用talker字段代替user_id
        return current_msg.talker == history_msg[-1].talker if history_msg else False


@dataclass
class CompositeStrategy(ConversationStrategy):
    """组合多个策略的复合策略"""

    strategies: List[ConversationStrategy]
    require_all: bool = True  # True表示所有策略都满足，False表示任一策略满足即可

    def is_same_conversation(self, history_msg: List[ChatMessage], current_msg: ChatMessage) -> bool:
        results = [s.is_same_conversation(history_msg, current_msg) for s in self.strategies]
        return all(results) if self.require_all else any(results)
