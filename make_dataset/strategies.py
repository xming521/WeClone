from dataclasses import dataclass
from typing import Protocol
from .qa_generator import ChatMessage

class ConversationStrategy(Protocol):
    def is_same_conversation(self, msg1: ChatMessage, msg2: ChatMessage) -> bool:
        """判断两条消息是否属于同一个对话"""
        
        pass



@dataclass
class TimeWindowStrategy(ConversationStrategy):
    """基于时间窗口的判断策略"""
    time_window: int  # 时间窗口（分钟）
    

    def is_same_conversation(self, msg1: ChatMessage, msg2: ChatMessage) -> bool:
        time_diff = abs((msg2.timestamp - msg1.timestamp))
        return time_diff <= self.time_window


@dataclass
class LagerModelStrategy(ConversationStrategy):
    """基于用户连续性的判断策略"""
    def is_same_conversation(self, msg1: ChatMessage, msg2: ChatMessage) -> bool:
        return msg1.user_id == msg2.user_id


@dataclass
class CompositeStrategy(ConversationStrategy):
    """组合多个策略的复合策略"""
    strategies: list[ConversationStrategy]
    require_all: bool = True  # True表示所有策略都满足，False表示任一策略满足即可

    def is_same_conversation(self, msg1: ChatMessage, msg2: ChatMessage) -> bool:
        results = [s.is_same_conversation(msg1, msg2) for s in self.strategies]
        return all(results) if self.require_all else any(results) 