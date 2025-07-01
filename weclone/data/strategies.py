from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from .models import ChatMessage


@dataclass
class ConversationStrategy(ABC):
    """Abstract base class for conversation strategies"""

    is_single_chat: bool

    @abstractmethod
    def is_same_conversation(self, history_msg: List[ChatMessage], current_msg: ChatMessage) -> bool:
        """Determine if two messages belong to the same conversation"""
        pass


@dataclass
class TimeWindowStrategy(ConversationStrategy):
    """Time window based judgment strategy"""

    time_window: int  # Time window in minutes

    def is_same_conversation(self, history_msg: List[ChatMessage], current_msg: ChatMessage) -> bool:
        time_diff = abs((current_msg.CreateTime - history_msg[-1].CreateTime)).total_seconds()
        return time_diff <= self.time_window


@dataclass
class LLMStrategy(ConversationStrategy):
    """LLM based judgment strategy"""

    def is_same_conversation(self, history_msg: List[ChatMessage], current_msg: ChatMessage) -> bool:
        # TODO: Implement LLM-based conversation detection logic
        return False


@dataclass
class CompositeStrategy(ConversationStrategy):
    """Composite strategy that combines multiple strategies"""

    strategies: List[ConversationStrategy]
    require_all: bool = True

    def is_same_conversation(self, history_msg: List[ChatMessage], current_msg: ChatMessage) -> bool:
        # TODO: Implement composite strategy logic
        return False
