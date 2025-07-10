import random
import time
from functools import wraps
from typing import Callable, List, Optional

from weclone.utils.log import logger


def retry_on_http_error(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_on_status: Optional[List[int]] = None,
    retry_on_exceptions: Optional[List[type]] = None,
):
    """
    HTTP请求重试装饰器，专门处理429状态码和其他网络错误

    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        backoff_factor: 退避因子，每次重试延迟时间乘以此因子
        jitter: 是否添加随机抖动，避免雷群效应
        retry_on_status: 需要重试的HTTP状态码列表，默认包含429, 500, 502, 503, 504
        retry_on_exceptions: 需要重试的异常类型列表
    """
    if retry_on_status is None:
        retry_on_status = [429, 500, 502, 503, 504]

    if retry_on_exceptions is None:
        retry_on_exceptions = [ConnectionError, TimeoutError]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    # 检查是否是HTTP响应对象
                    if hasattr(result, "status_code"):
                        if result.status_code in retry_on_status:
                            if attempt < max_retries:
                                delay = _calculate_delay(
                                    attempt, base_delay, max_delay, backoff_factor, jitter
                                )
                                logger.warning(
                                    f"HTTP请求返回状态码 {result.status_code}，"
                                    f"第 {attempt + 1}/{max_retries + 1} 次尝试，"
                                    f"将在 {delay:.2f} 秒后重试..."
                                )
                                time.sleep(delay)
                                continue
                            else:
                                logger.error(
                                    f"HTTP请求在 {max_retries + 1} 次尝试后最终失败，状态码: {result.status_code}"
                                )
                                return result

                    return result

                except Exception as e:
                    should_retry_on_exception = any(
                        isinstance(e, exc_type) for exc_type in retry_on_exceptions
                    )

                    if should_retry_on_exception and attempt < max_retries:
                        delay = _calculate_delay(attempt, base_delay, max_delay, backoff_factor, jitter)
                        logger.warning(
                            f"请求异常: {type(e).__name__}: {e}，"
                            f"第 {attempt + 1}/{max_retries + 1} 次尝试，"
                            f"将在 {delay:.2f} 秒后重试..."
                        )
                        time.sleep(delay)
                        continue
                    elif should_retry_on_exception:
                        logger.error(f"请求在 {max_retries + 1} 次尝试后最终失败: {type(e).__name__}: {e}")
                        raise
                    else:
                        logger.error(f"未知错误，不进行重试: {type(e).__name__}: {e}")
                        raise

            return None  # 理论上不会执行到这里

        return wrapper

    return decorator


def retry_openai_api(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
):
    """
    专门用于OpenAI API调用的重试装饰器
    处理OpenAI特有的异常类型
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    # 检查是否是速率限制或临时错误
                    error_message = str(e).lower()
                    should_retry = (
                        "rate limit" in error_message
                        or "429" in error_message
                        or "too many requests" in error_message
                        or "server error" in error_message
                        or "timeout" in error_message
                        or "connection" in error_message
                    )

                    if should_retry and attempt < max_retries:
                        delay = _calculate_delay(attempt, base_delay, max_delay, backoff_factor, jitter)
                        logger.warning(
                            f"OpenAI API调用失败: {type(e).__name__}: {e}，"
                            f"第 {attempt + 1}/{max_retries + 1} 次尝试，"
                            f"将在 {delay:.2f} 秒后重试..."
                        )
                        time.sleep(delay)
                        continue
                    else:
                        if attempt >= max_retries:
                            logger.error(
                                f"OpenAI API调用在 {max_retries + 1} 次尝试后最终失败: {type(e).__name__}: {e}"
                            )
                        raise

            return None

        return wrapper

    return decorator


def _calculate_delay(
    attempt: int, base_delay: float, max_delay: float, backoff_factor: float, jitter: bool
) -> float:
    """计算重试延迟时间"""
    delay = base_delay * (backoff_factor**attempt)
    delay = min(delay, max_delay)

    if jitter:
        # 添加±20%的随机抖动
        jitter_range = delay * 0.2
        delay += random.uniform(-jitter_range, jitter_range)
        delay = max(0, delay)  # 确保延迟不为负数

    return delay


class RetryConfig:
    """重试配置类，用于统一管理重试参数"""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retry_on_status: Optional[List[int]] = None,
        retry_on_exceptions: Optional[List[type]] = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retry_on_status = retry_on_status or [429, 500, 502, 503, 504]
        self.retry_on_exceptions = retry_on_exceptions or [ConnectionError, TimeoutError]

    def apply_to_function(self, func: Callable) -> Callable:
        """将重试配置应用到函数上"""
        return retry_on_http_error(
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            backoff_factor=self.backoff_factor,
            jitter=self.jitter,
            retry_on_status=self.retry_on_status,
            retry_on_exceptions=self.retry_on_exceptions,
        )(func)


# 预定义的重试配置
AGGRESSIVE_RETRY = RetryConfig(
    max_retries=5,
    base_delay=0.5,
    max_delay=30.0,
    backoff_factor=1.5,
)

CONSERVATIVE_RETRY = RetryConfig(
    max_retries=2,
    base_delay=2.0,
    max_delay=10.0,
    backoff_factor=2.0,
)

API_RETRY = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0,
    retry_on_status=[429, 500, 502, 503, 504],
)
