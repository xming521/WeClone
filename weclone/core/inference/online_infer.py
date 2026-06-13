import logging
import os
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Union
from urllib.parse import urlparse

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

from weclone.core.inference.offline_infer import extract_json_from_text
from weclone.utils.log import logger

logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

OPENROUTER_PROXY_URL_ENV = "WECLONE_OPENROUTER_PROXY_URL"
DEFAULT_OPENROUTER_PROXY_URL = "http://127.0.0.1:7899"
API_TIMEOUT_SECONDS = 45
OPENAI_API_MAX_RETRIES = 100
OPENAI_API_BASE_DELAY = 2.0
OPENAI_API_MAX_DELAY = 60.0
OPENAI_API_BACKOFF_FACTOR = 2.0
OPENAI_API_JITTER = True
INTERRUPTED_ERROR_MESSAGE = "OnlineLLM stopped by interrupt"


def _openrouter_proxy_url(base_url: str) -> Optional[str]:
    parsed_url = urlparse(base_url)
    if (parsed_url.hostname or "").lower() != "openrouter.ai":
        return None

    proxy_url = os.environ.get(OPENROUTER_PROXY_URL_ENV, DEFAULT_OPENROUTER_PROXY_URL)
    return proxy_url or None


def _calculate_retry_delay(
    attempt: int,
    base_delay: float = OPENAI_API_BASE_DELAY,
    max_delay: float = OPENAI_API_MAX_DELAY,
    backoff_factor: float = OPENAI_API_BACKOFF_FACTOR,
    jitter: bool = OPENAI_API_JITTER,
) -> float:
    delay = base_delay * (backoff_factor**attempt)
    delay = min(delay, max_delay)

    if jitter:
        jitter_range = delay * 0.2
        delay += random.uniform(-jitter_range, jitter_range)
        delay = max(0, delay)

    return delay


class OnlineLLM:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        default_system: Optional[str] = None,
        max_workers: int = 10,
        prompt_with_system: bool = False,
        response_format: str = "json_object",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.default_system = default_system
        self.max_workers = max_workers
        proxy_url = _openrouter_proxy_url(self.base_url)
        http_client = httpx.Client(proxy=proxy_url, timeout=API_TIMEOUT_SECONDS) if proxy_url else None
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=0,
            timeout=API_TIMEOUT_SECONDS,
            http_client=http_client,
        )
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._stop_event = threading.Event()
        self.prompt_with_system = prompt_with_system
        self.response_format = response_format

    def chat(
        self,
        messages,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
        top_p: Optional[float] = None,
        stream: bool = False,
        extra_body: Optional[dict[str, Any]] = None,
    ):
        # messages: List[ChatCompletionMessageParam] = []
        # messages = [
        #     {"role": "system", "content": self.default_system},
        #     {"role": "user", "content": messages},
        # ]

        params = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if extra_body:
            params["extra_body"] = extra_body

        if self.response_format:
            params["response_format"] = {"type": self.response_format}

        for attempt in range(OPENAI_API_MAX_RETRIES + 1):
            if self._stop_event.is_set():
                raise RuntimeError(INTERRUPTED_ERROR_MESSAGE)

            try:
                return self.client.chat.completions.create(**params)
            except Exception as e:
                if self._stop_event.is_set():
                    raise RuntimeError(INTERRUPTED_ERROR_MESSAGE) from e

                if attempt < OPENAI_API_MAX_RETRIES:
                    delay = _calculate_retry_delay(attempt)
                    logger.warning(
                        f"OpenAI API调用失败: {type(e).__name__}: {e}，"
                        f"第 {attempt + 1}/{OPENAI_API_MAX_RETRIES + 1} 次尝试，"
                        f"将在 {delay:.2f} 秒后重试..."
                    )
                    if self._stop_event.wait(delay):
                        raise RuntimeError(INTERRUPTED_ERROR_MESSAGE) from e
                    continue

                logger.error(
                    f"OpenAI API调用在 {OPENAI_API_MAX_RETRIES + 1} 次尝试后最终失败: "
                    f"{type(e).__name__}: {e}"
                )
                raise

    def chat_async(
        self,
        prompt_text: str,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
        top_p: Optional[float] = None,
        stream: bool = False,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> Future:
        """Submit a chat request to the thread pool for async processing"""
        return self.executor.submit(
            self.chat,
            prompt_text,
            temperature,
            max_tokens,
            top_p,
            stream,
            extra_body,
        )

    def chat_batch(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
        top_p: Optional[float] = None,
        stream: bool = False,
        extra_body: Optional[dict[str, Any]] = None,
        callback: Optional[Callable[[int, Any], None]] = None,
        guided_decoding_class: Optional[type[BaseModel]] = None,
    ) -> Union[List[Union[ChatCompletion, Exception]], List[Union[BaseModel, str]]]:
        """Process multiple chat requests concurrently using thread pool

        Args:
            prompts: List of prompt strings
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            stream: Whether to stream the response
            callback: Optional callback function called for each result
            guided_decoding_class: Pydantic model class for JSON validation

        Returns:
            If guided_decoding_class is None: List of ChatCompletion or Exception objects
            If guided_decoding_class is provided: List of parsed BaseModel objects or error message strings
        """
        futures = []

        for i, prompt in enumerate(prompts):
            future = self.chat_async(prompt, temperature, max_tokens, top_p, stream, extra_body)
            futures.append((i, future))

        results: List[Union[Any, Exception]] = [None] * len(prompts)

        for i, future in futures:
            try:
                result = future.result()
                results[i] = result
                if callback:
                    callback(i, result)
            except Exception as e:
                results[i] = e
                if callback:
                    callback(i, e)

        if guided_decoding_class:
            parsed_results: List[Union[BaseModel, str]] = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_msg = f"Exception: {type(result).__name__}: {str(result)}"
                    parsed_results.append(error_msg)
                    logger.warning(f"Request at index {i} failed with exception: {result}")
                elif isinstance(result, ChatCompletion):
                    finish_reason = result.choices[0].finish_reason
                    if finish_reason != "stop":
                        error_msg = f"finish_reason: {finish_reason}"
                        parsed_results.append(error_msg)
                        logger.warning(f"Request at index {i} finished with reason: {finish_reason}")
                    else:
                        try:
                            content = result.choices[0].message.content
                            if content is None or content.strip() == "":
                                raise ValueError("Message content is None")
                            json_text = extract_json_from_text(content)
                            parsed_result = guided_decoding_class.model_validate_json(json_text)
                            parsed_results.append(parsed_result)
                        except Exception as e:
                            error_msg = f"model_validate_json: {type(e).__name__}: {str(e)}"
                            parsed_results.append(error_msg)
                            content = result.choices[0].message.content
                            log_text = (content[:100] + "...") if content else "None"
                            logger.warning(
                                f"Failed to parse JSON from result at index {i}: {log_text}, error: {e}"
                            )
                else:
                    error_msg = f"Unexpected result type: {type(result).__name__}"
                    parsed_results.append(error_msg)
                    logger.warning(f"Unexpected result type at index {i}: {type(result)}")

            return parsed_results

        return results

    def close(self, wait: bool = True, cancel_futures: bool = False):
        """Clean up thread pool resources"""
        self._stop_event.set()
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        if hasattr(self, "client"):
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        force_stop = exc_type is not None and issubclass(exc_type, KeyboardInterrupt)
        self.close(wait=not force_stop, cancel_futures=force_stop)
