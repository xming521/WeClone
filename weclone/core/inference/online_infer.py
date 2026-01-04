import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Union

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

from weclone.core.inference.offline_infer import extract_json_from_text
from weclone.utils.log import logger
from weclone.utils.retry import retry_openai_api

logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


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
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.prompt_with_system = prompt_with_system
        self.response_format = response_format

    @retry_openai_api(max_retries=200, base_delay=30.0, max_delay=180.0)
    def chat(
        self,
        prompt_text,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        stream: bool = False,
    ):
        messages: List[ChatCompletionMessageParam] = []
        if self.prompt_with_system:
            messages = prompt_text
        else:
            messages = [
                # {"role": "system", "content": self.default_system},
                {"role": "user", "content": prompt_text},
            ]

        params = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            # extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        }

        if self.response_format:
            params["response_format"] = {"type": self.response_format}

        response = self.client.chat.completions.create(**params)

        return response

    def chat_async(
        self,
        prompt_text: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        stream: bool = False,
    ) -> Future:
        """Submit a chat request to the thread pool for async processing"""
        return self.executor.submit(self.chat, prompt_text, temperature, max_tokens, top_p, stream)

    def chat_batch(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        stream: bool = False,
        callback: Optional[Callable[[int, Any], None]] = None,
        guided_decoding_class: Optional[type[BaseModel]] = None,
    ) -> Union[List[Union[ChatCompletion, Exception]], tuple[List[Optional[BaseModel]], List[int]]]:
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
            If enable_json_decode is False: List of ChatCompletion or Exception objects
            If enable_json_decode is True: Tuple of (parsed_results, failed_indices)
        """
        futures = []

        for i, prompt in enumerate(prompts):
            future = self.chat_async(prompt, temperature, max_tokens, top_p, stream)
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
            parsed_results: List[Optional[BaseModel]] = [None] * len(prompts)
            failed_indexs: List[int] = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_indexs.append(i)
                    logger.warning(f"Request at index {i} failed with exception: {result}")
                elif isinstance(result, ChatCompletion):
                    try:
                        content = result.choices[0].message.content
                        if content is None:
                            raise ValueError("Message content is None")
                        json_text = extract_json_from_text(content)
                        parsed_result = guided_decoding_class.model_validate_json(json_text)
                        parsed_results[i] = parsed_result
                    except Exception as e:
                        content = result.choices[0].message.content
                        log_text = (content[:100] + "...") if content else "None"
                        logger.warning(
                            f"Failed to parse JSON from result at index {i}: {log_text}, error: {e}"
                        )
                        failed_indexs.append(i)
                else:
                    logger.warning(f"Unexpected result type at index {i}: {type(result)}")
                    failed_indexs.append(i)

            return parsed_results, failed_indexs

        return results

    def close(self):
        """Clean up thread pool resources"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
