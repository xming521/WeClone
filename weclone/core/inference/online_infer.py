from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Union

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from weclone.utils.retry import retry_openai_api


class OnlineLLM:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        default_system: Optional[str] = None,
        max_workers: int = 10,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.default_system = default_system
        self.max_workers = max_workers
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    @retry_openai_api(max_retries=200, base_delay=30.0, max_delay=180.0)
    def chat(
        self,
        prompt_text,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        stream: bool = False,
    ):
        messages: List[ChatCompletionMessageParam] = [
            # {"role": "system", "content": self.default_system},
            {"role": "user", "content": prompt_text},
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_format={"type": "json_object"},
        )

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
    ) -> List[Union[Any, Exception]]:
        """Process multiple chat requests concurrently using thread pool"""
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

        return results

    def close(self):
        """Clean up thread pool resources"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
