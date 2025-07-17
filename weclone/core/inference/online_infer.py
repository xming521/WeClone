from openai import OpenAI

from weclone.utils.retry import retry_openai_api


class OnlineLLM:
    def __init__(self, api_key: str, base_url: str, model_name: str, default_system: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.default_system = default_system
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)

    # TODO 需要做一个线程池进行并发
    @retry_openai_api(max_retries=200, base_delay=30.0, max_delay=180.0)
    def chat(
        self,
        prompt_text,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        stream: bool = False,
        enable_thinking: bool = False,
    ):
        messages = [
            {"role": "system", "content": self.default_system},
            {"role": "user", "content": prompt_text},
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        return response
