import json
import time
import requests
from openai import OpenAI

class OnlineLLM:
    def __init__(self, api_key: str, base_url: str,model_name: str,default_system: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.default_system = default_system
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )


    def chat(self,prompt_text, 
            temperature: float = 0.7, 
            max_tokens: int = 1024, 
            top_p: float = 0.95, 
            stream: bool = False,
            enable_thinking: bool = False):
        messages = [
            {"role": "system", "content": self.default_system},
            {"role": "user", "content": prompt_text},
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=stream,
            temperature = temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            # enable_thinking=enable_thinking   适配Qwen3动态开启推理  

        )

        return response

