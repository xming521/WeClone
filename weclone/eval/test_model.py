import json
from typing import List, cast  # 导入 cast

import openai
from openai import OpenAI  # 导入 OpenAI 类
from openai.types.chat import ChatCompletionMessageParam  # 导入消息参数类型
from tqdm import tqdm

from weclone.utils.config import load_config

config = load_config("web_demo")

config = {
    "default_prompt": config["default_system"],
    "model": "gpt-3.5-turbo",
    "history_len": 15,
}

config = type("Config", (object,), config)()

# 初始化 OpenAI 客户端
client = OpenAI(api_key="""sk-test""", base_url="http://127.0.0.1:8005/v1")


def handler_text(content: str, history: list, config):
    messages = [{"role": "system", "content": f"{config.default_prompt}"}]
    for item in history:
        messages.append(item)
    messages.append({"role": "user", "content": content})
    history.append({"role": "user", "content": content})
    try:
        # 使用新的 API 调用方式
        # 将 messages 转换为正确的类型
        typed_messages = cast(List[ChatCompletionMessageParam], messages)
        response = client.chat.completions.create(
            model=config.model,
            messages=typed_messages,  # 传递转换后的列表
            max_tokens=50,
        )
    except openai.APIError as e:
        history.pop()
        return "AI接口出错,请重试\n" + str(e)

    resp = str(response.choices[0].message.content)  # type: ignore
    resp = resp.replace("\n ", "")
    history.append({"role": "assistant", "content": resp})
    return resp


def main():
    test_list = json.loads(open("dataset/test_data.json", "r", encoding="utf-8").read())["questions"]
    res = []
    for questions in tqdm(test_list, desc=" Testing..."):
        history = []
        for q in questions:
            handler_text(q, history=history, config=config)
        res.append(history)

    res_file = open("test_result-my.txt", "w")
    for r in res:
        for i in r:
            res_file.write(i["content"] + "\n")
        res_file.write("\n")


if __name__ == "__main__":
    main()
