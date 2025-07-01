import json
from typing import List, cast  # 导入 cast

import openai
from openai import OpenAI  # 导入 OpenAI 类
from openai.types.chat import ChatCompletionMessageParam  # 导入消息参数类型
from tqdm import tqdm

from weclone.utils.config import load_config
from weclone.utils.config_models import TestModelArgs, WCInferConfig

infer_config = cast(WCInferConfig, load_config("web_demo"))
test_config = cast(TestModelArgs, load_config("test_model"))

completion_config = {
    "default_prompt": infer_config.default_system,
    "model": "gpt-3.5-turbo",
    "history_len": 15,
}

completion_config = type("Config", (object,), completion_config)()

client = OpenAI(api_key="""sk-test""", base_url="http://127.0.0.1:8005/v1")


def handler_text(content: str, history: list, config):
    messages = [{"role": "system", "content": f"{config.default_prompt}"}]
    for item in history:
        messages.append(item)
    messages.append({"role": "user", "content": content})
    history.append({"role": "user", "content": content})
    try:
        typed_messages = cast(List[ChatCompletionMessageParam], messages)
        response = client.chat.completions.create(
            model=config.model,
            messages=typed_messages,
            max_tokens=50,
        )
    except openai.APIError as e:
        history.pop()
        return "AI interface error, please try again\n" + str(e)

    resp = str(response.choices[0].message.content)  # type: ignore
    resp = resp.replace("\n ", "")
    history.append({"role": "assistant", "content": resp})
    return resp


def main():
    test_list = json.loads(open(test_config.test_data_path, "r", encoding="utf-8").read())["questions"]
    res = []
    for questions in tqdm(test_list, desc=" Testing..."):
        history = []
        for q in questions:
            handler_text(q, history=history, config=completion_config)
        res.append(history)

    res_file = open("test_result-my.txt", "w")
    for r in res:
        for i in r:
            res_file.write(i["content"] + "\n")
        res_file.write("\n")


if __name__ == "__main__":
    main()
