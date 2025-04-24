import json
import openai

from tqdm import tqdm
from typing import List, Dict

from weclone.utils.config import load_config

config = load_config("web_demo")

config = {
    "default_prompt": config["default_system"],
    "model": "gpt-3.5-turbo",
    "history_len": 15,
}

config = type("Config", (object,), config)()

openai.api_key = """sk-test"""
openai.api_base = "http://127.0.0.1:8005/v1"


def handler_text(content: str, history: List[Dict[str, str]], config):
    messages = [{"role": "system", "content": f"{config.default_prompt}"}]
    for item in history:
        messages.append(item)
    messages.append({"role": "user", "content": content})
    history.append({"role": "user", "content": content})
    try:
        response = openai.ChatCompletion.create(model=config.model, messages=messages, max_tokens=50)
    except openai.APIError as e:
        history.pop()
        return "AI接口出错,请重试\n" + str(e)

    resp = str(response.choices[0].message.content) # type: ignore
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
