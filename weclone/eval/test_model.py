import json
import sys
from typing import cast  # 导入 cast

import httpx
from weclone.core.inference import OpenAICompatibleClient, RetryPolicy
from tqdm import tqdm

from weclone.utils.config import load_config
from weclone.utils.config_models import TestModelArgs, WCInferConfig
from weclone.utils.log import logger

infer_config = cast(WCInferConfig, load_config("web_demo"))
test_config = cast(TestModelArgs, load_config("test_model"))

completion_config = {
    "default_prompt": infer_config.default_system,
    "model": "gpt-3.5-turbo",
    "history_len": 15,
}

completion_config = type("Config", (object,), completion_config)()

client = OpenAICompatibleClient(api_key="""sk-test""", base_url="http://127.0.0.1:8005/v1", retry_policy=RetryPolicy(max_retries=2, base_delay=0.5, max_delay=8))


def _check_api_server() -> None:
    """Verify that the API server is running before starting tests.

    Exits with an error message if the server is not reachable.
    """
    try:
        response = httpx.get(str(client.base_url).rstrip("/") + "/models", headers={"Authorization": f"Bearer {client.api_key}"}, timeout=45)
        response.raise_for_status()
    except httpx.HTTPError:
        logger.error(
            f"Cannot connect to the API server at {client.base_url}. "
            "Please start the server first by running: weclone-cli server"
        )
        sys.exit(1)


def handler_text(content: str, history: list, config):
    messages = [{"role": "system", "content": f"{config.default_prompt}"}]
    for item in history:
        messages.append(item)
    messages.append({"role": "user", "content": content})
    history.append({"role": "user", "content": content})
    response = client.chat(messages, model=config.model, max_tokens=50, timeout=600)
    if not response.ok:
        history.pop()
        return "AI interface error, please try again\n" + str(response.error)

    resp = str(response.text)  # type: ignore
    resp = resp.replace("\n ", "")
    history.append({"role": "assistant", "content": resp})
    return resp


def main():
    _check_api_server()
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
