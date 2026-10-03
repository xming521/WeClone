import sys
from typing import cast  # 导入 cast

import click
import httpx
from tqdm import tqdm

from weclone.core.inference import OpenAICompatibleClient, RetryPolicy
from weclone.utils.config import load_config
from weclone.utils.config_models import TestModelArgs, WCInferConfig
from weclone.utils.log import logger
from weclone.utils.secure_storage import lock, read_json, unlock, write_text

infer_config = cast(WCInferConfig, load_config("web_demo"))
test_config = cast(TestModelArgs, load_config("test_model"))

completion_config = {
    "default_prompt": infer_config.default_system,
    "model": "gpt-3.5-turbo",
    "history_len": 15,
}

completion_config = type("Config", (object,), completion_config)()

client = OpenAICompatibleClient(
    api_key="""sk-test""",
    base_url="http://127.0.0.1:8005/v1",
    retry_policy=RetryPolicy(max_retries=2, base_delay=0.5, max_delay=8),
)


def _check_api_server() -> None:
    """Verify that the API server is running before starting tests.

    Exits with an error message if the server is not reachable.
    """
    try:
        response = httpx.get(
            str(client.base_url).rstrip("/") + "/models",
            headers=client.client.default_headers,
            timeout=45,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        logger.error(
            f"Cannot connect to the API server at {client.base_url}. "
            "Please start the server first by running: weclone-cli server --inference --port 8005"
        )
        sys.exit(1)


def _authenticate_server() -> None:
    password = click.prompt("输入网页与数据读写共用密码", hide_input=True)
    try:
        unlock(password)
        server_url = str(client.base_url).rstrip("/").removesuffix("/v1")
        response = httpx.post(
            server_url + "/api/auth/login",
            json={"password": password},
            headers={"X-WeClone-Request": "1"},
            timeout=45,
        )
        if response.status_code != 200:
            raise click.ClickException(f"网页鉴权失败（HTTP {response.status_code}），未发送评估聊天内容。")
        cookie = "; ".join(f"{name}={value}" for name, value in response.cookies.items())
        if not cookie:
            raise click.ClickException("网页登录未返回会话，无法执行评估。")
        client.client = client.client.with_options(
            default_headers={"Cookie": cookie, "X-WeClone-Request": "1"}
        )
    except httpx.HTTPError:
        raise click.ClickException(
            "无法连接网页服务，请先启动 weclone-cli server --inference --port 8005。"
        ) from None
    finally:
        del password


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
    try:
        _authenticate_server()
        _check_api_server()
        test_list = read_json(test_config.test_data_path, import_plaintext=True)["questions"]
        res = []
        for questions in tqdm(test_list, desc=" Testing..."):
            history = []
            for q in questions:
                response = handler_text(q, history=history, config=completion_config)
                if not history or history[-1].get("role") != "assistant":
                    raise click.ClickException(response)
            res.append(history)

        output = "test_result-my.txt"
        write_text(output, "\n\n".join("\n".join(item["content"] for item in result) for result in res))
    finally:
        client.close()
        lock()


if __name__ == "__main__":
    main()
