from llamafactory.webui.interface import create_web_demo

from weclone.utils.config import load_config
from weclone.utils.secure_storage import is_encrypted_mode


def main():
    if is_encrypted_mode():
        raise RuntimeError(
            "加密模式不支持 webchat-demo。请用 weclone-cli server --inference 启动接口，"
            "再用 weclone-cli test-model 或接入的聊天机器人测试。"
        )
    load_config("web_demo")
    demo = create_web_demo()
    demo.queue()
    demo.launch(server_name="127.0.0.1", share=False, inbrowser=True)


if __name__ == "__main__":
    main()
