from llamafactory.webui.interface import create_web_demo

from weclone.utils.config import load_config
from weclone.utils.secure_storage import is_encrypted_mode


def main():
    if is_encrypted_mode():
        raise RuntimeError("安全模式请使用 weclone-cli server --inference，通过统一网页登录查看聊天数据。")
    load_config("web_demo")
    demo = create_web_demo()
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=True, inbrowser=True)


if __name__ == "__main__":
    main()
