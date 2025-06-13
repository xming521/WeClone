from llamafactory.webui.interface import create_web_demo

from weclone.utils.config import load_config


def main():
    load_config("web_demo")
    demo = create_web_demo()
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=True, inbrowser=True)


if __name__ == "__main__":
    main()
