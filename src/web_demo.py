from llmtuner import create_web_demo
from template import template_register
from utils.config import load_config

config = load_config('web_demo')

template_register()

def main():
    demo = create_web_demo()
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=True, inbrowser=True)


if __name__ == "__main__":
    main()
