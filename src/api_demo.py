import os

import uvicorn

from llmtuner import ChatModel, create_app

# python LLaMA-Factory/src/web_demo.py \


config = {
    "model_name_or_path": "./chatglm3-6b",
    "adapter_name_or_path": "./model_output",
    "template": "chatglm3",
    "finetuning_type": "lora",
    "repetition_penalty": 1.2,
    "temperature": 0.5,
    "max_length": 50,
    "top_p": 0.65
}


def main():
    chat_model = ChatModel(config)
    app = create_app(chat_model)
    print("Visit http://localhost:{}/docs for API document.".format(os.environ.get("API_PORT", 8000)))
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", 8000)), workers=1)


if __name__ == "__main__":
    main()
