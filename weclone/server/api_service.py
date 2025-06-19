import os

import uvicorn
from llamafactory.api.app import create_app
from llamafactory.chat import ChatModel

from weclone.utils.config import load_config


def main():
    config = load_config("api_service")
    chat_model = ChatModel(config.model_dump(mode="json"))
    app = create_app(chat_model)
    print("Visit http://localhost:{}/docs for API document.".format(os.environ.get("API_PORT", 8005)))
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", 8005)), workers=1)


if __name__ == "__main__":
    main()
