from weclone.utils.config import load_config


def create_inference_app():
    from llamafactory.api.app import create_app
    from llamafactory.chat import ChatModel

    config = load_config("api_service")
    chat_model = ChatModel(config.model_dump(mode="json"))
    return create_app(chat_model)
