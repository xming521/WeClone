from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc
from llamafactory.hparams import get_infer_args

from weclone.train.security import model_runtime, parsed_cli_arguments
from weclone.utils.secure_storage import is_encrypted_mode


def _chat_loop(chat_model):
    try:
        import platform

        if platform.system() != "Windows":
            import readline  # noqa: F401
    except ImportError:
        print("Install `readline` for a better experience.")

    messages = []
    print(
        "Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application."
    )

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            messages = []
            torch_gc()
            print("History has been removed.")
            continue

        messages.append({"role": "user", "content": query})
        print("Assistant: ", end="", flush=True)

        response = ""
        for new_text in chat_model.stream_chat(messages):
            print(new_text, end="", flush=True)
            response += new_text
        print()
        messages.append({"role": "assistant", "content": response})


def main():
    if not is_encrypted_mode():
        _chat_loop(ChatModel())
        return
    with model_runtime(parsed_cli_arguments(get_infer_args)) as runtime:
        _chat_loop(ChatModel(runtime))


if __name__ == "__main__":
    main()
