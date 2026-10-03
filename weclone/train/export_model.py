from llamafactory.hparams import get_infer_args
from llamafactory.train.tuner import export_model

from weclone.train.security import model_runtime, parsed_cli_arguments
from weclone.utils.secure_storage import is_encrypted_mode


def main():
    if not is_encrypted_mode():
        export_model()
        return
    config = parsed_cli_arguments(get_infer_args)
    if config.get("export_hub_model_id"):
        raise ValueError("Encrypted model export requires a local export_dir")
    with model_runtime(config) as runtime:
        export_model(runtime)


if __name__ == "__main__":
    main()
