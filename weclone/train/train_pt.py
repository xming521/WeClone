import json
from pathlib import Path
from typing import Any, cast

from llamafactory.extras.misc import get_current_device
from llamafactory.train.tuner import run_exp

from weclone.utils.config import load_config
from weclone.utils.config_models import WCTrainPtConfig
from weclone.utils.log import logger


def _resolve_dataset_path(dataset_dir: str, dataset_name: str) -> Path:
    dataset_info_path = Path(dataset_dir) / "dataset_info.json"
    if not dataset_info_path.exists():
        raise FileNotFoundError(f"Dataset info file does not exist: {dataset_info_path}")

    with dataset_info_path.open("r", encoding="utf-8") as f:
        dataset_info: dict[str, Any] = json.load(f)

    dataset_entry = dataset_info.get(dataset_name)
    if dataset_entry is None:
        raise ValueError(f"Dataset '{dataset_name}' is not defined in {dataset_info_path}")

    if dataset_entry.get("formatting") == "sharegpt":
        raise ValueError(
            f"Dataset '{dataset_name}' is a ShareGPT dataset. "
            "LlamaFactory pre-training requires Alpaca-style data with columns.prompt mapped to text."
        )

    prompt_column = (dataset_entry.get("columns") or {}).get("prompt")
    if prompt_column is None:
        raise ValueError(f"Dataset '{dataset_name}' must define columns.prompt for pre-training.")

    dataset_file_name = dataset_entry.get("file_name")
    if not dataset_file_name:
        raise ValueError(f"Dataset '{dataset_name}' must define file_name in {dataset_info_path}")

    data_path = Path(dataset_file_name)
    if not data_path.is_absolute():
        data_path = Path(dataset_dir) / data_path

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file '{data_path}' does not exist.")

    return data_path


def main():
    train_config = cast(WCTrainPtConfig, load_config(arg_type="train_pt"))

    if train_config.stage != "pt":
        raise ValueError(f"train-pt requires stage='pt', got stage={train_config.stage!r}")

    device = get_current_device()
    if device == "cpu":
        logger.warning("Please note you are using CPU for training, non-Mac devices may encounter issues")

    data_path = _resolve_dataset_path(train_config.dataset_dir, train_config.dataset)
    logger.info(f"Using pre-training dataset: {data_path}")

    formatted_config = json.dumps(train_config.model_dump(mode="json"), indent=4, ensure_ascii=False)
    logger.info(f"Continued pre-training configuration:\n{formatted_config}")

    config_dict = train_config.model_dump(mode="json", exclude_none=True)
    config_dict.pop("quantization", None)

    run_exp(config_dict)


if __name__ == "__main__":
    main()
