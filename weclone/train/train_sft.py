import json
import os
from typing import cast

from llamafactory.extras.misc import get_current_device
from llamafactory.train.tuner import run_exp

from weclone.data.clean.strategies import LLMCleaningStrategy
from weclone.train.security import run_training
from weclone.utils.config import load_config
from weclone.utils.config_models import WCMakeDatasetConfig, WCTrainSftConfig
from weclone.utils.log import logger
from weclone.utils.secure_storage import file_exists, is_encrypted_mode, read_json


def main():
    train_config: WCTrainSftConfig = cast(WCTrainSftConfig, load_config(arg_type="train_sft"))
    dataset_config: WCMakeDatasetConfig = cast(WCMakeDatasetConfig, load_config(arg_type="make_dataset"))

    device = get_current_device()
    if device == "cpu":
        logger.warning("Please note you are using CPU for training, non-Mac devices may encounter issues")

    dataset_info_path = os.path.join(dataset_config.dataset_dir, "dataset_info.json")

    dataset_info = read_json(dataset_info_path, import_plaintext=True)
    dataset_file = dataset_info.get(train_config.dataset, {}).get("file_name")
    if not dataset_file:
        raise ValueError(f"Dataset '{train_config.dataset}' is not defined in dataset_info.json")
    data_path = os.path.join(dataset_config.dataset_dir, dataset_file)
    if not file_exists(data_path):
        raise FileNotFoundError(
            f"Dataset file '{data_path}' does not exist, please check if make-dataset was executed"
        )

    if not dataset_config.clean_dataset.enable_clean:
        logger.info("Data cleaning is not enabled, will use the original dataset.")
    else:
        cleaner = LLMCleaningStrategy(make_dataset_config=dataset_config)
        train_config.dataset = cleaner.clean()

    if not is_encrypted_mode():
        formatted_config = json.dumps(train_config.model_dump(mode="json"), indent=4, ensure_ascii=False)
        logger.info(f"Fine-tuning configuration:\n{formatted_config}")

    # Build config dict and remove nested 'quantization' key (its fields are already flattened at top level)
    config_dict = train_config.model_dump(mode="json", exclude_none=True)
    config_dict.pop("quantization", None)

    run_training(config_dict, run_exp)


if __name__ == "__main__":
    main()
