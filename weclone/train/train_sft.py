import json
import os
from typing import cast

from llamafactory.extras.misc import get_current_device
from llamafactory.train.tuner import run_exp

from weclone.data.clean.strategies import LLMCleaningStrategy
from weclone.utils.config import load_config
from weclone.utils.config_models import WCMakeDatasetConfig, WCTrainSftConfig
from weclone.utils.log import logger


def main():
    train_config: WCTrainSftConfig = cast(WCTrainSftConfig, load_config(arg_type="train_sft"))
    dataset_config: WCMakeDatasetConfig = cast(WCMakeDatasetConfig, load_config(arg_type="make_dataset"))

    device = get_current_device()
    if device == "cpu":
        logger.warning("Please note you are using CPU for training, non-Mac devices may encounter issues")

    dataset_info_path = os.path.join(dataset_config.dataset_dir, "dataset_info.json")

    with open(dataset_info_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)
        data_path = os.path.join(
            dataset_config.dataset_dir, dataset_info.get(train_config.dataset, {}).get("file_name")
        )
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset file '{data_path}' does not exist, please check if make-dataset was executed"
            )

    if not dataset_config.clean_dataset.enable_clean:
        logger.info("Data cleaning is not enabled, will use the original dataset.")
    else:
        cleaner = LLMCleaningStrategy(make_dataset_config=dataset_config)
        train_config.dataset = cleaner.clean()

    formatted_config = json.dumps(train_config.model_dump(mode="json"), indent=4, ensure_ascii=False)
    logger.info(f"Fine-tuning configuration:\n{formatted_config}")

    run_exp(train_config.model_dump(mode="json"))


if __name__ == "__main__":
    main()
