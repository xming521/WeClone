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
        logger.warning("请注意你正在使用CPU训练，非Mac设备可能会出现问题")

    dataset_info_path = os.path.join(dataset_config.dataset_dir, "dataset_info.json")

    with open(dataset_info_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)
        data_path = os.path.join(
            dataset_config.dataset_dir, dataset_info.get(train_config.dataset, {}).get("file_name")
        )
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件 '{data_path}' 不存在，请检查是否执行了make-dataset")

    if not dataset_config.clean_dataset.enable_clean or "image" in dataset_config.include_type:
        logger.info("数据清洗未启用或包含图像，将使用原始数据集。")
    else:
        cleaner = LLMCleaningStrategy(make_dataset_config=dataset_config)
        train_config.dataset = cleaner.clean()

    formatted_config = json.dumps(train_config.model_dump(mode="json"), indent=4, ensure_ascii=False)
    logger.info(f"微调配置：\n{formatted_config}")

    run_exp(train_config.model_dump(mode="json"))


if __name__ == "__main__":
    main()
