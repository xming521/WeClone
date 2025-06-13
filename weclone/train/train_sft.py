import json
import os
import sys
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

    cleaner = LLMCleaningStrategy(make_dataset_config=dataset_config)
    final_dataset_name = cleaner.clean()

    if train_config.dataset != final_dataset_name:
        logger.info(
            f"根据清洗结果，将训练数据集从 '{train_config.dataset}' 动态更新为 '{final_dataset_name}'。"
        )
        train_config.dataset = final_dataset_name

    dataset_info_path = os.path.join(train_config.dataset_dir, "dataset_info.json")
    try:
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
        target_file_name = dataset_info.get(final_dataset_name, {}).get("file_name")
        if not target_file_name:
            raise FileNotFoundError(
                f"在 dataset_info.json 中未找到数据集 '{final_dataset_name}' 的 file_name 配置。"
            )
        final_data_path = os.path.join(train_config.dataset_dir, target_file_name)
        if not os.path.exists(final_data_path):
            raise FileNotFoundError(f"最终要使用的SFT数据文件 '{final_data_path}' 不存在。")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"校验最终数据集时出错: {e}")
        sys.exit(1)

    formatted_config = json.dumps(train_config.model_dump(mode="json"), indent=4, ensure_ascii=False)
    logger.info(f"微调配置：\n{formatted_config}")

    run_exp(train_config.model_dump(mode="json"))


if __name__ == "__main__":
    main()
