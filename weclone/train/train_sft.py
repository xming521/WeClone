import json
import os
import sys
from typing import cast

from llamafactory.extras.misc import get_current_device
from llamafactory.train.tuner import run_exp

from weclone.data.clean.strategies import LLMCleaningStrategy
from weclone.utils.config_models import WCMakeDatasetConfig, WCTrainSftConfig
from weclone.utils.configV2 import load_config
from weclone.utils.log import logger


def main():
    train_config: WCTrainSftConfig = cast(WCTrainSftConfig, load_config(arg_type="train_sft"))
    dataset_config: WCMakeDatasetConfig = cast(WCMakeDatasetConfig, load_config(arg_type="make_dataset"))

    device = get_current_device()
    if device == "cpu":
        logger.warning("请注意你正在使用CPU训练，非Mac设备可能会出现问题")

    cleaner = LLMCleaningStrategy(make_dataset_config=dataset_config.model_dump(mode="json"))
    cleaned_data_path = cleaner.clean()

    if not os.path.exists(cleaned_data_path):
        logger.error(f"错误：文件 '{cleaned_data_path}' 不存在，请确保数据处理步骤已正确生成该文件。")
        sys.exit(1)

    formatted_config = json.dumps(train_config.model_dump(mode="json"), indent=4, ensure_ascii=False)
    logger.info(f"微调配置：\n{formatted_config}")

    run_exp(train_config.model_dump(mode="json"))


if __name__ == "__main__":
    main()
