import os
import sys
from llamafactory.train.tuner import run_exp
from llamafactory.extras.misc import get_current_device
from weclone.utils.config import load_config
from weclone.utils.log import logger

# todo 添加一个test的环境变量 放一个test的配置文件

def main():
    config = load_config(arg_type="train_sft")

    device = get_current_device()
    if device == "cpu":
        logger.warning("请注意你正在使用CPU训练，非Mac设备可能会出现问题")

    sft_json_path = os.path.join(config["dataset_dir"], "sft-my.json")
    if not os.path.exists(sft_json_path):
        logger.error(f"错误：文件 '{sft_json_path}' 不存在，请确保数据处理步骤已正确生成该文件。")
        sys.exit(1)

    run_exp(config)


if __name__ == "__main__":
    main()
