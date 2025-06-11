import os
import sys

import commentjson

from .log import logger
from .tools import dict_to_argv


def load_config(arg_type: str):
    config_path = os.environ.get("WECLONE_CONFIG_PATH", "./settings.jsonc")
    logger.info(f"Loading configuration from: {config_path}")  # Add logging to see which file is loaded
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            s_config: dict = commentjson.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)  # Exit if config file is not found
    except Exception as e:
        logger.error(f"Error loading configuration file {config_path}: {e}")
        sys.exit(1)

    if arg_type == "cli_args":
        config = s_config["cli_args"]
    elif arg_type == "web_demo" or arg_type == "api_service":
        # infer_args和common_args求并集
        config = {**s_config["infer_args"], **s_config["common_args"]}
    elif arg_type == "train_pt":
        config = {**s_config["train_pt_args"], **s_config["common_args"]}
    elif arg_type == "train_sft":
        config = {**s_config["train_sft_args"], **s_config["common_args"]}
        if s_config["make_dataset_args"]["prompt_with_history"]:
            dataset_info_path = os.path.join(config["dataset_dir"], "dataset_info.json")
            dataset_info = commentjson.load(open(dataset_info_path, "r", encoding="utf-8"))[config["dataset"]]
            if dataset_info["columns"].get("history") is None:
                logger.warning(
                    f"{config['dataset']}数据集不包history字段，尝试使用wechat-sft-with-history数据集"
                )
                config["dataset"] = "wechat-sft-with-history"
        if "image" in s_config["make_dataset_args"]["include_type"]:
            if config["vision_api"].get("enable", False):
                config["dataset"] = "wechat-img-rec-sft"  # 图像识别类模型使用的数据集
            else:
                config["dataset"] = "wechat-mllm-sft"  # 多模态模型使用的数据集

    elif arg_type == "make_dataset":
        config = {**s_config["make_dataset_args"], **s_config["common_args"]}
        config["dataset"] = s_config["train_sft_args"]["dataset"]
        config["dataset_dir"] = s_config["train_sft_args"]["dataset_dir"]
        config["cutoff_len"] = s_config["train_sft_args"]["cutoff_len"]
        if "image" in config["include_type"]:
            if config["vision_api"].get("enable", False):
                config["dataset"] = "wechat-img-rec-sft"  # 图像识别类模型使用的数据集
            else:
                config["dataset"] = "wechat-mllm-sft"  # 多模态模型使用的数据集

    else:
        raise ValueError("暂不支持的参数类型")

    if "train" in arg_type:
        config["output_dir"] = config["adapter_name_or_path"]
        config.pop("adapter_name_or_path")
        config["do_train"] = True

    sys.argv += dict_to_argv(config)

    return config
