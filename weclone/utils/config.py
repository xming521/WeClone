import os
import commentjson
import sys

from .log import logger
from .tools import dict_to_argv


def load_config(arg_type: str):
    with open("./settings.json", "r", encoding="utf-8") as f:
        s_config: dict = commentjson.load(f)
    if arg_type == "web_demo" or arg_type == "api_service":
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
                logger.warning(f"{config['dataset']}数据集不包history字段，尝试使用wechat-sft-with-history数据集")
                s_config["make_dataset_args"]["dataset"] = "wechat-sft-with-history"

    elif arg_type == "make_dataset":
        config = {**s_config["make_dataset_args"], **s_config["common_args"]}
    else:
        raise ValueError("暂不支持的参数类型")

    if "train" in arg_type:
        config["output_dir"] = config["adapter_name_or_path"]
        config.pop("adapter_name_or_path")
        config["do_train"] = True

    sys.argv += dict_to_argv(config)

    return config
