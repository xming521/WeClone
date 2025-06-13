import os
import sys
from typing import Any, Dict, cast

import commentjson
from omegaconf import OmegaConf
from pydantic import BaseModel

from .config_models import (
    WcConfig,
    WCInferConfig,
    WCMakeDatasetConfig,
    WCTrainSftConfig,
)
from .log import logger
from .tools import dict_to_argv


def load_base_config() -> WcConfig:
    """加载基础配置文件并创建WcConfig对象"""
    config_path = os.environ.get("WECLONE_CONFIG_PATH", "./settings.jsonc")
    logger.info(f"Loading configuration from: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            s_config_dict: Dict[str, Any] = commentjson.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration file {config_path}: {e}")
        sys.exit(1)

    # 使用 OmegaConf 解析配置，然后转换为 Pydantic 模型验证
    try:
        omega_config = OmegaConf.create(s_config_dict)
        config_dict_for_validation = OmegaConf.to_container(omega_config, resolve=True)
        if not isinstance(config_dict_for_validation, dict):
            raise TypeError(
                f"Configuration should be a dictionary, but got {type(config_dict_for_validation)}"
            )
        wc_config = WcConfig(**cast(Dict[str, Any], config_dict_for_validation))
    except Exception as e:
        logger.error(f"Error parsing configuration with OmegaConf and WcConfig: {e}")
        sys.exit(1)

    return wc_config


def create_config_by_arg_type(arg_type: str, wc_config: WcConfig) -> BaseModel:
    """根据参数类型创建对应的配置对象,添加可能用到的参数,添加的参数会在model_validator中删除"""
    if arg_type == "cli_args":
        return wc_config.cli_args

    common_config = wc_config.common_args.model_dump()

    if arg_type == "web_demo" or arg_type == "api_service":
        config_dict = {**common_config, **wc_config.infer_args.model_dump()}
        return WCInferConfig(**config_dict)

    elif arg_type == "vllm":
        return wc_config.vllm_args

    elif arg_type == "test_model":
        return wc_config.test_model_args

    elif arg_type == "train_sft":
        config_dict = {**common_config, **wc_config.train_sft_args.model_dump()}
        return WCTrainSftConfig(**config_dict)

    elif arg_type == "make_dataset":
        make_dataset_config = wc_config.make_dataset_args.model_dump()
        # ToDo 下面三个参数放到common里？
        train_sft_args = wc_config.train_sft_args
        extra_values = {
            "dataset": train_sft_args.dataset,
            "dataset_dir": train_sft_args.dataset_dir,
            "cutoff_len": train_sft_args.cutoff_len,
        }
        config_dict = {**common_config, **make_dataset_config, **extra_values}
        return WCMakeDatasetConfig(**config_dict)

    else:
        raise ValueError("暂不支持的参数类型")


def process_config_dict_and_argv(arg_type: str, config_pydantic: BaseModel) -> None:
    """处理配置字典并更新sys.argv"""
    config_dict = config_pydantic.model_dump(mode="json")

    sys.argv += dict_to_argv(config_dict)


def load_config(arg_type: str) -> BaseModel:
    """加载配置的主函数"""
    # 加载基础配置
    wc_config = load_base_config()

    # 根据类型创建配置对象
    config_pydantic = create_config_by_arg_type(arg_type, wc_config)

    # 处理配置字典和命令行参数
    process_config_dict_and_argv(arg_type, config_pydantic)

    return config_pydantic


if __name__ == "__main__":
    load_config("train_sft")
