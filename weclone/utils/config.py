import os
import sys
from typing import Any, Dict, cast

import pyjson5
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
    """Load base configuration file and create WcConfig object"""
    config_path = os.environ.get("WECLONE_CONFIG_PATH", "./settings.jsonc")
    logger.info(f"Loading configuration from: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            s_config_dict: Dict[str, Any] = pyjson5.loads(f.read())
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration file {config_path}: {e}")
        sys.exit(1)

    # Use OmegaConf to parse configuration, then convert to Pydantic model for validation
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
    """Create corresponding configuration object based on argument type, merge common_config"""
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
        common_config["include_type"] = wc_config.make_dataset_args.include_type
        config_dict = {**common_config, **wc_config.train_sft_args.model_dump()}
        return WCTrainSftConfig(**config_dict)

    elif arg_type == "make_dataset":
        make_dataset_config = wc_config.make_dataset_args.model_dump()
        # TODO: Should the following three parameters be moved to common?
        train_sft_args = wc_config.train_sft_args
        extra_values = {
            "dataset": train_sft_args.dataset,
            "dataset_dir": train_sft_args.dataset_dir,
            "cutoff_len": train_sft_args.cutoff_len,
        }
        config_dict = {**common_config, **make_dataset_config, **extra_values}
        return WCMakeDatasetConfig(**config_dict)

    else:
        raise ValueError("Unsupported argument type")


def process_config_dict_and_argv(arg_type: str, config_pydantic: BaseModel) -> None:
    """Process configuration dictionary and update sys.argv"""
    config_dict = config_pydantic.model_dump(mode="json")

    sys.argv += dict_to_argv(config_dict)


def load_config(arg_type: str) -> BaseModel:
    """Main function for loading configuration"""
    # Load base configuration
    wc_config = load_base_config()

    config_pydantic = create_config_by_arg_type(arg_type, wc_config)

    process_config_dict_and_argv(arg_type, config_pydantic)

    return config_pydantic


if __name__ == "__main__":
    load_config("train_sft")
