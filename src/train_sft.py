from llamafactory.train.tuner import run_exp
from llamafactory.extras.misc import get_current_device
from template import template_register
from utils.config import load_config
from utils.log import logger

config = load_config(arg_type="train_sft")

device = get_current_device()
if device == "cpu":
    logger.warning("请注意你正在使用CPU训练，非Mac设备可能会出现问题")


template_register()

run_exp(config)
