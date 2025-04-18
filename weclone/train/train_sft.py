from llamafactory.train.tuner import run_exp
from llamafactory.extras.misc import get_current_device
from weclone.train.template import template_register
from weclone.utils.config import load_config
from weclone.utils.log import logger

config = load_config(arg_type="train_sft")

device = get_current_device()
if device == "cpu":
    logger.warning("请注意你正在使用CPU训练，非Mac设备可能会出现问题")

template_register()

if __name__ == "__main__":
    run_exp(config)
