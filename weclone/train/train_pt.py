from llamafactory.train.tuner import run_exp

from weclone.utils.config import load_config

config = load_config("train_pt")
run_exp(config)
