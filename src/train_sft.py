from llmtuner import run_exp
from template import template_register
from src.utils.config import load_config

config = load_config(arg_type='train_sft')

template_register()

run_exp(config)
