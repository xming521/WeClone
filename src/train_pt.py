from llmtuner import run_exp
from utils.config import load_config

config = load_config('train_pt')
run_exp(config)
