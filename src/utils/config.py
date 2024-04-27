import json
from .utils import dict_to_argv
import sys


def load_config(arg_type: str):
    with open('./settings.json', 'r') as f:
        config: dict = json.load(f)
    if arg_type == 'web_demo' or arg_type == 'api_service':
        # infer_args和common_args求并集
        config = {**config['infer_args'], **config['common_args']}
    elif arg_type == 'train_pt':
        config = {**config['train_pt_args'], **config['common_args']}
    elif arg_type == 'train_sft':
        config = {**config['train_sft_args'], **config['common_args']}
    else:
        raise ValueError('暂不支持的类型')

    if 'train' in arg_type:
        config['output_dir'] = config['adapter_name_or_path']
        config.pop('adapter_name_or_path')
        config['do_train'] = True

    sys.argv += dict_to_argv(config)

    return config
