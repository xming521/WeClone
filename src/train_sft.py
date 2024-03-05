from llmtuner import run_exp
from template import template_register

train_args = {
    "stage": "sft",
    "do_train": True,
    "model_name_or_path": './chatglm3-6b', # 本地下载好的模型
    "output_dir": "model_output",  # 保存模型的文件夹
    "dataset": "wechat-sft",
    "template": "chatglm3-weclone",
    "dataset_dir": './data/res_csv/sft',
    "finetuning_type": "lora",  # "lora", "freeze", "full"
    "lora_target": "query_key_value",  # all  query_key_value
    "lora_rank": 4,
    "lora_dropout": 0.5,
    "overwrite_cache": True,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,  # 可以增大以应对数据集噪声过多
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_steps": 150,
    "learning_rate": 1e-4,
    "num_train_epochs": 3.0,
    "plot_loss": True,
    "fp16": True,
}


template_register()

run_exp(train_args)
