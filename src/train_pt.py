import os

from llmtuner import run_exp
train_args = {
    "stage": "pt",
    "do_train": True,
    "model_name_or_path": './chatglm3-6b',
    "dataset": "wechat-pt",
    "dataset_dir": './data/res_csv/pt',
    "finetuning_type": "lora",  # "lora", "freeze", "full"]
    "lora_target": "query_key_value",  # all  query_key_value
    "lora_rank": 2,
    "lora_dropout": 0.1,
    "output_dir": "model_output",
    "overwrite_cache": True,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_steps": 1000,
    "learning_rate": 1e-3,
    "num_train_epochs": 30.0,
    "plot_loss": True,
    "fp16": True
}


run_exp(train_args)
