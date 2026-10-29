简体中文｜[English](../en/finetune.md)

# 微调分身

用聊天记录微调开源模型，得到延续你说话风格的数字分身。这条路线需要 GPU 和[微调依赖](installation.md#微调依赖)。

> [!IMPORTANT]
> - 微调效果很大程度取决于模型大小以及聊天数据的数量和质量。7B 模型效果一般，14B 及以上的模型效果更好。
> - WeClone 仍在快速迭代，当前效果不代表最终效果。

## 硬件要求

默认使用 Qwen2.5-7B-Instruct，以 LoRA 方法进行 SFT，约需 16GB 显存，4-bit QLoRA 约需 6GB。也可以使用 [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md#%E6%A8%A1%E5%9E%8B) 支持的其他模型和方法，各种规模的显存估算见文档站的[所需硬件配置](https://docs.weclone.love/zh/docs/deploy/hardware.html)。

## 下载模型

中国境内推荐使用 [ModelScope](https://www.modelscope.cn/docs/models/download) 下载：

```bash
uv pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./models/Qwen2.5-7B-Instruct
```

换用其他模型时，修改 `settings.jsonc` 中的 `model_name_or_path`、`template` 和 `lora_target`。

微调多模态模型时：

- `model_name_or_path` 换成视觉语言模型，例如 `./models/Qwen2.5-VL-7B-Instruct`
- `template` 改为 `qwen2_vl`
- `include_type` 加上 `"image"`
- `lora_target` 末尾加上 `,visual.merger.mlp.0,visual.merger.mlp.2`

聊天以英文为主时，把 `default_system` 改成英文，`test_model_args.test_data_path` 改为 `dataset/eval/test_data-en.json`。

## 生成训练数据

```bash
weclone-cli make-dataset
```

生成训练数据时会过滤隐私信息：

- 默认通过 Microsoft Presidio 去除电话号码、电子邮件地址、信用卡号码（12–19 位数字）、IP 地址、地理位置名称、国际银行账户号码、加密货币钱包地址、年龄信息和通用身份证号码，但不能保证 100% 识别。
- `make_dataset_args.blocked_words` 中可以添加需要过滤的词句，包含禁用词的整句会被去掉。

> [!IMPORTANT]
> 🚨 请一定注意保护个人隐私，不要泄露个人信息！

可以根据自己的聊天习惯调整 `make_dataset_args` 中的时间窗口等参数，说明见文档站的[数据预处理](https://docs.weclone.love/zh/docs/deploy/data_preprocessing.html#%E7%9B%B8%E5%85%B3%E5%8F%82%E6%95%B0)。

## 训练

- 修改 `per_device_train_batch_size` 和 `gradient_accumulation_steps` 调整显存占用。
- 根据数据的数量和质量调整 `train_sft_args` 中的 `num_train_epochs`、`lora_rank`、`lora_dropout` 等参数。

单卡训练：

```bash
weclone-cli train-sft
```

多卡训练：取消 `settings.jsonc` 中 `deepspeed` 一行的注释，然后执行：

```bash
uv pip install "deepspeed<=0.16.9"
deepspeed --num_gpus=<显卡数量> weclone/train/train_sft.py
```

加密模式暂不支持分布式训练。

## 推理与测试

在浏览器中试聊，找到合适的 `temperature`、`top_p` 后写回 `settings.jsonc` 的 `infer_args`：

```bash
weclone-cli webchat-demo
```

试聊页只在明文模式下可用。加密模式下请用下面的接口和 `test-model` 测试。

启动 `/v1` 下的 OpenAI 兼容接口，调用时 API Key 填网页「机器人接入」页生成的令牌，见[接入聊天机器人](chatbot.md)：

```bash
weclone-cli server --inference --port 8005
```

用常见的日常聊天问题测试（不包含询问个人信息的问题）。保持上面的服务运行，另开一个终端执行以下命令，结果写入 `test_result-my.txt`：

```bash
weclone-cli test-model
```
