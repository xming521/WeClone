[简体中文](../zh/finetune.md)｜English

# Fine-tuned Clone

Fine-tune an open model on your chat history to get a clone that carries your speaking style. This route needs a GPU and the [fine-tuning dependencies](installation.md#fine-tuning-dependencies).

> [!IMPORTANT]
> - Results depend heavily on model size and on the amount and quality of your chat data. 7B models give average results. Models with 14B or more parameters do better.
> - WeClone is still iterating quickly, and current results do not represent final performance.

## Hardware

By default WeClone fine-tunes Qwen2.5-7B-Instruct with LoRA in the SFT stage, which needs about 16GB of VRAM. 4-bit QLoRA needs about 6GB. Other models and methods supported by [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main#supported-models) also work. See [Hardware Requirements](https://docs.weclone.love/en/docs/deploy/hardware.html) on the docs site for VRAM estimates at other sizes.

## Download a model

Download from [Hugging Face](https://huggingface.co/docs/hub/models-downloading):

```bash
hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
```

To use another model, change `model_name_or_path`, `template` and `lora_target` in `settings.jsonc`.

To fine-tune a multimodal model:

- set `model_name_or_path` to a vision-language model such as `./models/Qwen2.5-VL-7B-Instruct`
- set `template` to `qwen2_vl`
- add `"image"` to `include_type`
- append `,visual.merger.mlp.0,visual.merger.mlp.2` to `lora_target`

For chats mainly in English, also translate `default_system` and set `test_model_args.test_data_path` to `dataset/eval/test_data-en.json`.

## Build the training data

```bash
weclone-cli make-dataset
```

Personal information is filtered out while the training data is built:

- By default, Microsoft Presidio removes phone numbers, email addresses, credit card numbers, IP addresses, geographic location names, international bank account numbers, cryptocurrency wallet addresses, age information and generic ID numbers, but it cannot guarantee 100% detection.
- Add words or phrases to filter to `make_dataset_args.blocked_words`. Any sentence containing one is removed.

> [!IMPORTANT]
> 🚨 Please protect personal privacy and do not leak personal information!

You can tune the time windows and other `make_dataset_args` settings to your chat habits. See [Data Preprocessing](https://docs.weclone.love/en/docs/deploy/data_preprocessing.html#related-parameters) on the docs site.

## Train

- Adjust `per_device_train_batch_size` and `gradient_accumulation_steps` to control VRAM usage.
- Adjust `num_train_epochs`, `lora_rank`, `lora_dropout` and other `train_sft_args` to the amount and quality of your data.

Single GPU:

```bash
weclone-cli train-sft
```

Multiple GPUs: uncomment the `deepspeed` line in `settings.jsonc`, then run:

```bash
uv pip install "deepspeed<=0.16.9"
deepspeed --num_gpus=<number_of_gpus> weclone/train/train_sft.py
```

Encrypted mode does not support distributed training yet.

## Inference and testing

Chat with the model in a browser, find suitable `temperature` and `top_p` values, and write them back to `infer_args` in `settings.jsonc`:

```bash
weclone-cli webchat-demo
```

The chat page works only in plaintext mode. In encrypted mode, test with the API below and `test-model`.

Start the OpenAI-compatible API under `/v1`. Callers use the token from the bot access page of the web UI as the API key, see [Chat bots](chatbot.md):

```bash
weclone-cli server --inference --port 8005
```

Test with common everyday chat questions (no questions asking for personal information). Keep the server above running and run this in another terminal. Results go to `test_result-my.txt`:

```bash
weclone-cli test-model
```
