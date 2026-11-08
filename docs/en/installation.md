[简体中文](../zh/installation.md)｜English

# Installation and Configuration

## What to install

| Use | Install | Also needs |
|---|---|---|
| Life portrait, web UI, assistant access | `uv pip install -e .` | Codex CLI or any OpenAI-compatible API |
| Speaking style (`build-profile --purpose clone`) | `uv pip install --group style --torch-backend cpu -e .` | The LTP model, which runs on CPU |
| Fine-tuned clone | `uv pip install --group main -e .` | NVIDIA GPU, CUDA 12.6 or later, a spaCy model |

## Base install

Python 3.12 is required. [uv](https://docs.astral.sh/uv/) is recommended for managing the environment:

```bash
git clone https://github.com/xming521/WeClone.git && cd WeClone
uv venv .venv --python=3.12
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e .
cp settings.template.jsonc settings.jsonc
```

All settings live in `settings.jsonc`. Windows has not been tested thoroughly, so WSL2 is recommended.

## Model backend

Portrait extraction, memory organization and the portrait summary all call API models. `agent_distill_args.llm_provider` picks the backend. The settings below can also be filled in and tested under Model service on the web Settings page, which changes only these entries and keeps the comments and other settings of `settings.jsonc` as they are:

- `codex_exec` (default): calls models through the local [Codex CLI](https://github.com/openai/codex), which must be installed and signed in. The model and reasoning effort are set in `codex_exec_args`.
- `api`: any endpoint compatible with OpenAI Chat Completions. Put the address, key and model name in `base_url`, `llm_api_key` and `model_name` under `make_dataset_args`, and the rest under `agent_distill_args`:
  - `context_window`: the model's context window, 128000 when unset. A request uses at most 95% of it. The connection test on the Settings page fills it in when the endpoint lists it.
  - `reasoning_effort`: sent as `reasoning_effort` as it is. The provider's docs list the values it takes; leave it out for the provider's default.
  - `extra_body`: a JSON object merged into every request body, for a provider's own parameters such as a thinking switch.
  - `structured_output`: the format the model is asked to answer in, `json_schema` or `json_object`. Left out, it is `json_schema`. Set `json_object` for a service that does not take `json_schema`, such as DeepSeek. Test connection on the Settings page picks the one the service takes, and choosing DeepSeek as the provider fills it in.
  - `batch_size`, `timeout`, `max_tokens`: concurrency, timeout and output limit.

  These are sent as they are, and a request the provider refuses fails with its error. Test connection on the Settings page sends the same request a build does, so it can check them first.

## Speaking style dependencies

```bash
uv pip install --group style --torch-backend cpu -e .
hf download LTP/base --local-dir models/LTPbase
```

[LTP](https://github.com/HIT-SCIR/ltp) handles Chinese word segmentation and parsing. It runs on CPU by default, and `--torch-backend cpu` installs the CPU build of PyTorch, so CUDA isn't needed.

## Fine-tuning dependencies

1. Install CUDA 12.6 or later (skip if already installed).
2. Install the fine-tuning dependencies and the spaCy model used for PII filtering (`zh_core_web_sm` for Chinese chats):
   ```bash
   uv pip install --group main -e .
   uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
   ```
3. Check that PyTorch can see CUDA (not needed on Mac):
   ```bash
   python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
   ```
4. (Optional) Install FlashAttention to speed up training and inference: `uv pip install flash-attn --no-build-isolation`. If the versions do not match, use a prebuilt wheel from [prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases).

See [Installation](https://docs.weclone.love/en/docs/deploy/environment) on the docs site for more environment issues.
