简体中文｜[English](../en/installation.md)

# 安装与配置

## 按用途安装

| 用途 | 安装 | 其他要求 |
|---|---|---|
| 人生画像、网页、助手接入 | `uv pip install -e .` | Codex CLI 或任意 OpenAI 兼容 API |
| 说话风格（`build-profile --purpose clone`） | 另装 `ltp`、`scipy` | LTP 模型，CPU 即可运行 |
| 微调分身 | `uv pip install --group main -e .` | NVIDIA GPU、CUDA 12.6 及以上、spaCy 模型 |

## 基础安装

需要 Python 3.12，推荐使用 [uv](https://docs.astral.sh/uv/) 管理环境。下载较慢时可以开启代理或使用镜像：

```bash
git clone https://github.com/xming521/WeClone.git && cd WeClone
uv venv .venv --python=3.12
source .venv/bin/activate   # Windows 下执行 .venv\Scripts\activate
uv pip install -e .         # 国内镜像：-i https://pypi.tuna.tsinghua.edu.cn/simple/
cp settings.template.jsonc settings.jsonc
```

所有配置都在 `settings.jsonc` 中修改。Windows 环境未经严格测试，建议使用 WSL2。

## 配置模型后端

抽取画像、整理记忆和生成领域总结都调用 API 模型，由 `agent_distill_args.llm_provider` 选择后端：

- `codex_exec`（默认）：通过本机的 [Codex CLI](https://github.com/openai/codex) 调用模型，需要先安装并登录。模型和推理强度在 `codex_exec_args` 中设置。
- `api`：任意 OpenAI 兼容接口。地址、密钥和模型名填在 `make_dataset_args` 的 `base_url`、`llm_api_key`、`model_name`，并发数、超时和输出上限在 `agent_distill_args` 的 `batch_size`、`timeout`、`max_tokens` 中设置。运行 `build-profile` 时必须用 `--max-context-tokens` 指定模型的输入上限。

## 说话风格依赖

```bash
uv pip install ltp scipy
hf download LTP/base --local-dir models/LTPbase
```

[LTP](https://github.com/HIT-SCIR/ltp) 用于中文分词和句法分析，默认在 CPU 上运行。国内下载较慢时，可以在命令前加上 `HF_ENDPOINT=https://hf-mirror.com`。

## 微调依赖

1. 安装 CUDA 12.6 及以上版本（已安装可跳过），参考 [LLaMA Factory 安装文档](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/installation.html#cuda)。
2. 安装微调依赖，以及隐私过滤使用的 spaCy 模型（英文聊天换成 `en_core_web_sm`）：
   ```bash
   uv pip install --group main -e .
   uv pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.8.0/zh_core_web_sm-3.8.0-py3-none-any.whl
   ```
3. 检查 PyTorch 能否识别 CUDA（Mac 不需要）：
   ```bash
   python -c "import torch; print('CUDA是否可用:', torch.cuda.is_available())"
   ```
4. （可选）安装 FlashAttention 加速训练和推理：`uv pip install flash-attn --no-build-isolation`。版本不匹配时可以使用 [prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases) 的预编译包。

更多环境问题见文档站的[环境配置](https://docs.weclone.love/zh/docs/deploy/environment.html)。
