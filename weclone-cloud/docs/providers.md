# 各平台配置示例

通用的判断方法：**打开平台文档，搜 `fine_tuning/jobs`。**
有 → 用 `provider="openai"`；没有 → 用 `provider="generic"`，只导出 JSONL。

---

## OpenAI

```jsonc
{
  "api":      { "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "api_key_env": "WECLONE_API_KEY" },
  "finetune": { "provider": "openai", "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY", "base_model": "gpt-4o-mini-2024-07-18",
                "api_style": "method", "epochs": 3 }
}
```

```powershell
$env:WECLONE_API_KEY = "sk-..."
$env:OPENAI_API_KEY  = "sk-..."
weclone-cloud finetune --train finetune_data\train.jsonl --val finetune_data\val.jsonl
```

注意：验证集要么不给，要么 ≥10 条；`export` 已经处理了这个下限。

---

## Azure OpenAI

Azure 的路径是 `/openai/deployments/{deployment}/chat/completions`，且用 `api-key` 头而不是 `Authorization`。本项目按标准 OpenAI 形状发请求，所以需要网关层兼容（例如 LiteLLM / APIM 做协议转换），或把 `base_url` 指到已改写的兼容前缀：

```jsonc
"api": { "base_url": "https://<resource>.openai.azure.com/openai/v1", "model": "<deployment>",
         "api_key_env": "AZURE_OPENAI_API_KEY" }
```

微调同理，多数情况下建议直接用 `generic` 导出、在 Azure 控制台/Azure CLI 上传。

---

## 阿里云百炼 / 火山引擎 / 智谱

这类平台通常提供自己的 HTTP API（签名方式不同），**不保证** OpenAI 的 `/files` + `/fine_tuning/jobs` 路由。稳妥流程：

```jsonc
"finetune": { "provider": "generic" }
```

```powershell
weclone-cloud export --input app\dataset\res_csv\sft\sft-my.scored.json --min-score 3 --out-dir finetune_data
# 到平台控制台上传 finetune_data\train.jsonl，训练完成后拿到模型名，写入：
# finetune_state.json -> {"model": "qwen-finetune-xxxx", "status": "succeeded"}
weclone-cloud serve
```

如果该平台的"兼容模式"声称支持 OpenAI 协议，先用 `doctor` 连通性检查，再 `finetune --dry-run` 看请求形状是否被接受。

---

## DeepSeek（默认后端）

```jsonc
{
  "api": {
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-flash",
    "api_key_env": "WECLONE_API_KEY",
    "json_mode": "object",
    "thinking": "auto",
    "max_tokens": 400,
    "max_concurrency": 8,
    "price_input_per_million": 2,
    "price_output_per_million": 8,
    "price_currency": "CNY"
  },
  "finetune": { "provider": "generic" }
}
```

```powershell
$env:WECLONE_API_KEY = "sk-..."    # platform.deepseek.com 创建
weclone-cloud doctor
weclone-cloud score --input app\dataset\res_csv\sft\sft-my.json --limit 20   # 先抽 20 条试跑
```

**没有微调接口**（截至 2026-10 官方文档只列了对话 / JSON / 工具 / Files / 搜索 / 缓存）。训练走 `provider="generic"` 导出 JSONL，或回 WeClone 本机 LoRA。

三个已处理的细节：

| 现象 | 原因 | 代码里的处理 |
|---|---|---|
| 打分变慢、贵、结果被截断 | 思考模式**默认开启**，且思考 token 计入输出 | `thinking: "auto"` → 命中 deepseek 时发 `{"thinking":{"type":"disabled"}}` |
| 偶发返回空 content | 官方 JSON Output 已知问题 | 去掉 `response_format` 重试一次（`api.py:_execute`） |
| `400 response_format` | 只支持 `json_object`，不支持 `json_schema` | 自动降级为 `json_object`（`api.py:_downgrade`） |

模型与价格（元/百万 token，高峰价；空闲时段减半）：

| 模型 | 输入(未命中缓存) | 输出 | 备注 |
|---|---|---|---|
| `deepseek-flash` | 2 | 8 | 1M 上下文，并发 2500，日常打分够用 |
| `deepseek-v4-pro` | 9 | 27 | 更强，适合疑难样本复核 |

建议先用 `flash` 跑全量，再用 `pro` 抽查 1-2 分和 5 分边界样本，校准你的 rubric。

---

## 本地 vLLM / Ollama / LM Studio

只有 `/chat/completions`，没有 fine-tune 接口：

```jsonc
"api": { "base_url": "http://127.0.0.1:8000/v1", "model": "Qwen2.5-7B-Instruct", "api_key": "EMPTY", "json_mode": "none" }
```

打分可用（`json_mode: "none"` 时靠文本里的 JSON 抽取，`api.py:parse_json_object` 支持围栏与前后缀）。
训练请用 WeClone 原生 `train-sft`。

---

## 通用排错

| 现象 | 大概率原因 | 处理 |
|---|---|---|
| `401 invalid api key` | 环境变量没设置 / 设在了另一个 shell | `weclone-cloud doctor` 看 Key 来源 |
| `400 response_format` | 网关不支持 `json_schema` | 已自动降级为 `json_object`；若仍失败，改 `json_mode: "none"` |
| 分数全是 0 / 解析失败 | 模型不愿输出 JSON | 换更强的模型，或 `json_mode: "schema"` 强约束 |
| 上传 413 | 文件超平台上限 | 用 `export --max-tokens` 或按平台上限拆分后多次 `finetune` |
| 任务一直 `validating_files` | 数据格式被拒 | 看 `weclone-cloud status --watch` 的 events；多半是 messages 顺序/角色问题 |
