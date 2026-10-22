# 架构：为什么"复制 WeClone"而不是重写

## 1. 复用与替换的边界

WeClone 的价值集中在前半段：多平台聊天导出解析、时间窗合并、QA 配对、PII 脱敏、按人切分（stage2）、以及后来那套记忆/画像流水线。这些逻辑和"用谁的算力"无关。

真正卡住"纯 API 运行"的只有三处：

| 卡点 | 原实现 | 后果 |
|---|---|---|
| 数据打分 | `LLMCleaningStrategy` → `vllm_infer()`（`weclone/data/clean/strategies.py:74`） | 无 vLLM（Windows/macOS/CPU）直接退出 |
| 训练 | `run_training()` → LLaMA-Factory `run_exp` | 需要本地 GPU、数十 GB 显存 |
| 推理 | `server --inference` → `ChatModel` 加载权重 | 需要把模型文件放本机 |

所以本项目**只替换这三处**，其余原样保留（连 `weclone-cli make-dataset` 都继续用）。这让"复制 WeClone"不是负担，而是白拿一个成熟的数据清洗器。

## 2. 为什么新开一个包，而不是改 `weclone/`

- 上游 `weclone` 仍在快速迭代（v0.4 刚加了档案/加密/审核台）；直接改它，升级就是大面积冲突。
- `weclone_cloud` 只依赖它的**产物**（`sft-my.json`）和**配置约定**（`settings.jsonc`），不 import 它的内部函数，因此可以独立测试、独立发布。
- 需要调用 WeClone 时，走它的 CLI（子进程语义），边界清晰。

唯一的耦合点是数据集文件格式，而它恰好是 LLaMA-Factory 的 sharegpt 约定，非常稳定：

```json
{"id":"0","time":"2024-05-01T10:00:00","score":4,
 "messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],
 "images":[],"system":"请你扮演一名人类…","chat_with":"peer"}
```

## 3. 打分阶段的设计取舍

**为什么单条请求，不拼批。**
把 k 条对话拼进一个 prompt 让模型返回 k 个分数，能省 k 倍输入 token。但实测类系统里，这种"批内对齐"是解析失败的主要来源（漏项、串号、id 重复）。WeClone 自己也踩过：它用 `QaPairScoreWithId` + 逐条对齐 + 断言兜底（`data/clean/strategies.py:131`）。

本项目选择**单条 + 高并发 + 缓存**：成本靠并发摊薄，可靠性靠契约保证。若确实要压成本，正确做法是"两阶段"——先用小模型粗筛掉 1-2 分，再对大模型只跑 3-5 分区间的复核。

**为什么把分数写回原文件。**
写回 `score` 字段后，产物同时满足两个消费者：

- 本项目的 `export`（按 `min_score` 过滤）；
- WeClone 的 `CleaningStrategy.clean()`（按 `accept_score` 过滤，`data/clean/strategies.py:32`）。

于是"用 API 打分 + 用 WeClone 本地训练"这种混合用法也成立，不需要转换格式。

**断点续跑的三层保障**：

1. `score_records` 默认跳过 `score_source=cloud` 的记录（`--rescore` 强制重打）；
2. `CompletionCache` 以 `model + messages + response_format + temperature` 的 sha256 为键，重跑命中缓存不发请求；
3. 每批结果即时落盘（`write_json` 原子替换），Ctrl+C 不会丢已完成的部分。

## 4. 导出阶段：为什么必须"清洗"而不只是"转换"

直接把 WeClone 的 `messages` 塞进 JSONL 会有四类脏数据，云平台不会替你修：

| 问题 | 处理 | 为什么 |
|---|---|---|
| 首条是 `system`，且每条不同 | 丢弃样本内 system，统一由 `--system` 注入 | 微调时 system 不一致会让模型学不到稳定人设 |
| 最后一条不是 `assistant` | 丢弃 | 训练目标是"给定上下文生成回复"，尾部必须是目标 |
| 只有 `assistant` 一条 | 丢弃（`too_short`） | 没有输入，模型只会学到无条件输出 |
| 完全重复的对话 | 按 messages 的 sha256 去重 | 重复样本会让模型过拟合到那几句话 |
| 带 `images` | 丢弃（`image_sample`） | 云平台多模态格式各不相同，强行转会失败 |

`manifest.json` 记录每类丢弃的数量与文件 sha256，便于回查"为什么只剩 3000 条"。

## 5. 微调阶段：两种请求形状

OpenAI 在 2024 年把超参从顶层 `hyperparameters` 挪到 `method.supervised.hyperparameters`，但大量兼容网关还停在老形状。所以：

```jsonc
// api_style = "method"（默认，新接口）
{"model":"...","training_file":"file-x",
 "method":{"type":"supervised","supervised":{"hyperparameters":{"n_epochs":3}}}}

// api_style = "legacy"（老网关）
{"model":"...","training_file":"file-x","hyperparameters":{"n_epochs":3}}
```

`finetune_state.json` 是**幂等锚点**：上传成功就记住 `file_id`，建任务成功就记住 `job_id`。所以"网络断了重跑"不会重复上传、重复计费。

## 6. 服务阶段：为什么是代理而不是本地推理

微调完成后，模型在云端，本地没有任何权重。此时"部署"退化成三件事：

1. 把客户端的 `model` 换成微调模型名；
2. 在缺省时注入固定 `system`（保证人设一致）；
3. 加鉴权，防止别人白用你的额度。

`serve.py` 就是这三件事，外加流式透传（`stream=true` 直接把上游 SSE 转发，不改写内容）。这样 AstrBot/LangBot 这类"只认 OpenAI 协议"的客户端零改动接入。

## 7. 与 WeClone 画像流水线的关系

v0.4 的记忆/画像（`weclone/data/agent/`）与 SFT 是**两条独立链路**：

- SFT：聊天记录 → 问答对 → 微调模型（学"怎么说"）；
- 画像：聊天记录 → 结构化记忆 → 审核 → 画像文件（学"知道什么"）。

本项目只接管前者。若两者都要，推荐：

```
weclone-cli make-dataset          # 一次解析，两处复用
├─ weclone-cli build-profile      # 产出画像（审核后作为 system / RAG 记忆）
└─ weclone-cloud score → export → finetune
```

注意 `build-profile` 默认走 `codex_exec` 后端；改成 `agent_distill_args.llm_provider = "api"` 后即可复用本项目的 `base_url` 配置（WeClone 读的是 `make_dataset_args` 里那三个字段）。

## 8. DeepSeek 适配：为什么多了三个开关

DeepSeek 的 API 形状和 OpenAI 有几处实质差异，直接套用会"看起来能跑，但结果不对"：

| 差异 | 不加处理的后果 | 处理 |
|---|---|---|
| 思考模式**默认开启** | 每条打分先花几百 thinking token，慢 3-5 倍、贵数倍；`max_tokens` 被思考吃掉后 `content` 为空 | `api.thinking="auto"`：base_url 命中 deepseek 时注入 `{"thinking":{"type":"disabled"}}` |
| JSON 输出要求 prompt 含 "json" + 给示例，且**偶发返回空 content** | 空 content 被解析成"评分失败" | prompt 写死 json 示例；空内容时去掉 `response_format` 重试一次（模型会按示例直接吐 JSON） |
| 只有 `json_object`，没有 `json_schema` | 400 直接失败 | 捕获 400 且提示里含 `response_format` → 降级重发 |

顺带两个工程收益：

- **单价进配置**（`api.price_input_per_million` / `output`），`score` 结束时打印真实 token 与估算费用。DeepSeek 有高峰/空闲两档价，默认填高峰值做保守估计。
- **思考内容单独解析**（`Completion.reasoning`），但不参与评分逻辑——它只用于排查"为什么模型答歪了"。

不做微调时，`persona` 命令把高分样本拼成少样本人设提示词，`serve` 直接用它跑 DeepSeek 基座。这是"零训练成本"的兜底路径：学不到权重层面的风格，但能把句长、口头禅、语气词对齐，也顺便验证数据质量是否值得花钱训练。
