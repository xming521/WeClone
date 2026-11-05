# 本地部署手册（Windows / 已有数据）

三种"本地"要分开看，别混：

| 你想要的 | 数据去哪 | 需要什么 |
|---|---|---|
| **A. 本机跑管线**（打分、导出、人设） | 发到你配的 API（默认 DeepSeek 云） | 一个 API Key |
| **B. 本机对外提供服务**（局域网/公网能用） | 同上，只是入口开在 0.0.0.0 | 端口 + 防火墙 |
| **C. 完全离线**（一个字节都不出本机） | 本机 vLLM/Ollama | ≥16G 显存或能接受慢 |

---

## 0. 一次性准备

```powershell
cd C:\Users\Administrator\Documents\deepseek-harness\default-workspace\weclone-cloud

# 复制 WeClone + 装轻量依赖（不含 CUDA torch）
.\bootstrap.ps1 -Install

# 密钥（写进当前会话；想永久生效用 [Environment]::SetEnvironmentVariable）
$env:WECLONE_API_KEY   = "sk-..."      # DeepSeek
$env:WECLONE_SERVE_KEY = "local-key"   # 本地服务口令，自己定

weclone-cloud doctor
pytest -q
```

`doctor` 预期看到：`思考模式 关闭`、`计价 CNY 2/8`、以及一行黄色提示「DeepSeek 没有微调接口」（正常）。

---

## 1. 你的数据属于哪种形态

先让工具自己认（只读，不改文件）：

```powershell
python tools\inspect_data.py "D:\你自己的导出\目录或文件"
```

它会打印「形态 + 字段 + 下一步命令」。常见输出：

| 输出 | 形态 | 怎么接 |
|---|---|---|
| `WeClone SFT 数据集` | 已是 `sft-my.json` | 跳到 §2，直接 `score` |
| `OpenAI 风格问答数组（无分数）` | 有 messages、没 score | 跳到 §2，`score` 会补分数 |
| `微信导出 CSV` | `is_sender/msg/CreateTime/type_name` | 先 §1.1 `make-dataset` |
| `Telegram 导出` | `result.json` | 先 §1.2 `make-dataset` |
| `SQLite 数据库` | 原始库 | 先用 chatlog / WeChatMsg 导出 CSV，再 §1.1 |
| `WeClone 加密文件` | `.enc` | 先解密或改明文重跑 |

### 1.1 微信 CSV → 数据集

```powershell
# 目录结构必须是：app\dataset\csv\<联系人>\xxx_<起始序号>_<结束序号>.csv
# 示例：app\dataset\csv\张三\0_10000.csv
weclone-cli make-dataset
```

**如果用的是 Alpha 导出**（每个联系人一个 `<N-名字>\<名字>.csv`，旁边有 `media-manifest.json`），先转换：

```powershell
python tools\from_alpha.py --input "C:\Users\Administrator\Desktop\wx\Alpha导出-20261020-123117" --dry-run
python tools\from_alpha.py --input "C:\Users\Administrator\Desktop\wx\Alpha导出-20261020-123117"
```

它默认只保留 `text` 行，会打印每个联系人的保留行数与被丢弃的类型统计。原因见脚本头部注释：Alpha 的 `sticker` 行正文是一整段 `<msg><emoji/></msg>` XML，直接喂给 WeClone 会被当正文；`image` 行的 `[图片]` 同理。

产物：`app\dataset\csv\<联系人>\<联系人>.csv`（UTF-8 无 BOM）。之后 `cd app && weclone-cli make-dataset`。

> 若某位联系人的保留行数为 0（纯图片/表情聊天），删掉对应目录即可。

产物：`app\dataset\res_csv\sft\sft-my.json`（若 `dataset_info.json` 里 `chat-sft.file_name` 不同，按那个名字）。

### 1.2 Telegram 导出 → 数据集

把 `ChatExport_*` 文件夹整批放进 `app\dataset\telegram\`，然后改 `app\settings.jsonc`：

```jsonc
"make_dataset_args": {
  "platform": "telegram",
  "telegram_args": { "my_id": "user1234567890" }   // 换成你自己的 from_id
}
```

`my_id` 在导出的 `result.json` 里找你发的消息的 `from_id`。然后 `weclone-cli make-dataset`。

> 加密模式（`storage_mode: "encrypted"`）会产出 `.enc`，本工具读不了。跑云端链路请先改成 `"plaintext"` 重跑，或 `weclone-cli decrypt` 导出明文。

---

## 2. 打分（本地跑，调用你配的 API）

```powershell
# 先抽 20 条，验证分数分布合不合理、花费多少
weclone-cloud score --input app\dataset\res_csv\sft\sft-my.json --limit 20

# 看结果再决定阈值；全量跑（可 Ctrl+C，重跑会续跑）
weclone-cloud score --input app\dataset\res_csv\sft\sft-my.json --min-score 3
```

产物：
- `sft-my.json.scored.json`：每条加了 `score` / `score_reason` / `score_source`
- `sft-my.json.scored.report.json`：分布、失败数、token、估算费用

> 数据里**原本没有 score** 也能用：`score` 会给每条补上；若你不想打分只想导出，用 `export --min-score 0`。

---

## 3. 起服务（本机 A / 局域网 B）

```powershell
# 本机自用
weclone-cloud serve --persona persona_system.md

# 局域网内别人也能用（注意设好 WECLONE_SERVE_KEY）
weclone-cloud serve --host 0.0.0.0 --port 8010 --persona persona_system.md
```

局域网放行（管理员 PowerShell，执行一次）：

```powershell
New-NetFirewallRule -DisplayName "weclone-8010" -Direction Inbound `
  -Protocol TCP -LocalPort 8010 -Action Allow -Profile Private
```

然后别人用 `http://<你的内网IP>:8010/v1`，`Authorization: Bearer local-key`。

后台常驻 + 日志：

```powershell
.\deploy.ps1 -Persona persona_system.md -Port 8010 -Background
# 日志：logs\serve-YYYYMMDD-HHMMSS.log，PID：logs\serve.pid
# 停止：Stop-Process -Id (Get-Content logs\serve.pid)
```

要开机自启，用 NSSM 注册成服务（比任务计划程序稳）：

```powershell
nssm install WeCloneServe "C:\path\to\.venv\Scripts\weclone-cloud.exe" "serve --host 0.0.0.0 --port 8010"
nssm set WeCloneServe AppDirectory "C:\Users\Administrator\Documents\deepseek-harness\default-workspace\weclone-cloud"
nssm start WeCloneServe
```

---

## 4. 完全离线（一个字节不出本机）

用 vLLM 或 Ollama 起一个 OpenAI 兼容端点，再把 `cloud.jsonc` 指过去：

```jsonc
"api": {
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "Qwen2.5-14B-Instruct",
  "api_key": "EMPTY",
  "json_mode": "object",
  "thinking": "omit"
}
```

vLLM 启动示例（Linux/WSL，显存 ≥24G）：

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct --port 8000 --max-model-len 16384
```

注意：

- **别指望本地跑 DeepSeek 本体**（671B MoE），本地只能跑 Qwen / Llama 这类稠密模型；
- 7B 打分的区分度一般，建议 14B 起步；打分提示词偏判断任务，14B 通常够用；
- `json_mode: "object"` 在 vLLM 上可用（内部走 guided decoding），若报错就改 `"none"`，代码会从文本里抠 JSON；
- 本地模型的 `max_concurrency` 别超过 vLLM 的并发能力，建议 2-4。

这样 `score` / `serve` 全在本机闭环，适合聊天记录不能出内网的场景。

---

## 5. 验收清单

```powershell
weclone-cloud doctor                        # 配置/密钥/连通性
weclone-cloud score --limit 5 ...           # 能打出分数
weclone-cloud persona -i ...scored.json     # 生成 persona_system.md
weclone-cloud serve --persona ...           # 网页能回话
curl http://127.0.0.1:8010/v1/models -H "Authorization: Bearer local-key"
```

## 6. 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| `缺少 API Key` | 环境变量没设/设错 shell | 重新 `$env:WECLONE_API_KEY=...`，再 `doctor` |
| `数据集不存在` | 路径不对（`app\...` 相对路径） | 用绝对路径，或先 `cd` 到项目根 |
| 分数全是 1-2 | 模型太弱 / prompt 里没 json 示例 | 换 `deepseek-flash` 试；检查 `json_mode` |
| 服务 401 | 请求没带 `Authorization` | `-H "Authorization: Bearer $env:WECLONE_SERVE_KEY"` |
| 局域网访问不到 | Windows 防火墙 | 见 §3 的 `New-NetFirewallRule` |
| 端口被占 | 上次后台没退出 | `Stop-Process -Id (Get-Content logs\serve.pid)` 或换端口 |
