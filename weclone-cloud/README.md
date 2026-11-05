> 本目录是 [WeClone](https://github.com/xming521/WeClone) 的子项目：
> 复用其数据解析能力，把打分 / 人设 / 对话换成纯 API 实现，无需 GPU。
> 独立仓库：https://github.com/lhongjie70-sudo/weclone-cloud
>
> **在 WeClone 仓库内使用**：父目录就是 WeClone，直接 `uv sync`，
> 不需要运行 `setup.cmd`（那会再克隆一份 WeClone），下文路径里的 `app\` 换成 `..\`。

# weclone-cloud

把「聊天记录 → 清洗打分 → 人设/微调 → 微信或网页分身」整条链路做成**纯 API 驱动、无需显卡**的一键方案。

基于 [WeClone](https://github.com/xming521/WeClone) 的完整副本改造：数据加工、风格总结、对话代理全部走 OpenAI 兼容 API（默认 DeepSeek），推理侧只跑一个带鉴权的本机代理。**没有 GPU 也能让「另一个你」在微信里回消息。**

```
微信小号 ──协议──> WeChatPadPro ──WS──> LangBot ──HTTP──> 本机分身代理 ──> DeepSeek
                                                          └ 人设注入 · 关思考 · 拟人延迟
```

---

## 特性

- **零显卡**：打分、人设、对话全部 API 化；本地只跑一个轻量 FastAPI 代理。
- **真实语料驱动**：从你的微信/Telegram 导出里抽高分对话，生成「说话风格规则 + 真实示例」的人设提示词。
- **可选的云端微调**：导出 OpenAI 微调格式 JSONL，交给支持 `/fine_tuning/jobs` 的平台；DeepSeek 无微调接口，走导出路线。
- **一套接口，多个前端**：OpenAI 兼容 `/v1/chat/completions`，可直接接 LangBot、AstrBot、任何 OpenAI SDK，也自带网页聊天。
- **微信落地**：Docker 跑 LangBot + WeChatPadPro，含两处适配器补丁（去重、异常修复），并做了拟人节奏与防封配置。
- **可复现**：打分有内容寻址缓存和断点续跑，导出带 `manifest.json`（条数、丢弃原因、token、sha256）。

---

## 测试版本（升级前先对照）

| 组件 | 已验证版本 | 固定方式 |
|---|---|---|
| WeClone | `master@d4af5ba`（2026-10-18） | 子目录跟随父仓库 |
| LangBot | **v4.11.1** | `start-langbot-docker.cmd`；镜像建议改成具体 tag |
| WeChatPad-Docker | commit `c6d1b7a` | `tools/prepare_wechatpad.ps1 -Ref`（已默认钉住） |
| Python | 3.12.x | `.venv` |

升级任何一项后，按 [维护手册](docs/maintenance.md) 第三节逐项复验（适配器补丁、compose 挂载、协议端接口）。

---

## 快速开始（只有分身，3 分钟）

### 小白三步（双击即用）

1. 双击 **`setup.cmd`** —— 自动拉取 WeClone、建虚拟环境、装依赖，并生成 `secrets.local.cmd`；
2. 用记事本打开 `secrets.local.cmd`，把 `WECLONE_API_KEY` 填成你的 DeepSeek key；
3. 双击 **`start-chat.cmd`** —— 自动打开 <http://127.0.0.1:8010/>。

接微信：双击 **`start-wechat.cmd`**（需 Docker Desktop），剩下几步在 [安装教程](docs/install.md) 第 5、6 节。

---

下面的命令等价于上面三步，适合命令行用户。

前置：Windows + PowerShell、Python 3.12、[uv](https://docs.astral.sh/uv/)、一把 DeepSeek API key。

```powershell
git clone https://github.com/<your-name>/weclone-cloud.git
cd weclone-cloud

# 1) 生成配置 + 安装依赖（轻量模式，约 300MB）
.\bootstrap.ps1 -Install

# 2) 写入密钥（也可以填进 secrets.local.cmd，启动脚本会自动加载）
$env:WECLONE_API_KEY   = "sk-..."
$env:WECLONE_SERVE_KEY = "local-key"

# 3) 自检
.\.venv\Scripts\python.exe -m weclone_cloud.cli doctor
```

生成人设并启动：

```powershell
# 从高分样本里挑 6 段真实对话，让模型总结你的说话风格
.\.venv\Scripts\python.exe -m weclone_cloud.cli persona `
  --input app\dataset\res_csv\sft\sft-my.scored.json --examples 6

# 启动（网页 + OpenAI 兼容接口）
.\.venv\Scripts\python.exe -m weclone_cloud.cli serve --persona persona_system.md --port 8010
```

打开 <http://127.0.0.1:8010/> 即可对话。

> 还没有数据？见 [安装教程](docs/install.md) 第二节：用 WeClone 的 `make-dataset` 把你的导出转成 `sft-my.json`。

---

## 快速开始（接入微信）

需要 Docker Desktop。完整步骤见 **[docs/install.md](docs/install.md)**，这里只列骨架：

```powershell
# 1) 分身（必须监听 0.0.0.0，容器才够得着）
.\start-persona-docker.cmd

# 2) 微信协议端
cd wechatpad-src; docker compose up -d; cd ..

# 3) LangBot（首次会自动 clone 并打补丁）
.\start-langbot-docker.cmd
```

然后在 <http://localhost:5300> 里：

| 配置项 | 值 |
|---|---|
| 模型 Base URL | `http://host.docker.internal:8010/v1` |
| 模型 API Key | 分身窗口打印的 `serve key` |
| 模型名 | `deepseek-flash` |
| 工具使用 | 关 |
| 系统提示词 | 留空（服务端强制注入人设） |

微信适配器（WeChatPad）的四个字段、扫码登录、掉线重连、封号规避，都在 [docs/runbook.md](docs/runbook.md)。

---

## 命令参考

| 命令 | 作用 | 关键参数 |
|---|---|---|
| `doctor` | 检查配置/密钥/连通性/任务状态 | — |
| `score` | API 打分，写回 `score` / `score_reason` | `--min-score --rescore --limit --workers --dry-run` |
| `export` | 转 OpenAI JSONL，去重、切分验证集 | `--system --min-score --val-ratio --max-tokens` |
| `finetune` | 上传 + 建任务 + 轮询（`generic` 只导出提示） | `--train --val --model --epochs --no-watch` |
| `status` / `cancel` | 查询 / 取消微调任务 | `--job-id --watch` |
| `persona` | 从高分样本生成人设提示词 | `--input --out --examples --min-score` |
| `serve` | OpenAI 兼容代理 + 网页 | `--host --port --model --persona` |
| `pipeline` | score + export 串联 | `--input --out-dir --min-score` |

使用 `python -m weclone_cloud.cli <命令>` 或安装后的 `weclone-cloud <命令>`。

---

## 目录结构

```
weclone-cloud/
├─ weclone_cloud/           # 核心包（不修改 WeClone 源码）
│  ├─ config.py             # pydantic 配置、密钥来源、计价
│  ├─ api.py                # OpenAI 兼容客户端（重试/缓存/思考开关/JSON 降级）
│  ├─ prompts.py            # 打分 rubric 与 JSON schema
│  ├─ scoring.py            # 并发打分、断点续跑、报告
│  ├─ dataset.py            # JSONL 导出、去重、切分、manifest
│  ├─ finetune.py           # /files + /fine_tuning/jobs
│  ├─ serve.py              # 鉴权代理：人设注入、关思考、剥离 reasoning、拟人延迟
│  └─ web/index.html        # 自带聊天页
├─ tools/                   # 数据转换、启动器、补丁脚本
│  ├─ from_alpha.py         # Alpha 导出 → WeClone CSV
│  ├─ inspect_data.py       # 识别数据形态
│  ├─ launch.py             # 一键启动 + 日志分流
│  ├─ prepare_langbot.ps1   # 拉取 LangBot + 国内镜像可选
│  ├─ prepare_wechatpad.ps1 # 拉起 WeChatPadPro
│  └─ fix_langbot_compose.py# 修 Windows 挂载路径
├─ tests/                   # 无网络单测（httpx.MockTransport）
├─ docs/
│  ├─ install.md            # 从零安装（Windows 全流程）
│  ├─ runbook.md            # 运维手册：启动顺序、重启恢复、故障速查
│  ├─ architecture.md       # 与 WeClone 的映射、设计取舍
│  ├─ providers.md          # 各平台配置示例
│  ├─ deploy.md             # 部署
│  └─ langbot.md            # LangBot 接入
├─ start-*.cmd / stop-*.cmd # 一键脚本
├─ bootstrap.ps1            # 复制 WeClone + 装依赖
└─ cloud.example.jsonc      # 配置模板
```

---

## 文档

| 文档 | 内容 |
|---|---|
| [docs/install.md](docs/install.md) | 从零安装：环境、数据、分身、Docker、微信 |
| [docs/runbook.md](docs/runbook.md) | 日常启动、断电恢复、微信重登、故障速查 |
| [docs/architecture.md](docs/architecture.md) | 与 WeClone 的对应关系、关键取舍 |
| [docs/providers.md](docs/providers.md) | OpenAI / 百炼 / 本地模型等配置 |
| [docs/langbot.md](docs/langbot.md) | LangBot 接入细节 |
| [docs/maintenance.md](docs/maintenance.md) | **长期维护**：版本矩阵、升级步骤、回归检查、发布流程 |

---

## 已知限制

- **DeepSeek 没有微调接口**：`finetune` 对它只能导出 JSONL；要训练权重请换支持微调的平台，或用 WeClone 原生 LoRA（需要 GPU）。
- `persona` 只是提示词模仿，**不改权重**，稳定性不如微调。
- 只支持文本；带图片的样本在导出时丢弃。
- 微信协议端（WeChatPadPro）有封号风险，**仅限小号自用**，不要做营销、不要群发。
- 未内置速率限制；对外提供服务请自行加网关。

---

## 免责声明

- 本项目仅供个人学习与自用。使用他人聊天记录前，**必须取得对方同意**。
- 微信个人号自动化可能违反平台协议并导致封号，风险自负。
- AI 生成内容应保留身份标识，不得用于冒充他人实施欺诈。

## License

AGPL-3.0，与上游 WeClone 保持一致。
独立仓库：https://github.com/lhongjie70-sudo/weclone-cloud
