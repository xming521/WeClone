# 本地部署 LangBot 并接入 WeClone 数字分身

```
QQ / 微信 / Telegram / 飞书
        │
        ▼
     LangBot  ←  http://localhost:5300（管理后台）
        │  OpenAI 兼容请求
        ▼
  weclone-cloud serve  ←  http://127.0.0.1:8010/v1（带人设 + 鉴权）
        │
        ▼
     DeepSeek API（或你微调后的模型）
```

LangBot 只负责「接消息平台 + 路由」，人格完全由 `weclone-cloud serve` 注入。

---

## 0. 先确认分身服务在跑

```powershell
Invoke-RestMethod http://127.0.0.1:8010/healthz
# {"ok":true,"model":"deepseek-flash"}

Invoke-RestMethod http://127.0.0.1:8010/v1/models -Headers @{Authorization="Bearer local-key"} |
  ConvertTo-Json -Depth 4
# {"object":"list","data":[{"id":"deepseek-flash",...}]}
```

记下返回的 `id`，等会儿填进 LangBot。没跑就先 `.\start-server.cmd`。

---

## 1. 安装并启动 LangBot

双击 📄 `start-langbot.cmd`（工作目录固定在 `langbot-data\`，数据不会散落）。

等价的命令行：

```powershell
cd C:\Users\Administrator\Documents\deepseek-harness\default-workspace\weclone-cloud
mkdir langbot-data -Force | Out-Null
cd langbot-data
uv tool run langbot@latest
```

首次会下载依赖（几分钟），完成后再访问 <http://localhost:5300> 进管理后台。

> LangBot 官方也支持 Docker；如果你机器上有 Docker Desktop，用官方镜像更省事，但要注意容器里的 `127.0.0.1` 不是宿主机——见 1.1 与文末排错。

### 1.1 Docker 方案（推荐，Windows 上更稳）

本机用 `uvx` 跑 LangBot 时会撞上两个 Windows 特有问题：ChromaDB 报 `(code: 14) unable to open database file`，以及系统不允许在项目外建目录。Docker 方案把这两个都绕开了。

1. 双击 📄 `start-langbot-docker.cmd`
   - 自动 clone 官方仓库到 `langbot-src\`（没有 git 就下 zip）
   - 默认使用官方 Docker Hub 镜像；网络不好时加 `-Mirror` 可切 `docker.langbot.app`
   - 执行 `docker compose --profile all up -d`
2. 首次要下载几个镜像，等 <http://localhost:5300> 能打开。

**关键差异：容器里访问宿主机不能用 `127.0.0.1`。**

所以分身服务必须监听 `0.0.0.0`：

```powershell
start-persona-docker.cmd
```

窗口会多打印两行：

```
NOTE: bound to all interfaces (needed by Docker / other machines)
serve key : local-key   <- paste this into LangBot
```

LangBot 里的模型配置相应改成：

| 字段 | 值 |
|---|---|
| 请求 URL | `http://host.docker.internal:8010/v1` |
| API Key | 窗口里打印的 `serve key`（默认 `local-key`） |
| 模型名称 | `deepseek-flash` |

想让 key 更安全，启动前设一个自己的：

```powershell
$env:WECLONE_SERVE_KEY = "lb-随便一串"
start-persona-docker.cmd
```

如果 LangBot 报连接超时，多半是 Windows 防火墙拦了容器网段，用**管理员** PowerShell 放行：

```powershell
New-NetFirewallRule -DisplayName "weclone-docker" -Direction Inbound -Protocol TCP `
  -LocalPort 8010 -RemoteAddress 172.16.0.0/12,10.0.0.0/8 -Action Allow
```

---

## 2. 配置模型（关键）

后台左侧「模型」→「对话模型」→ 新建：

| 字段 | 填什么 |
|---|---|
| 模型名称 | 上一步 `/v1/models` 返回的 `id`（如 `deepseek-flash`） |
| 供应商 | 新建一个，类型选 **OpenAI 兼容** |
| 请求 URL（Base URL） | `http://127.0.0.1:8010/v1` |
| API Key | 你启动服务时用的 `WECLONE_SERVE_KEY`（默认 `local-key`） |
| 视觉能力 | 关 |
| **工具使用能力** | **关**（必须关，否则模型会去调工具，破坏人设） |

保存后点「测试」或设为默认。

### 为什么必须关工具

LangBot 的流水线一旦检测到模型支持 function calling，就会挂上内置工具（搜索、沙箱、知识库）。微调/人设模型一旦调工具，回答就会变成"我来帮你查一下"，人格立刻消失。

---

## 3. 流水线设置

「流水线」→ 默认流水线：

- **系统提示词**：留空或随便填。`weclone-cloud` 默认 `system_mode=replace`，会丢弃 LangBot 发来的 system，强制用人设；
- 关闭「知识库」「MCP」「沙箱」等扩展（用不到就先关，减少干扰）；
- 模型选刚才新建的那个。

> 如果你希望 LangBot 的提示词也生效，把 `cloud.jsonc` 里 `serve.system_mode` 改成 `"prepend"`（人设在前、它的在后）或 `"if_missing"`（只在没有 system 时注入）。

---

## 4. 绑定消息平台

在「机器人」页新建，按平台不同：

| 平台 | 需要准备 | 备注 |
|---|---|---|
| **QQ（OneBot v11）** | NapCat / Lagrange 等实现端，配置反向 WS 到 LangBot | 个人号有风控风险，用小号 |
| **微信** | 需要 gewechat 之类的协议端 | 风险最高，账号可能被封 |
| **Telegram** | @BotFather 申请 bot token | 最简单，五分钟能用 |
| **飞书 / 钉钉** | 企业自建应用 | 需要企业管理员权限 |

建议先用 **Telegram** 或 **页面机器人（Web Page）** 验证链路，再上 QQ/微信。

### 4.1 页面机器人（不绑定任何 IM，最快验证）

1. 后台「机器人」→「创建机器人」；
2. 名称随便填，平台/适配器选 **页面机器人**，保存；
3. 在机器人配置页找到「嵌入代码」，复制里面 `src=` 后面的地址，形如
   `http://localhost:5300/api/v1/embed/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`；
4. 双击项目根目录的 📄 `langbot-demo.html`，把地址粘进去，点「加载聊天组件」；
5. 右下角出现气泡，点开就能对话——这时链路是：

```
langbot-demo.html → LangBot(5300) → weclone-cloud(8010) → DeepSeek
```

如果气泡里回复的风格像你，说明整条链路（人设注入、鉴权、转发）都通了，再去绑 QQ/微信即可。

---

## 5. 排错

| 现象 | 原因 | 处理 |
|---|---|---|
| 模型「测试」失败 401 | API Key 不对 | 和 `WECLONE_SERVE_KEY` 逐字对齐（默认 `local-key`） |
| 连接被拒绝 | 8010 没在跑，或填了 localhost 但服务只绑了 127.0.0.1 | 先 `start-server.cmd`，确认 `healthz` |
| Docker 版 LangBot 连不上 | 容器内的 `127.0.0.1` 是容器自己 | Base URL 改成 `http://host.docker.internal:8010/v1` |
| 能回但不像「你」 | LangBot 自带的 system 覆盖了人设 | 确认 `serve.system_mode` 为 `replace`，重启服务 |
| 回复很啰嗦、爱分点 | 人设没生效，或模型的思考模式开着 | 检查 `persona_system.md` 是否存在、`api.thinking` 是否为 `auto` |
| 回复里出现「作为 AI」 | 人设文件没被加载 | `serve` 启动日志里应显示 `人设：...persona_system.md` |

---

## 6. 关停

| 要关什么 | 怎么做 |
|---|---|
| LangBot | 在它的窗口按 `Ctrl+C` |
| 分身服务 | 双击 `stop-server.cmd` |
| 两个都关 | 先 Ctrl+C，再 stop-server |
