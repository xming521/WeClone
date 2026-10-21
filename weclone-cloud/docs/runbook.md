# 运维手册（已验证跑通的流程）

> 目标：微信小号 `kabuki` 收到消息 → LangBot → 本机分身（人设）→ DeepSeek → 回微信。

## 架构

```
微信 ──协议──> WeChatPadPro(9090) ──WS──> LangBot(5300) ──HTTP──> 分身(8010) ──> DeepSeek
```

| 组件 | 位置 | 端口 | 掉线影响 |
|---|---|---|---|
| 分身（人设代理） | 本机 Python | 8010 | 只影响回复 |
| LangBot | Docker | 5300 | 只影响回复 |
| WeChatPadPro | Docker | 9090 | **微信掉线需重扫** |
| MySQL / Redis | Docker | 13306 / 16381 | 存协议端数据 |

## 一、日常启动（三个窗口）

### 1. 分身

双击 `start-persona-docker.cmd`（**必须是 docker 版**，绑 `0.0.0.0`）。

横幅确认两行：

```
persona API : http://0.0.0.0:8010/v1
serve key : local-key
```

密钥从 `secrets.local.cmd` 自动读取，不用手输。

### 2. 微信协议端

```powershell
cd wechatpad-src
docker compose up -d
```

### 3. LangBot

```powershell
docker compose -f langbot-src\docker\docker-compose.yaml up -d
```

**顺序建议**：分身 → 协议端 → LangBot。LangBot 启动时会立刻连 9090，协议端没起来它就报
`Max retries exceeded ... /login/GetLoginStatus`，此时去 UI 关开一次开关即可。

## 二、重启电脑后的恢复

```powershell
# 1) 等 Docker 引擎（重启后 Docker Desktop 不会自动就绪）
for ($i = 1; $i -le 60; $i++) {
  docker info *> $null
  if ($LASTEXITCODE -eq 0) { "docker ready"; break }
  Start-Sleep -Seconds 5
}

# 2) 三个服务
cd C:\Users\Administrator\Documents\deepseek-harness\default-workspace\weclone-cloud
Start-Process ".\start-persona-docker.cmd"
cd wechatpad-src; docker compose up -d; cd ..
docker compose -f langbot-src\docker\docker-compose.yaml up -d
```

然后：

1. 重扫微信（见第三节）；
2. 登录 <http://localhost:5300>（后台会话也会丢）；
3. 更新 Token → 开开关。

## 三、微信重新登录（每次重启协议端都要做）

**旧授权码不能复用**：它会返回 `-3 该链接已绑定微信号`，必须生成新的。

```powershell
$r = Invoke-RestMethod "http://127.0.0.1:9090/admin/GenAuthKey1?key=fdc010310" `
  -Method Post -ContentType "application/json" -Body (@{ Count = 1; Days = 365 } | ConvertTo-Json)
$token = $r.Data.authKeys[0]
"TOKEN = $token"

$q = Invoke-RestMethod "http://127.0.0.1:9090/login/GetLoginQrCodeNew?key=$token" `
  -Method Post -ContentType "application/json" -Body (@{ Proxy = ""; Check = $false } | ConvertTo-Json)
Start-Process $q.Data.QrCodeUrl
```

扫码后确认：

```powershell
Invoke-RestMethod "http://127.0.0.1:9090/login/GetLoginStatus?key=$token" | ConvertTo-Json -Depth 5
```

看到 `"loginState": 1` 才算成功。把新 `TOKEN` 填进 LangBot 的 WeChatPad 配置。

> 异地使用必须给 `Proxy` 填同城 socks5（`socks5://user:pass@ip:port`），否则新号极易掉线/封号。

## 四、LangBot 配置（一次性，已配好，供重装参考）

| 字段 | 值 |
|---|---|
| WeChatPad_ERL | `http://host.docker.internal:9090` |
| WeChatPad_Ws | `ws://host.docker.internal:9090/ws` |
| Admin_Key | `fdc010310` |
| Token | 每次登录后的新授权码 |
| wxid | 可留 `auto`，运行时自动覆盖 |

模型（流水线里选）：

| 字段 | 值 |
|---|---|
| 请求 URL | `http://host.docker.internal:8010/v1` |
| API Key | `local-key` |
| 模型 | `deepseek-flash` |
| 工具使用 | 关 |
| 系统提示词 | 留空（服务端强制注入人设） |

## 五、项目内的两处本地补丁

它们通过 `langbot-src/docker/docker-compose.yaml` 的 bind mount 生效，**改完必须
`docker compose up -d --force-recreate`**，`docker restart` 不读新挂载。

| 补丁 | 位置 | 作用 |
|---|---|---|
| `event = None` | `src/langbot/pkg/platform/sources/wechatpad.py:591` | 修消息解析失败时的 `UnboundLocalError` |
| `_is_duplicate()` | 同上，`ws_message` 开头 | 按 `new_msg_id` 去重，修协议端重连重放导致的重复回复 |

验证：

```powershell
docker exec langbot grep -c "_is_duplicate" /app/src/langbot/pkg/platform/sources/wechatpad.py
```

应输出 `2`（定义 + 调用）。挂载确认：

```powershell
docker inspect langbot --format "{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}"
```

## 六、拟人节奏与防封

`cloud.example.jsonc` / `cloud.jsonc` 的 `serve` 段：

```jsonc
"reply_delay_min": 1.5,   // 回复前随机停顿下限（秒）
"reply_delay_max": 4.0,   // 上限
```

代理在调用上游前 `sleep(random.uniform(min, max))`，不再秒回。

另外建议在 LangBot 流水线里：

- 关闭「分段回复 / 多条消息」，避免一句话拆成好几条发；
- 关闭「错误提示推送给用户」，否则用户会看到 `Agent runner temporarily unavailable`；
- 并发改成 1（同一会话串行）。

协议端本身在风控雷达上：小号、低频率、不加人、不群发。

## 七、故障速查

| 现象 | 原因 | 处理 |
|---|---|---|
| `failed to connect to the docker API ... dockerDesktopLinuxEngine` | Docker Desktop 没启动 | 开始菜单启动，等 `docker info` 成功 |
| `Max retries exceeded ... host.docker.internal:9090` | 先起 LangBot、后起协议端 | 起来后关开机器人开关 |
| `HTTP 500 Internal Server Error`（分身） | 没设 DeepSeek key | 确认 `secrets.local.cmd` 有 `WECLONE_API_KEY`，重启分身 |
| `401 invalid api key`（分身） | serve key 不匹配 | 与横幅 `serve key :` 保持一致 |
| 微信回复带 `<think>` 英文 | 没走本机代理，或思考没关 | 确认流水线模型是 weclone 那个；代理已强制 `thinking: disabled` 并剥离 `reasoning_content` |
| 一条消息回好几条 | 协议端重放 + 适配器没去重 | 确认 `_is_duplicate` 补丁已挂载（`grep -c` = 2） |
| 扫码返回 `-3 该链接已绑定微信号` | 授权码已绑定旧会话 | 重新 `GenAuthKey1` 生成新码 |
| 微信突然掉线 | 重启了 wechatpad 容器 / 手机端退出登录 | 重扫；**别在手机上点"退出登录"** |

## 八、安全

- `secrets.local.cmd` 存 DeepSeek key，**明文**，已在 `.gitignore`；
- 这把 key 在聊天里出现过，建议在 <https://platform.deepseek.com/api_keys> 换新后回填；
- `ADMIN_KEY=fdc010310` 是镜像默认值，等同无密码，**别把 9090 暴露公网**；
- 分身监听 `0.0.0.0:8010`，仅局域网可见，用完关窗口。
