# 维护手册

长期维护这个子项目需要盯三类东西：**上游代码变化**、**第三方项目漂移**、**本项目的两处补丁**。

## 一、测试过的版本矩阵

升级任何一项之前，先对照这张表；升级后按第三节逐项复验。

| 组件 | 测试版本 | 固定方式 |
|---|---|---|
| WeClone | `master@d4af5ba`（2026-10-18） | 子目录跟随父仓库，无需固定 |
| LangBot | **v4.11.1**（镜像 `rockchin/langbot:latest`，拉取时即 4.11.1） | `start-langbot-docker.cmd` 克隆默认分支；镜像 tag 建议改成具体版本 |
| WeChatPad-Docker | commit **`c6d1b7a`** | `tools/prepare_wechatpad.ps1` 的 `-Ref`，默认钉在该提交 |
| Python | 3.12.x | `.venv` |
| Docker Desktop | 4.x（WSL2 后端） | — |

> 镜像 `rockchin/langbot:latest` 会漂。要真正固定，把 `langbot-src/docker/docker-compose.yaml` 里的
> `rockchin/langbot:latest` 改成 `rockchin/langbot:4.11.1`（如果该 tag 存在），或者记录拉取时的 digest：
> `docker inspect langbot --format '{{index .RepoDigests 0}}'`。

## 二、上游 WeClone 变了怎么办

子项目**不改 `weclone/` 任何一行**，所以上游更新绝大多数情况下直接 rebase 即可：

```powershell
cd D:\github\WeClone
git remote add upstream https://github.com/xming521/WeClone.git   # 只需一次
git fetch upstream
git rebase upstream/master
```

需要人工看一眼的只有两件事：

| 上游改动 | 影响 | 检查点 |
|---|---|---|
| `weclone/data/qa_generator.py` 里 `csv_folder` | 数据目录变化 | `docs/install.md` 第 2.1 节的路径表 |
| `make-dataset` 产物格式（`sft-my.json` 的字段） | `score` / `persona` 读不了 | 跑一次 `score --dry-run` |

`docs/architecture.md` 里记录了「本项目 ↔ WeClone」的对应关系，改上游行为时同步更新它。

## 三、第三方漂移：LangBot 与 WeChatPadPro

这两处最容易在半年后失效。升级后必须逐项复验：

### 1. WeChatPad 适配器补丁是否还在

补丁通过 bind mount 注入，文件在 `langbot-src/src/langbot/pkg/platform/sources/wechatpad.py`。

```powershell
docker exec langbot grep -c "_is_duplicate" /app/src/langbot/pkg/platform/sources/wechatpad.py
```

- 输出 `2`：补丁在位；
- 输出 `0`：**容器重建过或挂载丢了**，检查 `docker-compose.yaml` 里那条 `../src/.../wechatpad.py` 挂载。

升级 LangBot 之后如果页面上行号变了（补丁里记录的是 `591` 行），说明上游改了这个文件，需要：
1. 重新对照源码打一次补丁（`event = None` + `_is_duplicate`）；
2. 更新 `docs/runbook.md` 第五节的行号。

### 2. compose 的 Windows 挂载修复是否仍然必要

```powershell
Select-String -Path "langbot-src\docker\docker-compose.yaml" -Pattern "\$\{PWD\}/data/box"
```

- 有输出：上游还没修，`tools/fix_langbot_compose.py` 需要继续跑；
- 无输出：上游修了，可以删掉这个脚本调用。

### 3. WeChatPadPro 接口形状

适配器依赖这几个接口，升级协议端后逐个试：

| 接口 | 用途 | 失败现象 |
|---|---|---|
| `POST /admin/GenAuthKey1` | 换授权码 | `KeyError: 'authKeys'`（本项目已改为 `Data.authKeys[0]`） |
| `POST /login/GetLoginQrCodeNew` | 扫码 | 返回 `-3 该链接已绑定微信号` |
| `GET /login/GetLoginStatus` | 判断在线 | `loginState != 1` |
| `WS /ws/GetSyncMsg?key=` | 收消息 | 日志 `WebSocket error 1006` |

`tools/prepare_wechatpad.ps1` 的 `-Ref` 默认钉在 `c6d1b7a`。要升级：

```powershell
powershell -File tools\prepare_wechatpad.ps1 -Ref main -Force
```

升级后重跑第三节的三条命令，并把新的提交号写回本文的版本矩阵。

## 四、本项目自身的回归检查

改完代码后，最小验证集合：

```powershell
cd <仓库根>
.\.venv\Scripts\python.exe -m pytest -q            # 单测（不联网）
.\.venv\Scripts\python.exe -m weclone_cloud.cli doctor
```

再加一条「真实链路」抽查（会花几分钱）：

```powershell
$h = @{ Authorization = "Bearer local-key" }
$b = @{ model = "deepseek-flash"; messages = @(@{ role = "user"; content = "在吗" }); stream = $false } | ConvertTo-Json -Depth 5
$r = Invoke-WebRequest http://127.0.0.1:8010/v1/chat/completions -Method Post -Headers $h -ContentType "application/json" -Body $b -UseBasicParsing
([Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray()) | ConvertFrom-Json).choices[0].message.content
```

验收三条（`docs/install.md` 第 8 节同款）：

1. 微信只回**一条**，2–4 秒后到；
2. 回复是短句、**没有** `<think>` 英文块；
3. 分身窗口出现 `POST /v1/chat/completions 200`。

## 五、发布流程（两个仓库都要推）

本项目同时存在两处：

| 位置 | 用途 |
|---|---|
| `lhongjie70-sudo/weclone-cloud` | 独立仓库，给只想跑的人 |
| `xming521/WeClone` 的 `weclone-cloud/` 子目录（PR #253） | 给上游用 |

改完代码后的顺序：

```powershell
# 1) 独立仓库
cd <独立仓库>
git add -A
git commit -m "fix: ..."
git push

# 2) 同步到 fork 的子目录并更新 PR
robocopy <独立仓库> "D:\github\WeClone\weclone-cloud" /E /XD ".git" /NFL /NDL /NJH /NJS
cd D:\github\WeClone
git add weclone-cloud
git commit -m "fix(weclone-cloud): ..."
git push origin feat/weclone-cloud
```

> 注意：子目录里的 `.gitattributes` 是 `* text=auto`（全 LF）。不要恢复 `*.cmd eol=crlf`，
> 否则 pre-commit.ci 的 `mixed-line-ending` 会再次失败。

## 六、已知的长期风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| DeepSeek 改 API（模型名/思考参数） | 打分、对话全挂 | `config.api.model` / `thinking` 可配置；`doctor` 先查 |
| DeepSeek 涨价 | 成本 | `api.price_*` 只是估算，改用 `--dry-run` 先看 token |
| 微信风控升级 | 小号被封 | 降低频率、别群发；这是使用风险，不是代码问题 |
| LangBot 大版本（5.x） | 适配器接口重写 | 锁版本，等本项目适配后再升 |
| WeChatPadPro 停更 | 微信接不进来 | 备选：LangBot 的 OpenClaw / Satori 适配器 |

## 七、给贡献者的最小约定

- 只改 `weclone-cloud/` 下的文件；不碰 `weclone/`；
- 新增依赖写进 `pyproject.toml`，不要 import 未声明的包；
- 注释和 docstring 用**英文**（上游规范），用户文档用中文；
- 提交前跑 `uvx pre-commit run --files <改动文件>`；
- 涉及微信链路的改动，在 PR 里贴一次第四节的三条验收结果。
