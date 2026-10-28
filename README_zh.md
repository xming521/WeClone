![download](https://github.com/user-attachments/assets/cd4a87c6-1649-4ce5-bce8-bd5b08b278de)
<h3 align="center">绘制你的人生画像，让助理懂你，让分身像你</h3>


<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/xming521/WeClone?style=for-the-badge&logo=github&label=Stars&logoColor=white&color=ffda65)](https://github.com/xming521/WeClone/stargazers)
[![GitHub release](https://img.shields.io/github/v/release/xming521/WeClone?style=for-the-badge&logo=github&label=Release&logoColor=white&color=06d094)](https://github.com/xming521/WeClone/releases)
<a href="https://qm.qq.com/cgi-bin/qm/qr?k=wNdgbOVT6oFOJ2wlMLsolUXErW9ESLpk&jump_from=webapi&authKey=z/reOp6YLyvR4Tl2k2nYMsLoMC3w9/99ucgKMX0oRGlxDV/WbYnvq2QxODoIkfxn" target="_blank" style="text-decoration: none;">
  <img src="https://img.shields.io/badge/QQ群-708067078-12B7F5?style=for-the-badge&logo=qq&logoColor=white" alt="WeClone①" title="WeClone①">
</a>
[![小红书](https://img.shields.io/badge/WeClone-FE2C55?style=for-the-badge&logo=xiaohongshu&logoColor=white)](https://www.xiaohongshu.com/user/profile/628109730000000021029de4)
[![Twitter](https://img.shields.io/badge/Twitter-@weclone567-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/weclone567)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/+JEdak4m0XEQ3NGNl)

<a href="https://hellogithub.com/repository/12ab209b56cb4cfd885c8cfd4cfdd53e" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=12ab209b56cb4cfd885c8cfd4cfdd53e&claim_uid=RThlPDoGrFvdMY5" alt="Featured｜HelloGitHub" style="width: 150px; height: 28px;" /></a>
<a href="https://trendshift.io/repositories/13759" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13759" alt="xming521%2FWeClone | Trendshift" style="width: 220px; height: 50px;" /></a>
</div>

<p align="center">
简体中文｜
  <a href="https://github.com/xming521/WeClone/blob/master/README.md" target="_blank">English</a>｜
  <a href="https://www.weclone.love/" target="_blank"> 项目主页 </a> ｜
  <a href="https://docs.weclone.love/zh/docs/introduce/what-is-weclone.html" target="_blank"> 项目文档 </a>
</p>

WeClone 从你的数字生活中提炼身份、目标、偏好与近况，按生活板块整理成一份可审核、可追溯、按需授权的人生画像，并分析你的说话风格。画像可以作为 AI 助理的长期记忆，也将驱动具有你个人风格的数字分身。目前支持聊天记录，更多数据源持续接入中。

## ✨ 核心功能

- 🧠 **MCP 接入 AI 助理**：Claude Code、Codex 等助手可通过 MCP 读取你的画像与板块总结，开放范围按生活板块和隐私级别精细控制。全程基于 API 模型，无需 GPU。
- 🎭 **专属数字分身**：基于聊天记录微调开源模型，并通过 AstrBot、LangBot 部署至 Telegram、Discord、Slack、飞书等平台。训练数据经过隐私信息过滤。
- 🗣️ **说话风格分析**：覆盖词汇、句式、语气、情绪、修辞、互动节奏等十大类、179 项风格特征。
- 🗺️ **交互式人生画像**：画像按生活板块与主题逐层组织，每条事实均可追溯原文，支持逐条审核与修改。
- 🔒 **隐私优先**：默认本地加密存储，每条事实标注隐私级别，导出与授权范围由你决定。
- 🤖 **Agent 分身**（开发中）：无需训练，由大模型结合画像、记忆与说话风格实时驱动。

> [!NOTE]
> WeClone 现在不只做微调分身。原有的微调流程完整保留，见[微调分身](docs/zh/finetune.md)。

### 数据源平台适配

人生画像只读取文字消息，下表是微调分身支持的消息类型。导出和配置方法见[准备数据](docs/zh/data.md)。

| 平台 | 文字 | 图片 | 语音 | 视频 | 动画表情 | 链接（分享） | 引用 | 转发 | 位置 | 文件 |
|------|------|------|------|------|----------|-----------|------|------|------|------|
| Telegram | ✅ | ✅ | ❌ | ❌ | ⚠️ 转为 Emoji | ❌ | ❌ | ✅ | ✅ | ❌ |
| WhatsApp | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| Discord | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| Slack | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |

## 📰 近期更新

- [26/10] 人生画像：`build-profile` 一条命令完成画像抽取、整理与板块总结，并在网页上逐条审核
- [26/10] 助手接入：通过 MCP 向 Claude Code、Codex 等助手提供人生画像
- [26/10] 数据默认加密存储，聊天机器人通过令牌调用分身
- [25/07/10] 数据源增加 Telegram
- [25/06/05] 支持图片模态数据微调

## 🚀 快速开始

### 1. 安装

需要 Python 3.12，推荐使用 [uv](https://docs.astral.sh/uv/)：

```bash
git clone https://github.com/xming521/WeClone.git && cd WeClone
uv venv .venv --python=3.12
source .venv/bin/activate   # Windows 下执行 .venv\Scripts\activate
uv pip install -e .         # 国内镜像：-i https://pypi.tuna.tsinghua.edu.cn/simple/
cp settings.template.jsonc settings.jsonc
```

所有配置都在 `settings.jsonc` 中。基础安装即可绘制人生画像和接入助理。画像默认通过 [Codex CLI](https://github.com/openai/codex) 调用模型，需要先安装并登录，也可以改用任意 OpenAI 兼容 API，见[安装与配置](docs/zh/installation.md)。

### 2. 准备数据

目前支持 Telegram 聊天记录：用 Telegram Desktop 以 JSON 格式导出，把 `ChatExport_*` 文件夹放进 `dataset/telegram/`，在 `settings.jsonc` 中把 `platform` 设为 `telegram`，并在 `telegram_args.my_id` 填写你的用户 ID。详见[准备数据](docs/zh/data.md)。

### 3. 绘制人生画像，接入 AI 助理

```bash
weclone-cli build-profile --purpose assistant
weclone-cli server
```

打开 http://127.0.0.1:5175 逐条审核画像，在「板块总结」页确认总结，再到「助手接入」页生成令牌，按页面上的命令配置 Claude Code 或 Codex。详见[人生画像与助手接入](docs/zh/profile.md)。

### 4. 微调数字分身

需要 NVIDIA GPU，默认的 Qwen2.5-7B-Instruct 以 LoRA 微调约需 16GB 显存。按[微调分身](docs/zh/finetune.md)安装依赖、下载模型后执行：

```bash
weclone-cli make-dataset                     # 生成训练数据，同时过滤隐私信息
weclone-cli train-sft                        # LoRA 微调
weclone-cli server --inference --port 8005   # 启动 OpenAI 兼容接口
```

训练好的分身可以通过 AstrBot、LangBot 接入聊天平台，见[接入聊天机器人](docs/zh/chatbot.md)。

## 🔒 隐私与数据安全

- **抽取时调用模型服务**：绘制画像时，聊天内容会发送给你配置的模型服务，开始前会请你确认。已抽取过的内容读取本地缓存，不会再次发送。
- **默认加密存储**：聊天快照、画像、训练数据和微调模型加密保存在本地，网页和命令行共用一个密码。
- **按需授权**：每条事实带有隐私级别（公开、普通、私密、敏感），助理能读取哪些板块、到哪个级别由你在网页上设定。
- **训练数据去敏**：微调前通过 Microsoft Presidio 和自定义禁用词过滤隐私信息，但不能保证 100% 识别。

详见[数据安全](docs/zh/security.md)。

## 📚 文档

| 文档 | 内容 |
|---|---|
| [安装与配置](docs/zh/installation.md) | 按用途安装、模型后端、说话风格与微调依赖 |
| [准备数据](docs/zh/data.md) | 支持的数据源、Telegram 导出 |
| [人生画像与助手接入](docs/zh/profile.md) | build-profile、网页审核、板块总结、MCP |
| [微调分身](docs/zh/finetune.md) | 硬件、模型下载、训练数据、训练与推理 |
| [接入聊天机器人](docs/zh/chatbot.md) | 机器人令牌、AstrBot、LangBot |
| [数据安全](docs/zh/security.md) | 存储模式、密码、密钥与令牌、请求日志 |

更完整的教程、配置说明和常见问题见[文档站](https://docs.weclone.love/zh/)。

## 📌 路线图

- [x] 人生画像：画像抽取、审核与板块总结
- [x] 通过 MCP 为 AI 助理提供长期记忆
- [ ] Agent 分身：结合画像、记忆与说话风格驱动分身
- [ ] 支持更多数据源
- [ ] 更丰富的上下文：包括上下文对话、聊天对象信息、时间等
- [ ] 支持多模态：已支持图片
- [ ] 数据增强
- [ ] 支持COT思考

## 💬 问题解决

- 先查阅[官方文档 FAQ](https://docs.weclone.love/zh/docs/introduce/FAQ.html)，也可以使用 [DeepWiki](https://deepwiki.com/xming521/WeClone) 解决问题。
- 欢迎加入页面顶部的 QQ 群和 Telegram 群交流。社群内有部署好的 Qwen2.5-VL 32B Bot，可以体验微调效果。

## ❤️ 贡献代码

欢迎提交 Issue 和 Pull Request，也欢迎帮助审核 PR。新功能请先通过 Issue 讨论。开发环境、测试和前端构建见[贡献指南](docs/CONTRIBUTING.md)。

## 🙏 致谢
BUPT VCIS Lab的支持
感谢以下代码贡献者和社区里其他成员的贡献

<a href="https://github.com/xming521/WeClone/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xming521/WeClone" />
</a>

同时本项目受益于[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)、[AstrBot](https://github.com/AstrBotDevs/AstrBot)、[LangBot](https://github.com/RockChinQ/LangBot)等优秀开源项目。

## ⚠️ 免责声明
> [!CAUTION]
> **本项目仅供学习、研究和实验用途，用于生产环境存在较大风险，请谨慎评估。请勿用于非法用途，后果自负。**


<details>
<summary>点击查看免责条款</summary>

### 1. 使用风险自担
- 用户在使用本项目时，应充分理解并承担所有相关风险
- **本项目作者不对因使用本项目而产生的任何直接或间接损失承担责任**
- 包括但不限于：数据丢失、经济损失、法律纠纷、个人名誉损害、社会关系影响、心理创伤、职业发展受阻、商业信誉受损等

### 2. 生产环境风险警告
- **用于商业用途或对外提供服务需自行承担全部风险**
- 生产环境使用可能导致的所有后果（包括但不限于服务中断、数据安全问题、用户投诉、法律责任等）完全由用户承担
- **建议在生产环境使用前进行充分的测试、验证和风险评估**

### 3. 模型输出不可靠性
- 微调后的模型可能产生不准确、有害或误导性的内容
- 模型输出不代表真实人物的观点或意图
- 用户应对模型输出进行人工审核和验证

### 4. 数据安全与隐私
- 用户应确保上传的聊天记录等数据符合相关法律法规
- 用户应获得**数据相关人员的适当授权**
- 本项目不对**数据泄露或隐私侵犯**承担责任

### 5. 法律合规
- **用户应确保使用本项目符合当地法律法规**
- 涉及人工智能、数据保护、知识产权等相关法律
- **违法使用造成的后果由用户承担**

### 6. 技术支持限制
- 本项目按"现状"提供，不提供任何明示或暗示的保证
- 作者不承诺提供持续的技术支持或维护
- 不保证项目的稳定性、可靠性或适用性

## 使用建议

### 强制性Bot身份标识
**使用本项目生成的数字分身时，强烈建议：**
- 在每次对话开始时明确标识为"AI Bot"或"数字分身"
- 在用户界面显著位置标注"此为AI生成内容"
- 避免让用户误认为是真实人类在对话，从而造成风险

### 风险评估建议

如确需在生产环境使用，建议：
1. 进行全面的安全性测试
2. 建立完善的内容审核机制
3. 制定应急响应预案
4. 购买相应的保险保障
5. 咨询法律专业人士意见


本免责声明可能随项目更新而修订，用户应定期查看最新版本。继续使用本项目即表示同意最新的免责声明条款。

**一旦您下载、克隆、修改、分发或以任何方式使用本项目的代码或模型，即表示您已完整阅读、理解并同意无条件接受本免责声明的全部条款。**

</details>

**请用户慎重阅读并理解本免责声明的所有内容，确保在使用本项目时严格遵守相关规定。**
<br>  

## ⭐ Star History
> [!TIP] 
> 如果本项目对您有帮助，或者您关注本项目的未来发展，请给项目 Star，谢谢 

<div align="center">

[![Star History Chart](https://star-history.dera.page/svg?repos=xming521/WeClone&type=Date)](https://star-history.dera.page/#xming521/WeClone&Date)

</div>


<div align="center"> 克隆我们，保留灵魂的芬芳 </div>
