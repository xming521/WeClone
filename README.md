![download](https://github.com/user-attachments/assets/cd4a87c6-1649-4ce5-bce8-bd5b08b278de)

<h3 align="center">Build a portrait of your life. Assistants that know you. Clones that talk like you.</h3>

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/xming521/WeClone?style=for-the-badge&logo=github&label=Stars&logoColor=white&color=ffda65)](https://github.com/xming521/WeClone/stargazers)
[![GitHub release](https://img.shields.io/github/v/release/xming521/WeClone?style=for-the-badge&logo=github&label=Release&logoColor=white&color=06d094)](https://github.com/xming521/WeClone/releases)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/+JEdak4m0XEQ3NGNl)
[![Twitter](https://img.shields.io/badge/Twitter-@weclone567-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/weclone567)
[![小红书](https://img.shields.io/badge/WeClone-FE2C55?style=for-the-badge&logo=xiaohongshu&logoColor=white)](https://www.xiaohongshu.com/user/profile/628109730000000021029de4)
<a href="https://qm.qq.com/cgi-bin/qm/qr?k=wNdgbOVT6oFOJ2wlMLsolUXErW9ESLpk&jump_from=webapi&authKey=z/reOp6YLyvR4Tl2k2nYMsLoMC3w9/99ucgKMX0oRGlxDV/WbYnvq2QxODoIkfxn" target="_blank" style="text-decoration: none;">
  <img src="https://img.shields.io/badge/QQ群-708067078-12B7F5?style=for-the-badge&logo=qq&logoColor=white" alt="WeClone①" title="WeClone①">
</a>


<a href="https://hellogithub.com/repository/12ab209b56cb4cfd885c8cfd4cfdd53e" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=12ab209b56cb4cfd885c8cfd4cfdd53e&claim_uid=RThlPDoGrFvdMY5" alt="Featured｜HelloGitHub" style="width: 150px; height: 28px;" /></a>
<a href="https://trendshift.io/repositories/13759" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13759" alt="xming521%2FWeClone | Trendshift" style="width: 220px; height: 50px;" /></a>
</div>

<p align="center">
  <a href="https://github.com/xming521/WeClone/blob/master/README_zh.md" target="_blank">简体中文</a>｜
  English｜
  <a href="https://www.weclone.love/" target="_blank"> Project Homepage </a> ｜
  <a href="https://docs.weclone.love/en/docs/introduce/what-is-weclone" target="_blank"> Documentation </a> 
</p>

WeClone distills your identity, goals, preferences and current situation from your digital life into a portrait of your life, organized into domains such as work, family and health, that you can review, trace to its sources and share on your own terms. It also analyzes your speaking style. The portrait works as long-term memory for your AI assistants and will power clones with your personal style. Chat history is supported today, with more data sources coming.

## ✨ Core Features

- 🗺️ **Interactive life portrait**: your portrait is organized by domain and topic. Every fact traces back to its source and can be reviewed and edited. Extraction runs through Codex or any OpenAI-compatible API.
- 🎭 **Your own AI clone**: fine-tune an open model on your chat history and deploy it to Telegram, Discord, Slack, Feishu and more through AstrBot or LangBot. Training data is filtered for personal information.
- 🧠 **Connect AI assistants**: Claude Code, Codex and other assistants read your portrait and domain summaries over MCP. You choose which domains and privacy levels they can read.
- 🗣️ **Speaking style analysis**: 179 style features in ten categories, covering vocabulary, syntax, tone, emotion, rhetoric and conversational rhythm.
- 🤖 **Agent clone** (in development): no training required. A large model drives the clone in real time from your portrait, memories and speaking style.
- 🔒 **Privacy first**: encrypted local storage by default, a privacy level on every fact, and full control over what is exported and shared.

> [!NOTE]
> WeClone now goes beyond fine-tuned clones. The original fine-tuning workflow remains fully supported, see [Fine-tuned clone](docs/en/finetune.md).
>
> The portrait and its extraction prompts are currently in Chinese, and English chats have not been tested yet. The web UI is available in English and Chinese.

## 📰 Recent Updates

- [26/10] Life portrait: `build-profile` extracts, organizes and summarizes your portrait in one command, ready for fact-by-fact review in the web UI
- [26/10] Assistant access: Claude Code, Codex and other assistants read your portrait over MCP
- [26/10] Encrypted storage by default, and token access for chat bots
- [25/07/10] Telegram added as a data source
- [25/06/05] Fine-tuning on image data

## 📥 Supported Data Sources

The life portrait reads text messages only. The table shows the message types the fine-tuned clone supports. See [Preparing data](docs/en/data.md) for exporting and setup.

| Platform | Text | Images | Voice | Video | Animated Emojis/Stickers | Links (Sharing) | Quote | Forward | Location | Files |
|----------|------|--------|-------|-------|-----------------|-----------------|-------|---------|----------|-------|
| Telegram | ✅ | ✅ | ❌ | ❌ | ⚠️ Converted to Emoji | ❌ | ❌ | ✅ | ✅ | ❌ |
| WhatsApp | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| Discord | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| Slack | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |

Chats from other platforms can be converted to WeClone's generic CSV format and placed in `dataset/csv/`. See [Preparing Data](https://docs.weclone.love/en/docs/deploy/export-chat-history#generic-csv-format) on the docs site for the format.

## 🚀 Quick Start

### 1. Install

Python 3.12 is required. [uv](https://docs.astral.sh/uv/) is recommended:

```bash
git clone https://github.com/xming521/WeClone.git && cd WeClone
uv venv .venv --python=3.12
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e .
cp settings.template.jsonc settings.jsonc
```

All settings live in `settings.jsonc`. The base install is enough to build your portrait and connect assistants. By default the portrait is extracted through the [Codex CLI](https://github.com/openai/codex), which must be installed and signed in. Any OpenAI-compatible API works as well, see [Installation and configuration](docs/en/installation.md).

### 2. Prepare your data

Telegram chats are supported today. Export them as JSON with Telegram Desktop, put the `ChatExport_*` folders in `dataset/telegram/`, then set `platform` to `telegram` and `telegram_args.my_id` to your user ID in `settings.jsonc`. Chats from other platforms can be converted to the generic CSV format and placed in `dataset/csv/`. See [Preparing data](docs/en/data.md).

### 3. Build your portrait and connect assistants

```bash
weclone-cli build-profile --purpose assistant
weclone-cli server
```

Open http://127.0.0.1:5175 to review your portrait fact by fact, approve the summaries on the Summaries page, then generate a token on the Assistants page and run the commands it shows for Claude Code or Codex. See [Life portrait and assistants](docs/en/profile.md).

### 4. Fine-tune a clone

An NVIDIA GPU is required. The default Qwen2.5-7B-Instruct with LoRA needs about 16GB of VRAM. Install the dependencies and download the model as described in [Fine-tuned clone](docs/en/finetune.md), then run:

```bash
weclone-cli make-dataset                     # build training data, filtering personal information
weclone-cli train-sft                        # LoRA fine-tuning
weclone-cli server --inference --port 8005   # OpenAI-compatible API
```

Connect the trained clone to chat platforms through AstrBot or LangBot, see [Chat bots](docs/en/chatbot.md).

> [!TIP]
> **We're looking for interesting examples of native English speakers chatting with WeClone! Feel free to share them with us on Twitter.**

## 🔒 Privacy and Security

- **Model services during extraction**: while building your portrait, chat content is sent to the model service you configured, after you confirm. Content already extracted is read from a local cache and not sent again.
- **Encrypted by default**: the chat snapshot, portrait, training data and fine-tuned model are stored encrypted on your machine, with one password shared by the web UI and the command line.
- **Access on your terms**: every fact carries a privacy level (public, normal, private, sensitive), and you set in the web UI which domains assistants can read and up to which level.
- **Training data filtering**: Microsoft Presidio and your own blocked words remove personal information before fine-tuning, though 100% detection is not guaranteed.

See [Security](docs/en/security.md).

## 📚 Documentation

| Document | Covers |
|---|---|
| [Installation and configuration](docs/en/installation.md) | What to install, model backend, speaking style and fine-tuning dependencies |
| [Preparing data](docs/en/data.md) | Supported sources, Telegram export, generic CSV format |
| [Life portrait and assistants](docs/en/profile.md) | build-profile, web review, domain summaries, MCP |
| [Fine-tuned clone](docs/en/finetune.md) | Hardware, model download, training data, training and inference |
| [Chat bots](docs/en/chatbot.md) | Bot token, AstrBot, LangBot |
| [Security](docs/en/security.md) | Storage modes, password, keys and tokens, request logs |

Full tutorials, configuration reference and FAQ are on the [documentation site](https://docs.weclone.love/en/).

## 📌 Roadmap

- [x] Life portrait: extraction, review and domain summaries
- [x] Long-term memory for AI assistants over MCP
- [ ] Agent clone driven by your portrait, memories and speaking style
- [ ] Support more data sources
- [ ] Richer context: including contextual conversations, chat participant information, time, etc.
- [ ] Multimodal support: image support already implemented
- [ ] Data augmentation
- [ ] COT (Chain of Thought) thinking support

## 💬 Troubleshooting

- Start with the [official FAQ](https://docs.weclone.love/en/docs/introduce/FAQ). [DeepWiki](https://deepwiki.com/xming521/WeClone) can also help.
- Join the Telegram group linked at the top of this page.

## ❤️ Contributing

Issues and pull requests are welcome, and so is help reviewing PRs. Please discuss new features in an issue first. See the [contributing guide](docs/CONTRIBUTING.md#english) for the development environment, tests and frontend builds.

## 🙏 Acknowledgments

Thanks to the following code contributors and other community members for their contributions

<a href="https://github.com/xming521/WeClone/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xming521/WeClone" />
</a>

This project also benefits from excellent open source projects such as [PyWxDump](https://github.com/xaoyaoo/PyWxDump), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [AstrBot](https://github.com/AstrBotDevs/AstrBot), [LangBot](https://github.com/RockChinQ/LangBot), and others.

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")


## ⚠️ Disclaimer
> [!CAUTION]
> **This project is for learning, research and experimental purposes only. There are significant risks in using it for production environments, please assess carefully. Do not use for illegal purposes, consequences are at your own risk.**


<details>
<summary>Click to view disclaimer terms</summary>

### 1. Use at Your Own Risk
- Users should fully understand and bear all related risks when using this project
- **The project authors are not responsible for any direct or indirect losses arising from the use of this project**
- Including but not limited to: data loss, financial loss, legal disputes, personal reputation damage, social relationship impact, psychological trauma, career development obstacles, business reputation damage, etc.

### 2. Production Environment Risk Warning
- **Use for commercial purposes or providing external services requires bearing all risks yourself**
- All consequences that may result from production environment use (including but not limited to service interruption, data security issues, user complaints, legal liability, etc.) are entirely borne by the user
- **It is recommended to conduct thorough testing, verification and risk assessment before using in production environments**

### 3. Model Output Unreliability
- Fine-tuned models may produce inaccurate, harmful or misleading content
- Model outputs do not represent the views or intentions of real persons
- Users should conduct manual review and verification of model outputs

### 4. Data Security and Privacy
- Users should ensure that uploaded chat records and other data comply with relevant laws and regulations
- Users should obtain **appropriate authorization from data-related persons**
- This project is not responsible for **data leakage or privacy infringement**

### 5. Legal Compliance
- **Users should ensure that using this project complies with local laws and regulations**
- Involving artificial intelligence, data protection, intellectual property and other related laws
- **Users bear the consequences of illegal use**

### 6. Technical Support Limitations
- This project is provided "as is" without any express or implied warranties
- Authors do not promise to provide continuous technical support or maintenance
- No guarantee of project stability, reliability or applicability

## Usage Recommendations

### Mandatory Bot Identity Identification
**When using AI clones generated by this project, it is strongly recommended to:**
- Clearly identify as "AI Bot" or "AI Clone" at the beginning of each conversation
- Prominently mark "AI-generated content" in the user interface
- Avoid letting users mistake it for real human conversation, which could cause risks

### Risk Assessment Recommendations

If you must use in production environments, it is recommended to:
1. Conduct comprehensive security testing
2. Establish complete content review mechanisms
3. Develop emergency response plans
4. Purchase appropriate insurance coverage
5. Consult legal professionals for advice


This disclaimer may be revised with project updates, users should regularly check the latest version. Continuing to use this project indicates agreement with the latest disclaimer terms.

**Once you download, clone, modify, distribute or use the code or models of this project in any way, it indicates that you have fully read, understood and agreed to unconditionally accept all terms of this disclaimer.**

</details>

**Please carefully read and understand all contents of this disclaimer, ensuring strict compliance with relevant regulations when using this project.**
<br>  

## ⭐ Star History
> [!TIP] 
> If this project is helpful to you, or if you are interested in the future development of this project, please give the project a Star, thank you 

<div align="center">

[![Star History Chart](https://star-history.dera.page/svg?repos=xming521/WeClone&type=Date)](https://star-history.dera.page/#xming521/WeClone&Date)

</div>


<div align="center"> We clone, soul and all. </div>
