<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://docs.weclone.love/img/banner-dark.webp">
  <img alt="WeClone" src="https://docs.weclone.love/img/banner-light.webp">
</picture>

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

WeClone distills your identity, goals, preferences and current situation from your digital life into a life portrait, organized into domains such as work, family and health, that you can review, trace and share on your own terms. The portrait serves as long-term memory for your AI assistants and will power clones with your personal style.

<a href="https://docs.weclone.love/img/screenshots/atlas-light-en.webp">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://docs.weclone.love/img/screenshots/atlas-dark-en.webp">
    <img alt="Life portrait in the web UI" src="https://docs.weclone.love/img/screenshots/atlas-light-en.webp">
  </picture>
</a>

## ✨ Core Features

- 🗺️ **Interactive life portrait**: your portrait is organized by domain and topic. Every fact traces back to its source and can be reviewed and edited. Extraction runs through Codex or any OpenAI-compatible API.
- 🎭 **Your own AI clone**: fine-tune an open model on your chat history and deploy it to Telegram, Discord, Slack, Feishu and more through AstrBot or LangBot. Training data is filtered for personal information.
- 🧠 **Connect AI assistants**: Claude Code, Codex and other assistants read your portrait and domain summaries over MCP. You choose which domains and privacy levels they can read.
- 🗣️ **Speaking style analysis**: 179 style features in ten categories, covering vocabulary, syntax, tone, emotion, rhetoric and conversational rhythm.
- 🤖 **Agent clone** (in development): no training required. A large model drives the clone in real time from your portrait, memories and speaking style.
- 🔒 **Privacy first**: encrypted local storage by default, a privacy level on every fact, and full control over what is exported and shared.

<a href="https://docs.weclone.love/img/architecture/overview-light-en.webp">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://docs.weclone.love/img/architecture/overview-compact-dark-en.webp">
    <img alt="WeClone feature overview" src="https://docs.weclone.love/img/architecture/overview-compact-light-en.webp">
  </picture>
</a>

> [!NOTE]
> WeClone now goes beyond fine-tuned clones. The original fine-tuning workflow remains fully supported, see [Fine-tuned clone](docs/en/finetune.md).
>
> The portrait and its extraction prompts are currently in Chinese, and English chats have not been tested yet. The web UI is available in English and Chinese.

## 📰 Recent Updates

- [26/11/03] Life portrait: `build-profile` extracts, organizes and summarizes your portrait in one command, ready for fact-by-fact review in the web UI
- [26/11/03] Assistant access: Claude Code, Codex and other assistants read your portrait over MCP
- [26/11/03] Encrypted storage by default, and token access for chat bots
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
weclone-cli server
```

Open http://127.0.0.1:5175, set a password, and build your portrait on the Build portrait page, which checks the setup and shows the progress. Then review your portrait fact by fact, approve the summaries on the Summaries page, then generate a token on the Assistants page and run the commands it shows for Claude Code or Codex. See [Life portrait and assistants](docs/en/profile.md).

### 4. Fine-tune a clone

An NVIDIA GPU is required. The default Qwen2.5-7B-Instruct with LoRA needs about 16GB of VRAM. Install the dependencies and download the model as described in [Fine-tuned clone](docs/en/finetune.md), then run:

```bash
weclone-cli make-dataset                     # build training data, filtering personal information
weclone-cli train-sft                        # LoRA fine-tuning
weclone-cli server --inference --port 8005   # OpenAI-compatible API
```

Connect the trained clone to chat platforms through AstrBot or LangBot, see [Chat bots](docs/en/chatbot.md).

## 🔒 Privacy and Security

- **Model services during extraction**: while building your portrait, chat content is sent to the model service you configured, after you confirm. Content already extracted is read from a local cache and not sent again.
- **Encrypted by default**: the chat snapshot, portrait, training data and fine-tuned model are stored encrypted on your machine, with one password shared by the web UI and the command line.
- **Access on your terms**: every fact carries a privacy level (public, normal, private, sensitive), and you set in the web UI which domains assistants can read and up to which level.
- **Training data filtering**: Microsoft Presidio and your own blocked words remove personal information before fine-tuning, though 100% detection is not guaranteed.

See [Security](docs/en/security.md).

## 🏗️ Architecture

<details>
<summary>Show the architecture diagram: the modules, the data flow and the files each step writes. Click it for the full-size image.</summary>

<a href="https://docs.weclone.love/img/architecture/architecture-light-en.webp">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://docs.weclone.love/img/architecture/architecture-dark-en.webp">
    <img alt="WeClone architecture and implementation" src="https://docs.weclone.love/img/architecture/architecture-light-en.webp">
  </picture>
</a>

</details>

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

## 💬 Troubleshooting

- Start with the [official FAQ](https://docs.weclone.love/en/docs/introduce/FAQ). [DeepWiki](https://deepwiki.com/xming521/WeClone) can also help.
- Join the Telegram group linked at the top of this page.

## ❤️ Contributing

Issues and pull requests are welcome, and so is help reviewing PRs. Please discuss new features in an issue first. See the [contributing guide](docs/CONTRIBUTING.md#english) for the development environment, tests and frontend builds.

## 🙏 Acknowledgments

Thanks to BUPT VCIS Lab for its support, and to the following code contributors and other community members for their contributions.

<a href="https://github.com/xming521/WeClone/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xming521/WeClone" />
</a>

This project also benefits from excellent open source projects such as [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [AstrBot](https://github.com/AstrBotDevs/AstrBot), [LangBot](https://github.com/RockChinQ/LangBot), and others.

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")


## ⚠️ Disclaimer
> [!IMPORTANT]
> This project is intended for learning and research. Assess the risks yourself before using it in production or offering it as a service.

<details>
<summary>Click to view the disclaimer terms</summary>

- **Provided as is**: the project comes without warranties of any kind. The authors are not liable for any direct or indirect loss arising from its use and do not commit to ongoing maintenance or support.
- **Data and consent**: chat history contains other people's messages. Make sure that collecting and using the data complies with local laws and that the people involved have consented.
- **Portrait extraction by LLMs**: while your portrait is built, chat content, including the other side's messages, is sent to the model service you configured and handled under that service's data policy. Choose a provider you trust and check its data retention and training terms. Leave chats you don't want sent out of the data directory.
- **Portrait content**: the portrait collects and infers personal information about you, touches on the people around you, and may include sensitive topics such as health or relationships. Keep result directories and exported files safe, and review the portrait before giving assistants access.
- **Model output**: portraits, summaries and clone replies are generated by models. They may be inaccurate and do not represent the real views of you or anyone else.
- **AI disclosure**: when a clone talks to other people, make clear that it is an AI so that no one mistakes it for a real person.
- **Lawful use**: do not use this project for any unlawful purpose.

This disclaimer may be revised as the project evolves.

</details>

## ⭐ Star History
> [!TIP] 
> If this project is helpful to you, or if you are interested in the future development of this project, please give the project a Star, thank you 

<div align="center">

[![Star History Chart](https://star-history.dera.page/svg?repos=xming521/WeClone&type=Date)](https://star-history.dera.page/#xming521/WeClone&Date)

</div>


<div align="center"> We clone, soul and all. </div>
