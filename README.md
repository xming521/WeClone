![download](https://github.com/user-attachments/assets/5842e84e-004f-4afd-9373-af64e9575b78)
<h3 align="center">🚀 One-stop solution for creating your digital avatar from chat history 💡</h3>  

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
<a href="https://deepwiki.com/xming521/WeClone"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"  style="width: 134px; height: 23px;margin-bottom: 3px;"></a>
</div>

<p align="center">
  <a href="https://github.com/xming521/WeClone/blob/master/README_zh.md" target="_blank">简体中文</a>｜
  English</a>｜
  <a href="https://www.weclone.love/" target="_blank"> Project Homepage </a> ｜
  <a href="https://docs.weclone.love/what-is-weclone.html" target="_blank"> Documentation </a> 
</p>

> [!IMPORTANT]
> ### WhatsApp and Telegram chat logs integration for digital avatar creation is coming !

## ✨Core Features
- 💫 Complete end-to-end solution for creating digital avatars, including chat data export, preprocessing, model training, and deployment
- 💬 Fine-tune LLM using chat history with support for image modal data, infusing it with that authentic "flavor"
- 🔗 Integrate with Telegram, WeChat, WhatsApp (coming soon) to create your own digital avatar
- 🛡️ Privacy information filtering with localized fine-tuning and deployment for secure and controllable data

## 📋Features & Notes
> [!IMPORTANT]
> ### WeClone is currently not partnered with any platform and has not issued any cryptocurrency. The only official website is: [weclone.love](https://www.weclone.love). Beware of imitations.

### Chat Platform Support

| Platform | Text | Images | Voice | Video | Animated Emojis | Links (Sharing) | Quote | Forward | Location | Files |
|----------|------|--------|-------|-------|-----------------|-----------------|-------|---------|----------|-------|
| WeChat | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Telegram | ✅ | ✅ | ❌ | ❌ | ⚠️Convert to Emoji | ❌ | ❌ | ✅ | ✅ | ❌ |

 

> [!IMPORTANT]
> - WeClone is still in rapid iteration phase, current performance does not represent final results.  
> - LLM fine-tuning effectiveness largely depends on model size, quantity and quality of chat data. Theoretically, larger models with more data yield better results.
> - 7B models are prone to becoming "dumb", 14B models can barely communicate, while 32B+ models perform much better.   
> - Windows environment has not been rigorously tested. You can use WSL as the runtime environment.

### Recent Updates
[25/06/05] Support for image modal data fine-tuning

### Hardware Requirements

The project uses Qwen2.5-VL-7B-Instruct model by default with LoRA method for SFT stage fine-tuning. You can also use other models and methods supported by [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main#supported-models).

Estimated VRAM requirements (text-only large model memory usage as follows, vision models increase based on image quantity and size): 
| Method                          | Precision |   7B  |  14B  |  30B  |   70B  |   `x`B  |
| ------------------------------- | --------- | ----- | ----- | ----- | ------ | ------- |
| Full (`bf16` or `fp16`)         |    32     | 120GB | 240GB | 600GB | 1200GB | `18x`GB |
| Full (`pure_bf16`)              |    16     |  60GB | 120GB | 300GB |  600GB |  `8x`GB |
| Freeze/LoRA/GaLore/APOLLO/BAdam |    16     |  16GB |  32GB |  64GB |  160GB |  `2x`GB |
| QLoRA                           |     8     |  10GB |  20GB |  40GB |   80GB |   `x`GB |
| QLoRA                           |     4     |   6GB |  12GB |  24GB |   48GB | `x/2`GB |
| QLoRA                           |     2     |   4GB |   8GB |  16GB |   24GB | `x/4`GB |


## Environment Setup
1. CUDA installation (skip if already installed, **requires version 12.6 or above**)

2. It is recommended to use [uv](https://docs.astral.sh/uv/) to install dependencies, which is a very fast Python environment manager. After installing uv, you can use the following commands to create a new Python environment and install dependencies. 
```bash
git clone https://github.com/xming521/WeClone.git && cd WeClone
uv venv .venv --python=3.10
source .venv/bin/activate # windows .venv\Scripts\activate
uv pip install --group main -e . 
```

3. Copy the configuration file template and rename it to `settings.jsonc`, and make subsequent configuration changes in this file:

```bash
cp examples/tg.template.jsonc settings.jsonc
```

> [!NOTE]
> Training and inference related configurations are unified in the file `settings.jsonc`

4. Use the following command to test whether the CUDA environment is correctly configured and can be recognized by PyTorch (not needed for Mac):
```bash
  python -c "import torch; print('CUDA Available:', torch.cuda.is_available());"
```

5. (Optional) Install FlashAttention to accelerate training and inference: `uv pip install flash-attn --no-build-isolation`.

## Model Download
It is recommended to use [Hugging Face](https://huggingface.co/docs/hub/models-downloading) to download models, or use the following command:
```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct models/Qwen2.5-VL-7B-Instruct
```

## Data Preparation

Please use [Telegram Desktop](https://desktop.telegram.org/) to export chat records. Select Photos for message types and JSON for format. You can export multiple contacts (group chat records are not recommended), then place the exported `ChatExport_*` in the `./dataset/telegram` directory, meaning put different people's chat record folders together in `./dataset/telegram`.   


## Data Preprocessing

- By default, the project uses Microsoft Presidio to remove `phone numbers, email addresses, credit card numbers, IP addresses, geographic location names, international bank account numbers, cryptocurrency wallet addresses, age information, and generic ID numbers` from the data, but it cannot guarantee 100% identification.
- Therefore, a blocklist `blocked_words` is provided in `settings.jsonc`, allowing users to manually add words or phrases they want to filter (the entire sentence containing blocked words will be removed by default).

> [!IMPORTANT]
> 🚨 Please be sure to protect personal privacy and do not leak personal information!

- Execute the following command to process the data. You can modify the `make_dataset_args` in settings.jsonc according to your own chat style.
```bash
weclone-cli make-dataset
```
- Currently supports time window strategy. Messages from a single person are combined into one sentence by commas based on `single_combine_time_window`, and Q&A pairs are matched based on `qa_match_time_window`.
- For **training multimodal large models**: Enable by adding `images` to `include_type`, and control image quantity and size through `image_max_pixels` and `max_image_num` parameters to reduce VRAM usage.
- For **Image to Text**: Add `images` to `include_type` and configure `vision_api` parameters. The system will use external multimodal models to convert images to text, and the final generated dataset **is still used for training text-only LLM**.
- You can enable the `enable_clean` option in `clean_dataset` to clean the data for better results (multimodal data is not currently supported). The current system supports using `llm judge` to score chat records, providing **vllm offline inference** and **API online inference** methods. By default, offline inference is enabled. To switch to API-based online inference mode, modify `"online_llm_clear": false` to `true` in the `settings.jsonc` file, and configure relevant parameters such as `base_url`, `llm_api_key`, and `model_name`. All models compatible with OpenAI interface can be accessed.
- After obtaining the `llm scoring score distribution`, you can filter acceptable data by setting the `accept_score` parameter, and appropriately reduce the `lora_dropout` parameter in `train_sft_args` to improve the model's fitting effect.

## Configure Parameters and Fine-tune Model

- (Optional) Modify `model_name_or_path`, `template`, `lora_target` in `settings.jsonc` to select other locally downloaded models.   
- Modify `per_device_train_batch_size` and `gradient_accumulation_steps` to adjust VRAM usage.  
- You can modify parameters like `num_train_epochs`, `lora_rank`, `lora_dropout` in `train_sft_args` based on your dataset's quantity and quality.

### Single GPU Training
```bash
weclone-cli train-sft
```

### Multi-GPU Training
Uncomment the `deepspeed` line in `settings.jsonc` and use the following command for multi-GPU training:
```bash
uv pip install deepspeed
deepspeed --num_gpus=number_of_gpus weclone/train/train_sft.py
```

### 使用浏览器demo简单推理
可以在这一步测试出合适的temperature、top_p值，修改settings.jsonc的`infer_args`后，供后续推理时使用。
```bash
weclone-cli webchat-demo
```

### 使用接口进行推理

```bash
weclone-cli server
```

### 使用常见聊天问题测试
不包含询问个人信息的问题，仅有日常聊天。测试结果在test_result-my.txt。
```bash
weclone-cli server
weclone-cli test-model
```

## 🖼️ 微调效果
> [!TIP] 
> **QQ群内有部署好的Qwen2.5VL 32B Bot，可以体验效果。更多案例可以关注[小红书](https://www.xiaohongshu.com/user/profile/628109730000000021029de4)**  

使用Qwen2.5-14B-Instruct模型，大概3万条处理后的有效数据，loss降到了3.5左右的效果：
<details>
<summary>截图</summary>
<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/0775ec52-452b-485f-9785-c6eb7b277132" alt="alt text" style="width: 48%; min-width: 150px;">
  <img src="https://github.com/user-attachments/assets/8c7628b5-da70-4c37-9e51-fdfb0eadd2df" alt="alt text" style="width: 48%; min-width: 150px;">
  <img src="https://github.com/user-attachments/assets/523aa742-2aa3-40e9-bd67-b98b336e83a8" alt="alt text" style="width: 48%; min-width: 150px;">
  <img src="https://github.com/user-attachments/assets/dabf0603-dcc4-4a47-b5c3-2bbc036820d9" alt="alt text" style="width: 48%; min-width: 150px;">
</div>
</details>


## 🤖 部署到聊天机器人
### LangBot

[LangBot](https://github.com/RockChinQ/LangBot) 是一个开源的接入全球多种即时通信平台的 LLM 机器人平台，支持Discord、Telegram、Slack等平台，适合各种场景使用。

1. [部署 LangBot](https://github.com/RockChinQ/LangBot/blob/master/README_EN.md#-getting-started)
2. 在 LangBot 中添加一个机器人
4. 在模型页添加新模型，名称`gpt-3.5-turbo`，供应商选择 OpenAI，填写 请求 URL 为 WeClone 的地址，详细连接方式可以参考[文档](https://docs.langbot.app/en/workshop/network-details.html)，API Key 任意填写。

<img width="400px" alt="image" src="https://github.com/user-attachments/assets/fc167dea-7c93-4d94-9c5f-db709d0320ba" />

6. 在流水线配置中选择刚才添加的模型，或修改提示词配置

<img width="400px" alt="image" src="https://github.com/user-attachments/assets/dbb0fd0a-f760-42db-acd0-bb99c859b52e" />

### AstrBot

[AstrBot](https://github.com/AstrBotDevs/AstrBot) 是易上手的多平台 LLM 聊天机器人及开发框架 ✨ 平台支持 QQ、Telegram、微信、企微、飞书。      

使用步骤：
1. 部署 AstrBot
2. 在 AstrBot 中部署消息平台
3. 执行 `weclone-cli server` 启动api服务
4. 在 AstrBot 中新增服务提供商，类型选择OpenAI，API Base URL 根据AstrBot部署方式填写（例如docker部署可能为http://172.17.0.1:8005/v1） ，模型填写gpt-3.5-turbo,API Key随意填写一个
5. 微调后不支持工具调用，请先关掉默认的工具，消息平台发送指令： `/tool off all`，否则会没有微调后的效果。 
6. 根据微调时使用的default_system，在 AstrBot 中设置系统提示词。
![5](https://github.com/user-attachments/assets/19de7072-076a-4cdf-8ae6-46b9b89f536a)
> [!IMPORTANT]
> 检查api_service的日志，尽量保证大模型服务请求的参数和微调时一致，tool插件能力都关掉。
7. 调整采样参数，例如temperature、top_p、top_k等
[配置自定义的模型参数](https://astrbot.app/config/model-config.html#%E9%85%8D%E7%BD%AE%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0)


## 📌 路线图
- [ ] 支持更多数据源
- [ ] 更丰富的上下文：包括上下文对话、聊天对象信息、时间等 
- [ ] Memory 支持
- [ ] 支持多模态:已支持图片
- [ ] 数据增强
- [ ] 支持GUI
- [ ] 支持COT思考

## 问题解决
#### [官方文档FAQ](https://www.weclone.love/FAQ.html)    
同时建议使用[DeepWiki](https://deepwiki.com/xming521/WeClone)解决问题。



## ❤️ 贡献代码

欢迎任何 Issues/Pull Requests！

你可以通过查看Issues或帮助审核 PR（拉取请求）来贡献。对于新功能的添加，请先通过 Issue 讨论。   
开发环境：
```bash
uv pip install --group dev -e .
pre-commit install
```

项目使用`pytest`测试，`pyright`检查类型，`ruff`检查代码格式。

## 🙏 致谢

感谢以下代码贡献者和社区里其他成员的贡献

<a href="https://github.com/xming521/WeClone/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xming521/WeClone" />
</a>

同时本项目受益于[PyWxDump](https://github.com/xaoyaoo/PyWxDump)、[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)、[AstrBot](https://github.com/AstrBotDevs/AstrBot)、[LangBot](https://github.com/RockChinQ/LangBot)等优秀开源项目。

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
<br>  
<br>  

## ⭐ Star History
> [!TIP] 
> 如果本项目对您有帮助，或者您关注本项目的未来发展，请给项目 Star，谢谢 

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=xming521/WeClone&type=Date)](https://www.star-history.com/#xming521/WeClone&Date)

</div>


<div align="center"> 克隆我们，保留灵魂的芬芳 </div>
