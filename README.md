![download](https://github.com/user-attachments/assets/5842e84e-004f-4afd-9373-af64e9575b78)
<h3 align="center">🚀 One-stop solution for creating your digital avatar from chat history 💡</h3>  
<h3 align="center">🚀从聊天记录创造数字分身的一站式解决方案💡</h3>  


<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/xming521/WeClone?style=for-the-badge&logo=github&label=Stars&logoColor=white&color=ffda65)](https://github.com/xming521/WeClone/stargazers)
[![GitHub release](https://img.shields.io/github/v/release/xming521/WeClone?style=for-the-badge&logo=github&label=Release&logoColor=white&color=06d094)](https://github.com/xming521/WeClone/releases)
<a href="https://qm.qq.com/cgi-bin/qm/qr?k=wNdgbOVT6oFOJ2wlMLsolUXErW9ESLpk&jump_from=webapi&authKey=z/reOp6YLyvR4Tl2k2nYMsLoMC3w9/99ucgKMX0oRGlxDV/WbYnvq2QxODoIkfxn" target="_blank" style="text-decoration: none;">
  <img src="https://img.shields.io/badge/QQ群-708067078-12B7F5?style=for-the-badge&logo=qq&logoColor=white" alt="WeClone①" title="WeClone①">
</a>
[![Twitter](https://img.shields.io/badge/Twitter-@weclone567-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/weclone567)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/+JEdak4m0XEQ3NGNl)

<a href="https://hellogithub.com/repository/12ab209b56cb4cfd885c8cfd4cfdd53e" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=12ab209b56cb4cfd885c8cfd4cfdd53e&claim_uid=RThlPDoGrFvdMY5" alt="Featured｜HelloGitHub" style="width: 150px; height: 28px;" /></a>
<a href="https://trendshift.io/repositories/13759" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13759" alt="xming521%2FWeClone | Trendshift" style="width: 220px; height: 50px;" /></a>
<a href="https://deepwiki.com/xming521/WeClone"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"  style="width: 134px; height: 23px;margin-bottom: 3px;"></a>
</div>
<p align="center">
  <a href="https://blog.051088.xyz/2025/05/14/WeClone-%E7%94%A8%E5%BE%AE%E4%BF%A1%E8%81%8A%E5%A4%A9%E8%AE%B0%E5%BD%95%E6%89%93%E9%80%A0%E8%87%AA%E5%B7%B1%E7%9A%84AI%E6%95%B0%E5%AD%97%E5%88%86%E8%BA%AB/" target="_blank">
    Windows部署指南
  </a>
  <a>|</a>
  <a href="https://blog.051088.xyz/posts/weclone-linux-tutorial/" target="_blank">
    Linux部署指南【保姆级】
  </a>
</p>

> [!IMPORTANT]
> <h3> WhatsApp and Telegram chat logs integration for digital avatar creation is coming ! </h3>

## ✨核心功能
- 💫 涵盖打造数字分身的全链路方案，包括聊天数据导出、预处理、模型训练、部署
- 💬 使用微信聊天记录微调LLM，让大模型有"那味儿"
- 🔗 绑定到微信、QQ、Telegram、企微、飞书机器人，实现自己的数字分身
- 🛡️ 隐私信息过滤，本地化微调部署，数据安全可控

## 📋特性与说明

> [!IMPORTANT]
> - WeClone仍在快速迭代期，当前效果不代表最终效果。  
> - 微调LLM效果很大程度取决于模型大小、聊天数据的数量和质量，理论上模型越大，数据越多，效果越好。   
> - Windows环境未进行严格测试，可以使用WSL作为运行环境。详细教程可点击[Windows部署指南](https://blog.051088.xyz/2025/05/14/WeClone-%E7%94%A8%E5%BE%AE%E4%BF%A1%E8%81%8A%E5%A4%A9%E8%AE%B0%E5%BD%95%E6%89%93%E9%80%A0%E8%87%AA%E5%B7%B1%E7%9A%84AI%E6%95%B0%E5%AD%97%E5%88%86%E8%BA%AB/)查看。

### 硬件要求

项目默认使用Qwen2.5-7B-Instruct模型，LoRA方法对sft阶段微调，大约需要16GB显存。也可以使用[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md#%E6%A8%A1%E5%9E%8B)支持的其他模型和方法。

需要显存的估算值：
| 方法                             | 精度 |   7B  |  14B  |  30B  |   70B  |   `x`B  |
| ------------------------------- | ---- | ----- | ----- | ----- | ------ | ------- |
| Full (`bf16` or `fp16`)         |  32  | 120GB | 240GB | 600GB | 1200GB | `18x`GB |
| Full (`pure_bf16`)              |  16  |  60GB | 120GB | 300GB |  600GB |  `8x`GB |
| Freeze/LoRA/GaLore/APOLLO/BAdam |  16  |  16GB |  32GB |  64GB |  160GB |  `2x`GB |
| QLoRA                           |   8  |  10GB |  20GB |  40GB |   80GB |   `x`GB |
| QLoRA                           |   4  |   6GB |  12GB |  24GB |   48GB | `x/2`GB |
| QLoRA                           |   2  |   4GB |   8GB |  16GB |   24GB | `x/4`GB |


## 环境搭建
1.cuda安装(已安装可跳过，**要求版本12.4及以上**)：[LLaMA Factory](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/installation.html#cuda) 

2.建议使用 [uv](https://docs.astral.sh/uv/)安装依赖，这是一个非常快速的 Python 环境管理器。安装uv后，您可以使用以下命令创建一个新的Python环境并安装依赖项，注意这不包含音频克隆功能的依赖：
```bash
git clone https://github.com/xming521/WeClone.git
cd WeClone
uv venv .venv --python=3.10
source .venv/bin/activate # windows下执行 .venv\Scripts\activate
uv pip install --group main -e . 
```
> [!TIP]
> 如果要使用最新的模型进行微调，需要手动安装最新版LLaMA Factory：`uv pip install --upgrade git+https://github.com/hiyouga/LLaMA-Factory.git`,同时其他依赖版本也可能需要修改，例如vllm pytorch transforms

3.将配置文件模板复制一份并重命名为`settings.jsonc`，后续配置修改在此文件进行：
```bash
cp settings.template.jsonc settings.jsonc
```
> [!NOTE]
> 训练以及推理相关配置统一在文件`settings.jsonc`

4.使用以下命令测试CUDA环境是否正确配置并可被PyTorch识别，Mac不需要：
```bash
python -c "import torch; print('CUDA是否可用:', torch.cuda.is_available());"
```

5.（可选）安装FlashAttention，加速训练和推理：`uv pip install flash-attn --no-build-isolation`

## 模型下载
```bash
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen2.5-7B-Instruct.git
```
下载有问题使用其他方式下载：[模型的下载](https://www.modelscope.cn/docs/models/download)


## 数据准备

请使用[PyWxDump](https://github.com/xaoyaoo/PyWxDump)提取微信聊天记录（不支持4.0版本微信）。可以先将手机的聊天记录迁移（备份）到电脑，数据量更多一些。下载软件并解密数据库后，点击聊天备份，导出类型为CSV，可以导出多个联系人（不建议使用群聊记录），然后将导出的位于`wxdump_tmp/export` 的 `csv` 文件夹放在`./dataset`目录即可，也就是不同人聊天记录的文件夹一起放在 `./dataset/csv`。   

## 数据预处理

- 项目默认去除了数据中的手机号、身份证号、邮箱、网址。还在`settings.jsonc`中提供了一个禁用词词库`blocked_words`，可以自行添加需要过滤的词句（会默认去掉包括禁用词的整句）。
> [!IMPORTANT]
> 🚨 请一定注意保护个人隐私，不要泄露个人信息！

- 执行以下命令对数据进行处理，可以根据自己的聊天风格修改settings.jsonc的`make_dataset_args`。
```bash
weclone-cli make-dataset
```
- 目前仅支持时间窗口策略，根据`single_combine_time_window`将单人连续消息通过逗号连接合并为一句，根据`qa_match_time_window`匹配问答对。
- 可以启用`clean_dataset`中的`enable_clean`选项，对数据进行清洗，以达到更好效果。* 当前系统支持使用 `llm judge` 对聊天记录进行打分，提供 **vllm 离线推理** 和 **API 在线推理** 两种方式。可通过将 `settings.jsonc` 文件中的 `"online_llm_clear": false` 修改为 `true` 来启用 API 在线推理模式，并配置相应的 `base_url`、`llm_api_key`、`model_name` 等参数。所有兼容 OpenAI 接口的模型均可接入。
- 在获得 `llm 打分分数分布情况` 后，可通过设置 `accept_score` 参数筛选可接受的分数区间，同时可适当降低 `train_sft_args` 中的 `lora_dropout` 参数，以提升模型的拟合效果。

## 配置参数并微调模型

- (可选)修改 `settings.jsonc` 的 `model_name_or_path` 和 `template` 选择本地下载好的其他模型。  
- 修改`per_device_train_batch_size`以及`gradient_accumulation_steps`来调整显存占用。  
- 可以根据自己数据集的数量和质量修改`train_sft_args`的`num_train_epochs`、`lora_rank`、`lora_dropout`等参数。

### 单卡训练
```bash
weclone-cli train-sft
```
多卡环境单卡训练，需要先执行 `export CUDA_VISIBLE_DEVICES=0`

### 多卡训练
取消`settings.jsonc`中`deepspeed`行代码注释，使用以下命令多卡训练：
```bash
uv pip install deepspeed
deepspeed --num_gpus=使用显卡数量 weclone/train/train_sft.py
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
使用Qwen2.5-14B-Instruct模型，大概3万条处理后的有效数据，loss降到了3.5左右的效果。
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

### AstrBot

[AstrBot](https://github.com/AstrBotDevs/AstrBot) 是易上手的多平台 LLM 聊天机器人及开发框架 ✨ 平台支持 QQ、QQ频道、Telegram、微信、企微、飞书。      

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

### LangBot

[LangBot](https://github.com/RockChinQ/LangBot) 是一个开源的接入全球多种即时通信平台的 LLM 机器人平台，适合各种场景使用。

1. [部署 LangBot](https://github.com/RockChinQ/LangBot#-%E5%BC%80%E5%A7%8B%E4%BD%BF%E7%94%A8)
2. 在 LangBot 中添加一个机器人
4. 在模型页添加新模型，名称`gpt-3.5-turbo`，供应商选择 OpenAI，填写 请求 URL 为 WeClone 的地址，详细连接方式可以参考[文档](https://docs.langbot.app/zh/workshop/network-details.html)，API Key 任意填写。

<img width="400px" alt="image" src="https://github.com/user-attachments/assets/fc167dea-7c93-4d94-9c5f-db709d0320ba" />

6. 在流水线配置中选择刚才添加的模型，或修改提示词配置

<img width="400px" alt="image" src="https://github.com/user-attachments/assets/dbb0fd0a-f760-42db-acd0-bb99c859b52e" />

## 📌 路线图
- [ ] 更丰富的上下文：包括上下文对话、聊天对象信息、时间等 + 思考
- [ ] Memory 支持
- [ ] 支持多模态
- [ ] 数据增强
- [ ] 支持GUI

## 问题解决
- 微调问题：[LLaMA-Factory| FAQs | 常见问题](https://github.com/hiyouga/LLaMA-Factory/issues/4614) 或者更方便的 [![更方便的Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hiyouga/LLaMA-Factory)

## ❤️ 贡献代码

欢迎任何 Issues/Pull Requests！

你可以通过查看Issues或帮助审核 PR（拉取请求）来贡献。对于新功能的添加，请先通过 Issue 讨论。   
运行`uv pip install --group dev -e .`安装开发依赖。   
项目使用`pytest`测试(测试脚本待完善)，`pyright`检查类型，`ruff`检查代码格式。


## ⚠️ 免责声明
> [!CAUTION]
> 请勿用于非法用途，否则后果自负。
<details>
<summary>1. 使用目的</summary>

* 本项目仅供学习交流使用，**请勿用于非法用途**，**请勿用于非法用途**，**请勿用于非法用途**，否则后果自负。
* 用户理解并同意，任何违反法律法规、侵犯他人合法权益的行为，均与本项目及其开发者无关，后果由用户自行承担。

2. 使用期限

* 您应该在下载保存使用本项目的24小时内，删除本项目的源代码和程序；超出此期限的任何使用行为，一概与本项目及其开发者无关。

3. 操作规范

* 本项目仅允许在授权情况下使用数据训练，严禁用于非法目的，否则自行承担所有相关责任；用户如因违反此规定而引发的任何法律责任，将由用户自行承担，与本项目及其开发者无关。
* 严禁用于窃取他人隐私，严禁用于窃取他人隐私，严禁用于窃取他人隐私，否则自行承担所有相关责任。

4. 免责声明接受

* 下载、保存、进一步浏览源代码或者下载安装、编译使用本程序，表示你同意本警告，并承诺遵守它;

5. 禁止用于非法测试或渗透

* 禁止利用本项目的相关技术从事非法测试或渗透，禁止利用本项目的相关代码或相关技术从事任何非法工作，如因此产生的一切不良后果与本项目及其开发者无关。
* 任何因此产生的不良后果，包括但不限于数据泄露、系统瘫痪、侵犯隐私等，均与本项目及其开发者无关，责任由用户自行承担。

6. 免责声明修改

* 本免责声明可能根据项目运行情况和法律法规的变化进行修改和调整。用户应定期查阅本页面以获取最新版本的免责声明，使用本项目时应遵守最新版本的免责声明。

7. 其他

* 除本免责声明规定外，用户在使用本项目过程中应遵守相关的法律法规和道德规范。对于因用户违反相关规定而引发的任何纠纷或损失，本项目及其开发者不承担任何责任。

* 请用户慎重阅读并理解本免责声明的所有内容，确保在使用本项目时严格遵守相关规定。

</details>
请用户慎重阅读并理解本免责声明的所有内容，确保在使用本项目时严格遵守相关规定。

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
