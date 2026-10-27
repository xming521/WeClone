简体中文｜[English](../en/chatbot.md)

# 接入聊天机器人

机器人平台通过 `/v1` 下的 OpenAI 兼容接口调用[微调好的分身](finetune.md)。

| 平台 | 部署支持 |
|------|------|
| 个人微信 | ✅（基于 **openclaw-weixin**） |
| Telegram | ✅ |
| WhatsApp | 🚧 |
| Discord | ✅ |
| Slack | ✅ |

## 启动服务并生成令牌

1. 启动服务：
   ```bash
   weclone-cli server --inference --port 8005
   ```
   机器人平台用 Docker 部署时加上 `--host 172.17.0.1`（宿主机在 Docker 网络中的地址），否则容器访问不到服务。
2. 打开网页的「机器人接入」页（`#/bots`）生成令牌，页面上会给出 API Base URL 和 API Key。
   - 所有机器人共用这一个令牌，服务重启后仍然有效。重新生成或停用后，旧令牌立即失效。
   - 加密模式下，数据密钥用令牌加密后保存在本机，机器人调用时不需要有人登录。
   - 令牌要像密码一样保管，它和 `.weclone/bots.sqlite3` 放在一起就能解密全部数据。

## AstrBot

[AstrBot](https://github.com/AstrBotDevs/AstrBot) 是易上手的多平台 LLM 聊天机器人及开发框架，支持 Telegram、飞书等平台。

1. 部署 AstrBot，并在 AstrBot 中部署消息平台。
2. 按上一节启动服务并生成令牌。
3. 在 AstrBot 中新增服务提供商，类型选择 OpenAI。API Base URL 根据 AstrBot 的部署方式填写（例如 Docker 部署时可能为 `http://172.17.0.1:8005/v1`），模型填写 `gpt-3.5-turbo`，API Key 填生成的令牌。
4. 微调后的模型不支持工具调用。请先在消息平台发送 `/tool off_all` 关掉默认的工具，否则看不到微调后的效果。
5. 根据微调时使用的 `default_system`，在 AstrBot 中设置系统提示词。

> [!IMPORTANT]
> 检查 api_service 的日志，尽量保证大模型服务的请求参数和微调时一致，并关闭所有工具插件。

## LangBot

[LangBot](https://github.com/langbot-app/LangBot) 是开源的 LLM 机器人平台，可以接入全球多种即时通信平台。

1. [部署 LangBot](https://docs.langbot.app/zh/insight/guide)，并在 LangBot 中添加一个机器人。
2. 按上一节启动服务并生成令牌。
3. 在模型页添加新模型：名称填 `gpt-3.5-turbo`，供应商选择 OpenAI，请求 URL 填 WeClone 的地址（连接方式参考 [LangBot 文档](https://docs.langbot.app/zh/workshop/network-details)），API Key 填生成的令牌。
4. 在流水线配置中选择刚才添加的模型，或修改提示词配置。

带截图的步骤见文档站的 [AstrBot](https://docs.weclone.love/zh/docs/chatbot/AstrBot.html) 和 [LangBot](https://docs.weclone.love/zh/docs/chatbot/LangBot.html) 页面。
