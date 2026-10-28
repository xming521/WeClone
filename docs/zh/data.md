简体中文｜[English](../en/data.md)

# 准备数据

## 支持的数据源

目前支持 Telegram 聊天记录，其他平台可以转换成通用 CSV 格式导入，更多数据源持续接入中。人生画像只读取文字消息，微调分身支持的消息类型如下：

| 平台 | 文字 | 图片 | 语音 | 视频 | 动画表情 | 链接（分享） | 引用 | 转发 | 位置 | 文件 |
|------|------|------|------|------|----------|-----------|------|------|------|------|
| Telegram | ✅ | ✅ | ❌ | ❌ | ⚠️ 转为 Emoji | ❌ | ❌ | ✅ | ✅ | ❌ |
| WhatsApp | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| Discord | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| Slack | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |

## Telegram

1. 使用 [Telegram Desktop](https://desktop.telegram.org/) 导出聊天记录：在聊天界面点击右上角，选择导出聊天记录，消息类型选择照片，格式选择 JSON。可以导出多个联系人，不建议使用群聊记录。
2. 把导出的 `ChatExport_*` 文件夹都放进 `dataset/telegram/`，每个联系人一个文件夹。
3. 在 `settings.jsonc` 的 `make_dataset_args` 中设置：
   - `platform`：改为 `telegram`
   - `telegram_args.my_id`：你自己的 Telegram 用户 ID，即 `result.json` 中你发出的消息的 `from_id`
   - `language`：聊天常用语言，`zh` 或 `en`
   - `include_type`：微调使用的消息类型，默认 `["text"]`，微调多模态模型时加上 `"image"`

## 通用 CSV 格式

其他平台的聊天记录可以转换成 WeClone 的通用 CSV 格式，放进 `dataset/csv/`，并保持 `platform` 为 `chat`。CSV 的列和目录结构见文档站的[聊天平台适配](https://docs.weclone.love/zh/docs/deploy/export-chat-history.html)。
