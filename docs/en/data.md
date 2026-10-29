[简体中文](../zh/data.md)｜English

# Preparing Data

## Supported sources

Telegram chats are supported today. Chats from other platforms can be imported in the generic CSV format, and more data sources are coming. The life portrait reads text messages only. The fine-tuned clone supports these message types:

| Platform | Text | Images | Voice | Video | Animated Emojis/Stickers | Links (Sharing) | Quote | Forward | Location | Files |
|----------|------|--------|-------|-------|-----------------|-----------------|-------|---------|----------|-------|
| Telegram | ✅ | ✅ | ❌ | ❌ | ⚠️ Converted to Emoji | ❌ | ❌ | ✅ | ✅ | ❌ |
| WhatsApp | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| Discord | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| Slack | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |

## Telegram

1. Export your chats with [Telegram Desktop](https://desktop.telegram.org/): in a chat, click the top right corner, choose "Export chat history", select Photos as the message type and JSON as the format. You can export several contacts. Group chats are not recommended.
2. Put the exported `ChatExport_*` folders in `dataset/telegram/`, one folder per contact.
3. Set these under `make_dataset_args` in `settings.jsonc`:
   - `platform`: `telegram`
   - `telegram_args.my_id`: your own Telegram user ID, which is the `from_id` of the messages you sent in `result.json`
   - `language`: the main language of your chats, `zh` or `en`
   - `include_type`: the message types used for fine-tuning, `["text"]` by default. Add `"image"` to fine-tune a multimodal model.

## Generic CSV format

Chats from other platforms can be converted to WeClone's generic CSV format and placed in `dataset/csv/`, with `platform` left as `chat`. See [Preparing Data](https://docs.weclone.love/en/docs/deploy/export-chat-history#generic-csv-format) on the docs site for the columns and folder layout.
