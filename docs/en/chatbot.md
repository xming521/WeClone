[简体中文](../zh/chatbot.md)｜English

# Chat Bots

Bot platforms call the [fine-tuned clone](finetune.md) through the OpenAI-compatible API under `/v1`.

| Platform | Deployment Support |
|----------|--------------------|
| Telegram | ✅ |
| WhatsApp | 🚧 |
| WeChat (Personal Account) | ✅ (Based on **openclaw-weixin**) |
| Discord | ✅ |
| Slack | ✅ |

## Start the server and generate a token

1. Start the server:
   ```bash
   weclone-cli server --inference --port 8005
   ```
   If the bot platform runs in Docker, add `--host 172.17.0.1` (the host's address on the Docker network) so the container can reach the server.
2. Open the Bots page (`#/bots`) in the web UI and generate a token. The page shows the API Base URL and API Key.
   - All bots share this one token, and it keeps working across server restarts. Regenerating or withdrawing it stops the old one at once.
   - In encrypted mode the data key is stored on this machine encrypted under the token, so bots are served without anyone logged in.
   - Keep the token as safe as the password. Together with `.weclone/bots.sqlite3` it decrypts all data.

## AstrBot

[AstrBot](https://github.com/AstrBotDevs/AstrBot) is an easy-to-use multi-platform LLM chatbot and development framework. It supports Discord, Telegram, Slack, Feishu and other platforms.

1. Deploy AstrBot and connect your messaging platforms in it.
2. Start the server and generate a token as described above.
3. Add a service provider in AstrBot with the OpenAI type. Fill in the API Base URL according to how AstrBot is deployed (with Docker it may be `http://172.17.0.1:8005/v1`), set the model to `gpt-3.5-turbo`, and use the token as the API Key.
4. The fine-tuned model does not support tool calling. Send `/tool off_all` on the messaging platform to turn off the default tools, otherwise the fine-tuned behavior will not show.
5. Set the system prompt in AstrBot to the `default_system` used during fine-tuning.

> [!IMPORTANT]
> Check the api_service logs to keep the request parameters as close as possible to those used during fine-tuning, and turn off all tool plugins.

## LangBot

[LangBot](https://github.com/langbot-app/LangBot) is an open-source LLM chatbot platform that connects to instant messaging platforms around the world.

1. [Deploy LangBot](https://docs.langbot.app/en/insight/guide) and add a bot (Discord, Telegram, Slack, Lark and others) in it.
2. Start the server and generate a token as described above.
3. Add a model on the model page: name it `gpt-3.5-turbo`, choose OpenAI as the provider, set the request URL to WeClone's address (see the [LangBot documentation](https://docs.langbot.app/en/workshop/network-details) for connection details), and use the token as the API Key.
4. Select the new model in the pipeline configuration, or adjust the prompt configuration.

See the [AstrBot](https://docs.weclone.love/en/docs/chatbot/AstrBot.html) and [LangBot](https://docs.weclone.love/en/docs/chatbot/LangBot.html) pages on the docs site for illustrated steps.
