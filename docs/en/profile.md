[简体中文](../zh/profile.md)｜English

# Life Portrait and Assistants

## Build your life portrait

Put your chats in place as described in [Preparing data](data.md), choose a [model backend](installation.md#model-backend), then run from the project root:

```bash
weclone-cli build-profile
```

It asks what the portrait is for. You can also pass the choice directly:

- `--purpose assistant`: for personal assistants. Extracts the portrait and events.
- `--purpose clone`: for a clone. Also extracts your speaking style, which needs the [speaking style dependencies](installation.md#speaking-style-dependencies).

Before it starts, it shows the model backend and asks you to confirm, because your chat content is sent to that model service. It then runs these stages:

1. Prepare the chat data: group text messages by contact and split them into sessions
2. Extract profile memories and event memories
3. Induce life domains, then classify, group into topics and summarize the memories, giving every fact a privacy level and a time scope
4. Build the portrait hierarchy
5. Write domain summaries
6. Extract speaking style (`--purpose clone` only)

| Option | Description |
|---|---|
| `--output-dir <dir>` | Result directory, which must be new. Defaults to a timestamped directory under `dataset/res_csv/agent/profile_runs/` |
| `--max-context-tokens <limit>` | Input token limit for memory organization and the hierarchy. The Codex backend reads the model's context length by default. The API backend requires it |

Model answers are cached by input content in `dataset/res_csv/agent/llm_cache.sqlite`. If a run fails partway, rerun it into a new result directory and only the missing parts are requested. Delete the file to request everything again.

## Review in the web UI

```bash
weclone-cli server                    # serve the newest run
weclone-cli server --run <result-dir> # serve a specific run
```

Open http://127.0.0.1:5175 and sign in with the shared password (see [Security](security.md#password)). The web UI follows your browser's language, English or Chinese. The 中/EN switch in the top bar changes it:

| Page | Purpose |
|---|---|
| Portrait (`#/`) | Browse the portrait by domain and topic, adopt, edit or reject each fact, and see the chat each fact came from |
| Style (`#/style`) | View the speaking style extracted with `--purpose clone` |
| Summaries (`#/summary`) | Generate and approve a summary per domain and privacy level, and copy them for a personal assistant |
| Assistants (`#/assistant`) | Choose the level and domains assistants can read, and generate their token |
| Bots (`#/bots`) | Generate the token chat bots call the clone with. See [Chat bots](chatbot.md) |

Your review is saved in the run's `profile_hierarchy/` directory and belongs to that run only.

## Connect AI assistants

Claude Code, Codex and other MCP clients can read your life portrait:

1. Approve the summaries you want to share on the Summaries page. Assistants only get approved summaries.
2. On the Assistants page, choose the privacy level and the domains to share, then generate a token.
3. Run the commands shown on the page for Claude Code or Codex. All assistants share this one token. Changes to the level and domains apply from their next call.

Assistants connect over MCP to `http://127.0.0.1:5175/mcp` and get two read-only tools:

- `weclone_profile`: returns the approved domain summaries.
- `weclone_search`: finds portrait entries by keyword or domain, including entries still pending review and excluding rejected ones.

Keep `weclone-cli server` running while assistants use it. In encrypted mode, also turn on **Allow assistants to read** on the Assistants page for a chosen time. Turn it on again when the time is up or after the server restarts. Every call is recorded in the review database.

> [!WARNING]
> At the **All** level, private and sensitive entries also go to the cloud model behind the assistant.
