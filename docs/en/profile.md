[简体中文](../zh/profile.md)｜English

# Life Portrait and Assistants

## Build your life portrait

Put your chats in place as described in [Preparing data](data.md), choose a [model backend](installation.md#model-backend), then start the web UI from the project root:

```bash
weclone-cli server
```

Open http://127.0.0.1:5175, set a password and go to the Build portrait page. The home page leads there while you have no portrait, and later you reach it from Settings → Portrait versions. First choose what the portrait is for:

- Personal assistant: extracts the portrait and events.
- Clone: also extracts your speaking style, which needs the [speaking style dependencies](installation.md#speaking-style-dependencies).

The page checks your setup and lists any missing settings with how to fix them. Chats in the generic CSV format can be uploaded on this page, see [Preparing Data](data.md#generic-csv-format). An API backend also needs the model's input token limit. Agree to send your chats to the model service and start the build. It runs in a background process, so closing the page doesn't stop it. The page shows the progress of each stage, and the top bar of other pages shows "Building n/N". The stages are:

1. Prepare the chat data: group text messages by contact and split them into sessions
2. Extract profile memories and event memories
3. Induce domains, then classify and summarize the memories, giving every fact a privacy level and a time scope
4. Group each domain's facts into topics and build the portrait hierarchy
5. Write domain summaries
6. Extract speaking style (`--purpose clone` only)

| Option | Description |
|---|---|
| `--output-dir <dir>` | Result directory, which must be new. Defaults to a timestamped directory under `dataset/res_csv/agent/profile_runs/` |
| `--max-context-tokens <limit>` | Input token limit for memory organization and the hierarchy. The Codex backend reads the model's context length by default. The API backend requires it |
| `--yes` | Skip the confirmation to send, for scripts |

When the build finishes and the portrait shown has no reviews yet, the web UI switches to the new one. If you have reviewed it, your reviews stay with that version and you switch with **Open the new portrait**. Every version is listed under Settings → Portrait versions, where you can switch back at any time. Only one build runs at a time.

Model answers are cached by input content in `dataset/res_csv/agent/llm_cache.sqlite`. After a failed or cancelled build, **Build again** reruns it in a new result directory and only the missing parts are requested. Delete the file to request everything again.

### Build from the command line

```bash
weclone-cli build-profile
```

It asks what the portrait is for, or pass `--purpose assistant` or `--purpose clone`. It runs the same checks as the web page first and lists every problem before it exits. Then it shows the model backend and asks you to confirm. The result also shows up under Portrait versions in the web UI.

## Review in the web UI

```bash
weclone-cli server                    # serve the version picked on the web, else the newest run
weclone-cli server --run <result-dir> # serve a specific run for this start
weclone-cli server --demo en          # serve the bundled synthetic portrait in English
```

`--demo` gives a look at the web UI before you build your own portrait. Reviews and the password are kept in a temporary directory that is removed on exit, apart from your own data. The sign-in page shows the password, `demo`.

Open http://127.0.0.1:5175 and sign in with the shared password (see [Security](security.md#password)). The web UI follows your browser's language, English or Chinese. The 中/EN switch in the top bar changes it:

| Page | Purpose |
|---|---|
| Portrait (`#/`) | Browse the portrait by domain and topic, adopt, edit or reject each fact, and see the chat each fact came from |
| Style (`#/style`) | View the speaking style extracted with `--purpose clone` |
| Summaries (`#/summary`) | Generate and approve a summary per domain and privacy level, and copy them for a personal assistant |
| Assistants (`#/assistant`) | Choose the level and domains assistants can read, and generate their token |
| Bots (`#/bots`) | Generate the token chat bots call the clone with. See [Chat bots](chatbot.md) |
| Build portrait (`#/build`) | Check the setup, build a new portrait from your chats and follow its progress. Switch versions under Settings → Portrait versions |

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

For assistants without MCP, choose a privacy level on the Summaries page, copy the export text in one click and paste it into the assistant's "About me", custom instructions or USER.md. The text holds approved summaries only. What you paste can't be taken back, so read it over before copying.

> [!WARNING]
> At the **All** level, private and sensitive entries also go to the cloud model behind the assistant.
