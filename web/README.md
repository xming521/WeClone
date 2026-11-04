# Web UI

The WeClone web UI covers portrait review, domain summaries, speaking style, and assistant and bot access. It is a React app built with Vite and served by `weclone-cli server`. See [Life portrait and assistants](../docs/en/profile.md) for the pages and the review workflow.

## Run

From the repository root:

```sh
weclone-cli server         # built pages, http://127.0.0.1:5175
weclone-cli dev            # Vite dev server on http://127.0.0.1:5174, backend on 5175
weclone-cli dev --demo     # the same, with the synthetic test portrait (password: demo)
```

`weclone-cli server` opens the newest run under `dataset/res_csv/agent/profile_runs/`. Pass `--run <run-directory>` to open another run, or `--demo` (`--demo en` for English) to open the synthetic portrait in `tests/tests_data/`. The demo copies it into a temporary directory that is removed on exit.

## Routes

| Path | Serves | Access |
|---|---|---|
| `/` | the web UI | |
| `/api/` | portrait, review, summaries, style, and access settings | shared-password login |
| `/mcp` | read-only MCP tools for assistants | assistant token |
| `/v1/` | OpenAI-compatible API of the fine-tuned clone, with `--inference` only | login or bot token |
| `/docs` | API documentation | |

The server listens on localhost only by default. See [Security](../docs/en/security.md) for the password, tokens, and encrypted mode.

## Development

```sh
pnpm --dir web install
pnpm --dir web lint
pnpm --dir web test
pnpm --dir web build    # updates weclone/web/dist
```

The built pages in `weclone/web/dist` are committed and shipped with the Python package, so run `pnpm --dir web build` after changing the frontend source.

## README picture

The animated portrait at the top of the repository README is the map opened with `?screenshot` in the address: the map alone, with larger labels and the nodes packed round them so that every one fits. `scripts/screenshot.mjs` records one wave cycle of it in headless Chrome as an animated WebP that loops without a seam:

```sh
weclone-cli server --demo en --port 5180
WECLONE_PASSWORD=demo node web/scripts/screenshot.mjs http://127.0.0.1:5180 atlas-light-en.webp
```

Pass `--locale zh-CN` or `--theme dark` for the other versions, or `--still` for a PNG. The script needs Chrome or chrome-headless-shell (set `CHROME` when it is not on `PATH`) and ffmpeg with libwebp. The README loads the pictures from the documentation site, where they are `docs/public/img/screenshots/atlas-{light,dark}{,-en}.webp` in the WeClone-docs repository.

## Translations

The UI is in Chinese and English. It follows the browser language until the 中/EN switch saves a choice. UI copy lives in `src/i18n/locales/<language>/translation.json`, with `zh-CN` as the source catalog.

- After adding or renaming keys, update `en` as well and run `pnpm i18n:types`.
- `pnpm test` checks that both catalogs have the same keys, variables, and tags. ESLint rejects Chinese text in `src/` outside the catalogs.
- Do not run `i18next-cli extract`, because it drops keys indexed by variables.
- Server errors carry a `code` that the UI translates. Their `detail` stays in Chinese for other clients.
- Portrait content such as facts, summaries, and chats is shown as written.
