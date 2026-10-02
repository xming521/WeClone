# Profile Review

Start the service from the repository root (default: http://127.0.0.1:5175):

```sh
source .venv2/bin/activate
weclone-cli server
```

By default, the service keeps using an existing profile or review database in `dataset/res_csv/agent/memory_organization/`. Otherwise, it selects the newest timestamped run in `dataset/res_csv/agent/profile_runs/` that has a `profile_hierarchy/profile_hierarchy.json` file (`.json.enc` in encrypted mode) or a saved review database. Incomplete runs are skipped. Selection occurs at service startup.

The service initializes `profile_review.sqlite3` alongside the selected profile (`.sqlite3.enc` in encrypted mode). Later starts use that database's snapshot and do not re-import the source file. The database stores review statuses, edits, sources, and history; include it in backups. Use `weclone-cli server --source <profile_hierarchy.json.enc>` to select a different profile, including results generated with `build-profile --output-dir`. Its review database is stored alongside that source unless `--database` is specified.

For frontend development:

```sh
cd web
pnpm dev
```

Open http://127.0.0.1:5174. API requests are automatically proxied to the review service. The repository includes built assets in `weclone/web/dist` and packages them with the Python distribution. After starting the review service, the page is also available on port 5175. Run `pnpm build` after changing the frontend source to update the page.

Approved items become part of the avatar profile. After an edit, "Save" returns the item to pending review, while "Save and Approve" approves it directly. Saving without changes preserves its status. "Export Avatar Profile" exports only approved items through `GET /api/avatar-profile`. This endpoint is not yet connected to chat inference.

Profile cards, hover previews, and change history display the target role label `B` as `我`; search matches both the original and displayed wording. The shared `displayProfileText` formatter excludes Latin identifiers, numbers, hyphenated model names, and common terms such as `B站`, `B超`, `B型`, and `维生素B`. These lexical rules are independent of profile samples; extend the term exceptions when new spellings are encountered. Original evidence, chat messages, editor contents, stored data, and exported profiles retain their original wording.

The review UI labels these decisions “采纳” and “不采纳”; records remain available when they are not adopted. Each fact displays the source confidence and importance means on four-segment scales, and individual evidence entries show their original scores. Missing scores and manually created facts display “未评分”. Editing a fact does not recalculate its source scores.

The detail sidebar separates “总览” (counts, review progress, and child navigation) from “画像审核” (status filtering and review cards). Selecting a dimension or topic opens the overview; selecting an attribute opens review. Switching tabs retains the selected scope and review filter. The review tab badge counts pending facts within that scope across all statuses.

“查看原文” opens the chat sample used for extraction alongside the review panel. The authenticated `GET /api/sources/{source_id}/chat` endpoint resolves the saved origin and checks the sample ID and extracted memory before returning messages. It reads encrypted sources through the existing session key and does not cache responses. Keep the extraction output files referenced by `sources[*].origins`; older sources without these references or missing/changed files display an unavailable message. The panel shows the preserved extraction input, including any earlier preprocessing, and only displays message timestamps when they exist.

The service accepts `--database`, `--source`, `--host`, and `--port`. By default, it listens only on localhost for single-user use.

To enable the configured local inference model on the same port:

```sh
weclone-cli server --inference
```

The web page is served at `/`, the profile management API at `/api/`, the inference API at `/v1/`, and the API documentation at `/docs`. The inference model is not loaded by default; `--inference` loads it and uses GPU memory. Inference clients that previously connected to port 8005 can use `weclone-cli server --inference --port 8005`. The port can also be set through `API_PORT`. Serving both APIs on one port does not inject the profile into inference requests.
