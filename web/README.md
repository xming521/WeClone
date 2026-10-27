# Profile Review

Start the service from the repository root (default: http://127.0.0.1:5175):

```sh
source .venv2/bin/activate
weclone-cli server
```

By default, the service selects the newest timestamped run in `dataset/res_csv/agent/profile_runs/` that has a `profile_hierarchy/profile_hierarchy.json` file (`.json.enc` in encrypted mode) or a saved review database. Incomplete runs are skipped. Selection occurs at service startup.

The service initializes `profile_review.sqlite3` alongside the selected profile (`.sqlite3.enc` in encrypted mode). Later starts use that database's snapshot and do not re-import the source file. The database stores review statuses, edits, sources, and history; include it in backups. Use `weclone-cli server --run <run-directory>` to serve a different run, including one generated with `build-profile --output-dir`. Its review database is stored in that run's `profile_hierarchy/` directory. The style page shows the same run's `style/` directory.

For frontend development:

```sh
cd web
pnpm dev
```

Open http://127.0.0.1:5174. API requests are automatically proxied to the review service. The repository includes built assets in `weclone/web/dist` and packages them with the Python distribution. After starting the review service, the page is also available on port 5175. Run `pnpm build` after changing the frontend source to update the page.

The UI is in Chinese and English. It follows the browser language (Chinese for any `zh` variant, English otherwise) until the 中/EN switch in the top bar or on the sign-in page saves a choice in `localStorage` (`weclone.locale`). UI copy lives in `src/i18n/locales/<language>/translation.json`, with `zh-CN` as the source catalog. After adding or renaming keys, update `en` as well and run `pnpm i18n:types`. `pnpm test` checks that both catalogs have the same keys, variables and tags, and ESLint rejects Chinese text in `src/` outside the catalogs. Do not run `i18next-cli extract`: it drops keys indexed by variables. Profile content (facts, summaries, chats) is shown as written. Server errors the UI shows carry a `code` that the page words in its own language, while `detail` keeps the Chinese text for other clients.

Approved items become part of the avatar profile. After an edit, "Save" returns the item to pending review, while "Save and adopt" approves it directly. Saving without changes preserves its status. "Export this view" asks for confirmation with the current review view name and fact count, then downloads `profile_<view>.json` from that view's displayed data. Export follows the global All/Pending/Approved/Rejected filter and includes the matching hierarchy and referenced sources, with each exported fact carrying its review status; selecting a node or expanding the graph does not narrow the export. The file uses the data captured when the confirmation opens. `GET /api/avatar-profile` still returns only approved items and is not yet connected to chat inference.

Profile cards, hover previews, and change history display the target role label `B` as `我`; search matches both the original and displayed wording. The shared `displayProfileText` formatter excludes Latin identifiers, numbers, hyphenated model names, and common terms such as `B站`, `B超`, `B型`, and `维生素B`. These lexical rules are independent of profile samples; extend the term exceptions when new spellings are encountered. Original evidence, chat messages, editor contents, stored data, and exported profiles retain their original wording.

The review UI labels these decisions Adopt and Reject; records remain available when they are not adopted. Each fact displays the source confidence and importance means on four-segment scales, and individual evidence entries show their original scores. Missing scores and manually created facts display Unrated. Editing a fact does not recalculate its source scores.

Profiles built with domains (`build-profile` writes a `domains` list) can switch the left panel between grouping by dimension and by domain. The domain view regroups the same facts as domain → topic → attribute, merging a topic's attributes across dimensions; each review card then names its dimension. Review status, edits and export are unchanged, the choice is remembered in the browser, and profiles without domains show only the dimension view.

The detail sidebar separates Overview (counts, review progress, and child navigation) from Review (status filtering and review cards). Selecting a dimension or topic opens the overview; selecting an attribute opens review. Switching tabs retains the selected scope and review filter. The review tab badge counts pending facts within that scope across all statuses.

Profiles whose organizer assigned privacy levels (`sensitivity`: `public`, `normal`, `private` or `sensitive`) show a level badge on each fact and on each evidence entry, and the overview shows the level distribution within the selected scope. In the review tab, level toggles narrow the list to the chosen levels (none chosen shows all; the map is not filtered), and Select all selects every fact of the filtered list, including those not loaded yet. A batch that includes facts not shown asks for confirmation with the count. Each evidence entry lists its extraction tags; a fact card lists the tags of its sources, the most shared first. Facts and sources without a level show no badge.

View chat opens the chat sample used for extraction alongside the review panel. The authenticated `GET /api/sources/{source_id}/chat` endpoint resolves the saved origin and checks the sample ID and extracted memory before returning messages. It reads encrypted sources through the existing session key and does not cache responses. Keep the extraction output files referenced by `sources[*].origins`; older sources without these references or missing/changed files display an unavailable message. The panel shows the preserved extraction input, including any earlier preprocessing, and only displays message timestamps when they exist.

The service accepts `--database`, `--source`, `--host`, and `--port`. By default, it listens only on localhost for single-user use.

To enable the configured local inference model on the same port:

```sh
weclone-cli server --inference
```

The web page is served at `/`, the profile management API at `/api/`, the inference API at `/v1/`, and the API documentation at `/docs`. The inference model is not loaded by default; `--inference` loads it and uses GPU memory. Inference clients that previously connected to port 8005 can use `weclone-cli server --inference --port 8005`. The port can also be set through `API_PORT`. Serving both APIs on one port does not inject the profile into inference requests.

Requests to `/v1/` that carry the owner's session cookie, such as `weclone-cli test-model`, pass the usual login check. Requests without it, such as chat bot platforms, must send the token generated on the bot access page (`#/bots`) as their API key (`Authorization: Bearer`). All bots share that one token, stored only as a hash in the state directory (`bots.sqlite3`). Requests that carry an `Origin` header are refused, so web pages cannot use it. In encrypted mode, generating the token also stores the data key sealed under the token in the same file. A bot request opens it with the token it presents to load the model and generate replies, so no one has to be logged in, including after a restart. A password change keeps the token working. A password reset starts a new key, and the token must be generated again. Whoever holds both the token and `bots.sqlite3` can decrypt all data.
