# Profile Review

Start the service from the repository root (default: http://127.0.0.1:5175):

```sh
source .venv2/bin/activate
weclone-cli server
```

On first startup, the service initializes `profile_review.sqlite3` alongside `dataset/res_csv/agent/memory_organization/profile_hierarchy.json`. Later starts use the snapshot stored in the database and do not re-import the source file. The database stores review statuses, edits, sources, and history; include it in backups.

For frontend development:

```sh
cd web
pnpm dev
```

Open http://127.0.0.1:5174. API requests are automatically proxied to the review service. The repository includes built assets in `weclone/web/dist` and packages them with the Python distribution. After starting the review service, the page is also available on port 5175. Run `pnpm build` after changing the frontend source to update the page.

Approved items become part of the avatar profile. After an edit, "Save" returns the item to pending review, while "Save and Approve" approves it directly. Saving without changes preserves its status. "Export Avatar Profile" exports only approved items through `GET /api/avatar-profile`. This endpoint is not yet connected to chat inference.

The service accepts `--database`, `--source`, `--host`, and `--port`. By default, it listens only on localhost for single-user use.

To enable the configured local inference model on the same port:

```sh
weclone-cli server --inference
```

The web page is served at `/`, the profile management API at `/api/`, the inference API at `/v1/`, and the API documentation at `/docs`. The inference model is not loaded by default; `--inference` loads it and uses GPU memory. Inference clients that previously connected to port 8005 can use `weclone-cli server --inference --port 8005`. The port can also be set through `API_PORT`. Serving both APIs on one port does not inject the profile into inference requests.
