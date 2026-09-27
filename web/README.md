# 画像审核

在项目根目录启动服务（默认 http://127.0.0.1:5175）：

```sh
source .venv2/bin/activate
weclone-cli server
```

首次启动从 `dataset/res_csv/agent/memory_organization/profile_hierarchy.json` 初始化同目录的 `profile_review.sqlite3`，之后使用数据库中的固定快照，不重新导入原文件。数据库包含审核状态、编辑内容、来源和历史；备份时保留此文件。

开发前端：

```sh
cd web
pnpm dev
```

打开 http://127.0.0.1:5174，API 自动代理到审核服务。仓库已包含构建好的 `web/dist`；启动审核服务后，可直接在 5175 端口使用页面。修改前端源码后运行 `pnpm build` 更新页面。

审核通过即纳入分身画像。修改后“保存”回到待审核，“保存并通过”直接通过；无修改的保存保留状态。“导出分身画像”只导出已通过内容，对应 `GET /api/avatar-profile`。此接口尚未接入聊天推理。

服务支持 `--database`、`--source`、`--host` 和 `--port` 参数；默认仅监听本机，供单用户使用。

同一端口启用配置中的本地推理模型：

```sh
weclone-cli server --inference
```

网页位于 `/`，画像管理接口位于 `/api/`，推理接口位于 `/v1/`，接口文档位于 `/docs`。默认不加载推理模型；`--inference` 会加载模型并占用相应显存。原来连接 8005 端口的推理客户端可使用 `weclone-cli server --inference --port 8005`，端口也可通过 `API_PORT` 设置。同端口部署不代表已将画像注入推理请求。
