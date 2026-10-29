# 贡献指南 | Contributing

[简体中文](#简体中文)｜[English](#english)

## 简体中文

欢迎提交 Issue 和 Pull Request，也欢迎帮助审核 PR。新功能请先通过 Issue 讨论。

### 开发环境

```bash
uv pip install --group main --group dev -e .
pre-commit install
```

项目使用 `pytest` 测试，`pyright` 检查类型，`ruff` 检查代码格式。

### 测试

提交代码前运行 `pytest tests`，确保所有测试通过。说话风格的测试另需安装[说话风格依赖](zh/installation.md#说话风格依赖)，没有 `models/LTPbase` 时自动跳过。

端到端的微调测试默认不运行，用 `pytest -m e2e tests` 运行，需要 GPU 和 `tests/configs/` 里写的本地模型。它的聊天、数据集、模型输出和密钥状态都放在临时目录，不会移动或覆盖项目里的数据。

### 前端

网页源码在 `web/`，使用 pnpm 管理依赖：

```bash
pnpm --dir web install
weclone-cli dev   # 前端 http://127.0.0.1:5174，后端 http://127.0.0.1:5175
weclone-cli dev --demo   # 后端改用合成测试画像，登录密码 demo
```

仓库中包含构建好的页面 `weclone/web/dist`，随 Python 包一起发布。修改前端源码后运行 `pnpm --dir web build` 更新。

## English

Issues and pull requests are welcome, and so is help reviewing PRs. Please discuss new features in an issue first.

### Development environment

```bash
uv pip install --group main --group dev -e .
pre-commit install
```

The project uses `pytest` for tests, `pyright` for type checking and `ruff` for code style.

### Tests

Run `pytest tests` before submitting and make sure all tests pass. The speaking style tests also need the [speaking style dependencies](en/installation.md#speaking-style-dependencies) and are skipped when `models/LTPbase` is missing.

The end-to-end fine-tuning tests are skipped by default. Run them with `pytest -m e2e tests`. They need a GPU and the local models named in `tests/configs/`. They keep their chats, dataset, model output and key state in temporary directories, so the project's data is never moved or overwritten.

### Frontend

The web UI source is in `web/` and uses pnpm:

```bash
pnpm --dir web install
weclone-cli dev   # frontend http://127.0.0.1:5174, backend http://127.0.0.1:5175
weclone-cli dev --demo   # backend serves the synthetic test profile, password: demo
```

The built pages in `weclone/web/dist` are committed and shipped with the Python package. Run `pnpm --dir web build` after changing the frontend source to update them.
