"""Load inference models only after a request passes the web authentication gate."""

import asyncio
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from starlette.concurrency import run_in_threadpool

from weclone.utils import secure_storage as secure
from weclone.utils.config import create_config_by_arg_type, load_base_config


def _encrypted_personal_source(path: Path) -> None:
    path = secure.resolve_path(path)
    if not path.exists():
        return
    files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
    if not any(secure.is_encrypted_file(item) for item in files if not item.name.endswith(".lock")):
        raise secure.SecurityError("安全模式不能加载旧明文个人模型或 adapter，请重新训练生成加密模型。")


def _validate_model_sources(config, base) -> None:
    if not secure.is_encrypted_mode():
        return
    # This also prevents third-party initialization from prompting on the server.
    secure.get_unlocked_key()
    adapters = config.get("adapter_name_or_path")
    if isinstance(adapters, str):
        for path in adapters.split(","):
            if path.strip():
                _encrypted_personal_source(Path(path.strip()))
    elif isinstance(adapters, list):
        for path in adapters:
            _encrypted_personal_source(Path(path))
    outputs = [getattr(base.common_args, "adapter_name_or_path", None)]
    for section in (getattr(base, "train_pt_args", None), getattr(base, "train_sft_args", None)):
        outputs.append(getattr(section, "output_dir", None))
    model = config.get("model_name_or_path")
    if isinstance(model, str) and Path(model).exists():
        model_path = Path(model).resolve()
        for output in outputs:
            if isinstance(output, str) and output:
                output_path = Path(output).resolve()
                if model_path == output_path or output_path in model_path.parents:
                    _encrypted_personal_source(model_path)


class LazyInferenceModel:
    def __init__(self):
        self._model = None
        self._runtime = None
        self._generation = None
        self._lock = asyncio.Lock()
        self._lifespan = None
        self._api = None
        self._api_lifespan = None

    def __getattr__(self, name):
        if self._model is None:
            raise RuntimeError("推理模型尚未通过授权请求初始化")
        return getattr(self._model, name)

    def _load(self):
        from llamafactory.chat import ChatModel

        base = load_base_config()
        config = create_config_by_arg_type("api_service", base).model_dump(mode="json")
        _validate_model_sources(config, base)
        generation = secure.get_generation() if secure.is_encrypted_mode() else None
        runtime = ExitStack()
        try:
            config = runtime.enter_context(
                secure.sensitive_runtime(
                    config,
                    dataset_keys=(),
                    model_keys=("model_name_or_path", "adapter_name_or_path"),
                    output_keys=(),
                )
            )
            self._model = ChatModel(config)
            if generation is not None and secure.get_generation() != generation:
                raise secure.SecurityError("模型加载期间密码已重置，请重新生成个人模型。")
            self._generation = generation
            self._runtime = runtime
        except BaseException:
            runtime.close()
            self._model = None
            raise

    async def initialize(self):
        async with self._lock:
            if (
                self._model is not None
                and secure.is_encrypted_mode()
                and self._generation != secure.get_generation()
            ):
                await self._close()
                raise secure.SecurityError("密码已重置，请重新生成个人模型后重新启动推理服务。")
            if self._model is None:
                # Authentication middleware supplies the request key; threadpool copies its context.
                await run_in_threadpool(self._load)
                try:
                    self._lifespan = self._api_lifespan(self._api)
                    await self._lifespan.__aenter__()
                except BaseException:
                    await self._close()
                    raise

    async def _close(self):
        try:
            if self._lifespan is not None:
                lifespan, self._lifespan = self._lifespan, None
                await lifespan.__aexit__(None, None, None)
        finally:
            self._model = None
            self._generation = None
            if self._runtime is not None:
                runtime, self._runtime = self._runtime, None
                await run_in_threadpool(runtime.close)

    async def close(self):
        async with self._lock:
            await self._close()


def create_inference_app():
    from llamafactory.api.app import create_app

    model = LazyInferenceModel()
    original = create_app(model)
    model._api = original
    model._api_lifespan = original.router.lifespan_context

    @asynccontextmanager
    async def deferred_lifespan(_app):
        # LLaMA-Factory's lifespan reads engine.name, so enter it only after loading.
        yield

    original.router.lifespan_context = deferred_lifespan

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            await model.close()

    app = FastAPI(root_path=original.root_path, lifespan=lifespan)
    app.state.lazy_model = model
    app.include_router(original.router, dependencies=[Depends(model.initialize)])
    return app
