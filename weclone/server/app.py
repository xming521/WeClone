"""Unified WeClone web and optional inference server."""

import shutil
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path

from weclone.server.profile_review import create_app as create_profile_app
from weclone.utils import secure_storage as secure
from weclone.utils.log import logger

DEMO_DATA = Path(__file__).resolve().parents[2] / "tests" / "tests_data"
DEMO_PROFILES = {"zh": "profile_hierarchy.demo.json", "en": "profile_hierarchy.demo.en.json"}
DEMO_PASSWORD = "demo"


def create_app(
    *,
    run: Path | None = None,
    static_dir: Path | None = None,
    inference: bool = False,
):
    """Serve one build-profile run, the newest one when run is None."""
    inference_app = None
    if inference:
        from weclone.server.api_service import create_inference_app

        inference_app = create_inference_app()
    app = create_profile_app(
        None,
        run / "profile_hierarchy" / "profile_hierarchy.json" if run is not None else None,
        static_dir,
        inference_router=inference_app.router if inference_app is not None else None,
    )
    if inference_app is not None:
        app.root_path = inference_app.root_path
    return app


@contextmanager
def demo_run(language: str):
    """A run holding a copy of the synthetic test profile, with plaintext storage and the password DEMO_PASSWORD
    in a temporary directory, removed on exit. Real runs and their password stay untouched."""
    with tempfile.TemporaryDirectory(prefix="weclone-demo-") as directory:
        run = Path(directory) / "run"
        source = run / "profile_hierarchy" / "profile_hierarchy.json"
        source.parent.mkdir(parents=True)
        shutil.copyfile(DEMO_DATA / DEMO_PROFILES[language], source)
        with secure.demo_storage(Path(directory) / "state"):
            secure.ensure_initialized(DEMO_PASSWORD)
            secure.lock()
            yield run


def serve(*, host="127.0.0.1", port=5175, run=None, inference=False, demo=None):
    import uvicorn

    with demo_run(demo) if demo else nullcontext(run) as run:
        app = create_app(run=run, inference=inference)
        if demo:
            logger.info(f"演示模式：合成测试数据，退出后删除。登录密码：{DEMO_PASSWORD}")
        uvicorn.run(app, host=host, port=port)
