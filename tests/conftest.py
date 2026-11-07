import json

import pytest

from weclone.data.agent import llm_cache, profile_pipeline
from weclone.server import chat_upload
from weclone.utils import secure_storage


@pytest.fixture(autouse=True)
def isolated_llm_cache(tmp_path_factory, monkeypatch):
    """Every test gets an empty LLM cache of its own instead of the project's dataset/ one."""
    monkeypatch.setattr(llm_cache, "DEFAULT_PATH", tmp_path_factory.mktemp("llm_cache") / "llm_cache.sqlite")


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path_factory, monkeypatch):
    """Every test gets empty runs, uploaded chats and upload staging folders of its own, so a server it starts neither
    opens the project's latest build nor clears an upload under way there."""
    root = tmp_path_factory.mktemp("dataset")
    monkeypatch.setattr(profile_pipeline, "RUNS_DIR", root / "runs")
    monkeypatch.setattr(profile_pipeline, "UPLOAD_DIR", root / "uploaded")
    monkeypatch.setattr(chat_upload, "STAGING_DIR", root / "uploads")


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch):
    """Tests only reach local fakes. A shell proxy must not apply, and httpx rejects socks5h:// ones outright."""
    for name in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


@pytest.fixture
def owner_password():
    """The password that tests using `storage` give the web UI's first-time setup."""
    return {"password": "build-test", "confirmation": "build-test"}


@pytest.fixture
def storage(tmp_path, monkeypatch):
    """Returns configure(mode="plaintext", extra=None), which points the settings, the state directory and the runs
    and uploaded-chats folders at tmp_path and returns the settings file. `extra` adds sections to the settings."""

    def configure(mode="plaintext", extra=None):
        # One file per mode: settings are cached by path and modification time.
        settings = tmp_path / f"settings.{mode}.jsonc"
        settings.write_text(
            json.dumps(
                {
                    "security_args": {"storage_mode": mode, "state_dir": str(tmp_path / "state")},
                    **(extra or {}),
                }
            )
        )
        monkeypatch.setenv("WECLONE_CONFIG_PATH", str(settings))
        monkeypatch.setattr(profile_pipeline, "RUNS_DIR", tmp_path / "runs")
        monkeypatch.setattr(profile_pipeline, "UPLOAD_DIR", tmp_path / "uploaded")
        secure_storage.configure()
        return settings

    return configure
