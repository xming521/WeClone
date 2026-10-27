import pytest

from weclone.data.agent import llm_cache


@pytest.fixture(autouse=True)
def isolated_llm_cache(tmp_path_factory, monkeypatch):
    """Every test gets an empty LLM cache of its own instead of the project's dataset/ one."""
    monkeypatch.setattr(llm_cache, "DEFAULT_PATH", tmp_path_factory.mktemp("llm_cache") / "llm_cache.sqlite")


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch):
    """Tests only reach local fakes. A shell proxy must not apply, and httpx rejects socks5h:// ones outright."""
    for name in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
