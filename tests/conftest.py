import pytest

from weclone.data.agent import llm_cache


@pytest.fixture(autouse=True)
def isolated_llm_cache(tmp_path_factory, monkeypatch):
    """Every test gets an empty LLM cache of its own instead of the project's dataset/ one."""
    monkeypatch.setattr(llm_cache, "DEFAULT_PATH", tmp_path_factory.mktemp("llm_cache") / "llm_cache.sqlite")
