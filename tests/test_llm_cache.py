import json

from weclone.data.agent import llm_cache
from weclone.utils import secure_storage


def test_answers_round_trip_and_the_first_answer_stays(tmp_path):
    cache = llm_cache.LLMCache(tmp_path / "cache.sqlite")
    assert cache.get_many(["a"]) == {}
    assert not (tmp_path / "cache.sqlite").exists()

    cache.put_many([("a", "classify", {"items": [1]}, "model-a"), ("b", "classify", [], "")])
    cache.put_many([("a", "classify", {"items": [2]}, "model-b")])
    assert cache.get_many(["a", "b", "c", "a"]) == {"a": {"items": [1]}, "b": []}
    # Another run opening the same file sees the same answers.
    assert llm_cache.LLMCache(tmp_path / "cache.sqlite").get_many(["a"]) == {"a": {"items": [1]}}


def test_lookups_cover_more_keys_than_one_query_holds(tmp_path):
    cache = llm_cache.LLMCache(tmp_path / "cache.sqlite")
    rows = [(str(i), "state_distill", i, "") for i in range(1203)]
    cache.put_many(rows)
    assert cache.get_many(str(i) for i in range(1210)) == {str(i): i for i in range(1203)}


def test_key_covers_every_part_in_order():
    assert llm_cache.cache_key("classify", "prompt", {"a": 1, "b": 2}) == llm_cache.cache_key(
        "classify", "prompt", {"b": 2, "a": 1}
    )
    assert llm_cache.cache_key("a", "b") != llm_cache.cache_key("b", "a")
    assert llm_cache.cache_key("classify", "prompt") != llm_cache.cache_key("classify", "prompt", None)


def test_default_cache_follows_the_module_path(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_cache, "DEFAULT_PATH", tmp_path / "shared.sqlite")
    llm_cache.LLMCache().put_many([("a", "topics", {"topics": []}, "")])
    assert (tmp_path / "shared.sqlite").is_file()


def test_encrypted_mode_keeps_answers_encrypted(tmp_path, monkeypatch):
    settings = tmp_path / "settings.jsonc"
    settings.write_text(
        json.dumps({"security_args": {"storage_mode": "encrypted", "state_dir": str(tmp_path / "state")}})
    )
    monkeypatch.setenv("WECLONE_CONFIG_PATH", str(settings))
    secure_storage.configure()
    secure_storage.ensure_initialized("cache-test")
    try:
        path = tmp_path / "cache.sqlite"
        llm_cache.LLMCache(path).put_many([("a", "peers", {"relation": "大学同学"}, "")])
        assert not path.exists()
        stored = secure_storage.encrypted_path(path)
        assert secure_storage.is_encrypted_file(stored)
        assert "大学同学".encode() not in stored.read_bytes()
        assert llm_cache.LLMCache(path).get_many(["a"]) == {"a": {"relation": "大学同学"}}
    finally:
        secure_storage.lock()
