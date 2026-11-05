import json

import pytest

from weclone.data.agent.contacts import assign_contact_ids, contact_key
from weclone.utils import secure_storage


def test_contact_key_depends_on_platform_and_talker_only():
    assert contact_key("chat", "wxid_a") == contact_key("chat", "wxid_a")
    assert len({contact_key("chat", "wxid_a"), contact_key("chat", "wxid_b"), contact_key("telegram", "wxid_a")}) == 3
    with pytest.raises(ValueError):
        contact_key("chat", "")


def test_registry_keeps_ids_and_only_grows(tmp_path):
    path = tmp_path / "contacts.json"
    assert assign_contact_ids(["a", "b", "a"], path) == {"a": "P1", "b": "P2"}
    assert assign_contact_ids(["c", "b"], path) == {"c": "P3", "b": "P2"}
    assert json.loads(path.read_text()) == {"a": "P1", "b": "P2", "c": "P3"}
    path.write_text(json.dumps({"a": "P1", "b": "P1"}))
    with pytest.raises(ValueError, match="reuses"):
        assign_contact_ids(["a"], path)


def test_a_password_reset_sets_the_unreadable_registry_aside(tmp_path, monkeypatch):
    settings = tmp_path / "settings.jsonc"
    settings.write_text(
        json.dumps({"security_args": {"storage_mode": "encrypted", "state_dir": str(tmp_path / "state")}})
    )
    monkeypatch.setenv("WECLONE_CONFIG_PATH", str(settings))
    secure_storage.ensure_initialized("contacts-test")
    try:
        path = tmp_path / "contacts.json"
        assert assign_contact_ids(["a", "b"], path) == {"a": "P1", "b": "P2"}
        old = secure_storage.encrypted_path(path).read_bytes()
        generation = secure_storage.get_generation()
        secure_storage.reset_password("contacts-reset")
        assert assign_contact_ids(["b"], path) == {"b": "P1"}
        assert (tmp_path / f"contacts.json.enc.previous-{generation}").read_bytes() == old
    finally:
        secure_storage.lock()
