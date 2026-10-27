import json
import time

import pytest
from fastapi.testclient import TestClient

from weclone.server.profile_review import create_app
from weclone.utils import secure_storage

HEADERS = {"X-WeClone-Request": "1"}


@pytest.fixture
def web(tmp_path, monkeypatch):
    settings = tmp_path / "settings.jsonc"
    settings.write_text(
        json.dumps({"security_args": {"storage_mode": "plaintext", "state_dir": str(tmp_path / "state")}})
    )
    monkeypatch.setenv("WECLONE_CONFIG_PATH", str(settings))
    secure_storage.configure()
    source = tmp_path / "input.json"
    source.write_text('{"domains":[],"dimensions":[],"facts":[],"sources":{}}')
    return TestClient(create_app(tmp_path / "profile.sqlite3", source, tmp_path / "web"))


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/profile"), ("GET", "/api/avatar-profile?download=true"),
    ("GET", "/api/facts/example/history"), ("POST", "/api/facts"),
    ("PATCH", "/api/facts/example"), ("POST", "/api/reviews/batch"),
])
def test_private_routes_reject_anonymous(web, method, path):
    response = web.request(method, path, headers=HEADERS)
    assert response.status_code == 401
    assert response.headers["cache-control"] == "no-store"


def test_running_server_prunes_expired_access_and_forgets_it_on_shutdown(web):
    store = web.app.state.auth_store
    with web:
        store.allow_assistant(None, -1)
        deadline = time.monotonic() + 5
        while store._assistant is not None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert store._assistant is None
        store.allow_assistant(None, 3600)
    assert store._assistant is None
