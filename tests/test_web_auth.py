import time

import pytest
from fastapi.testclient import TestClient

from weclone.server.profile_review import create_app

HEADERS = {"X-WeClone-Request": "1"}


@pytest.fixture
def web(tmp_path, storage):
    storage()
    source = tmp_path / "input.json"
    source.write_text('{"domains":[],"dimensions":[],"facts":[],"sources":{}}')
    return TestClient(create_app(tmp_path / "profile.sqlite3", source, tmp_path / "web"))


@pytest.mark.parametrize("method,path", [("GET", "/api/profile"), ("POST", "/api/facts")])
def test_private_routes_reject_anonymous(web, method, path):
    response = web.request(method, path, headers=HEADERS)
    assert response.status_code == 401
    assert response.json()["code"] == "login_required"
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
