import pytest
from fastapi.testclient import TestClient

from weclone.server.profile_review import create_app

HEADERS = {"X-WeClone-Request": "1"}


@pytest.fixture
def web(tmp_path, storage):
    storage()
    source = tmp_path / "run" / "profile_hierarchy" / "profile_hierarchy.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"domains":[],"dimensions":[],"facts":[],"sources":{}}')
    client = TestClient(create_app(None, source, tmp_path / "web"))
    body = {"password": "old", "confirmation": "old"}
    assert client.post("/api/auth/setup", json=body, headers=HEADERS).status_code == 200
    return client


def test_password_change_keeps_this_session_and_ends_the_others(web, tmp_path):
    other = TestClient(web.app)
    assert other.post("/api/auth/login", json={"password": "old"}, headers=HEADERS).status_code == 200
    change = {"old_password": "old", "password": "new", "confirmation": "new"}
    assert web.post("/api/auth/password", json=change, headers=HEADERS).status_code == 200
    assert web.get("/api/run").status_code == 200
    assert other.get("/api/run").status_code == 401
    assert (
        web.post("/api/auth/login", json={"password": "old"}, headers=HEADERS).json()["code"]
        == "password_wrong"
    )
    assert web.post("/api/auth/login", json={"password": "new"}, headers=HEADERS).status_code == 200


def test_password_change_checks_the_old_password_and_the_confirmation(web):
    wrong = web.post(
        "/api/auth/password",
        json={"old_password": "x", "password": "n", "confirmation": "n"},
        headers=HEADERS,
    )
    assert wrong.json()["code"] == "password_wrong"
    mismatch = web.post(
        "/api/auth/password",
        json={"old_password": "old", "password": "n", "confirmation": "m"},
        headers=HEADERS,
    )
    assert mismatch.json()["code"] == "password_mismatch"
    # A wrong old password is no sign-out: the session still works.
    assert web.get("/api/run").status_code == 200


def test_session_length_applies_to_later_sign_ins(web):
    assert web.put("/api/auth/session-length", json={"seconds": 60}, headers=HEADERS).status_code == 422
    assert web.put("/api/auth/session-length", json={"seconds": 8 * 3600}, headers=HEADERS).status_code == 200
    assert web.get("/api/auth/status").json()["session_seconds"] == 8 * 3600
    response = web.post("/api/auth/login", json={"password": "old"}, headers=HEADERS)
    assert 8 * 3600 - 5 < response.json()["expires_at"] - __import__("time").time() <= 8 * 3600
    assert "Max-Age=28" in response.headers["set-cookie"]


def test_run_info_names_the_run(web):
    info = web.get("/api/run").json()
    assert info["name"] == "run"
    assert info["generated_at"] and info["reviewed_at"] is None
    assert info["demo"] is False and info["style"] is False


def test_demo_keeps_its_settings_and_paths(web, tmp_path):
    demo = TestClient(
        create_app(
            None,
            tmp_path / "run" / "profile_hierarchy" / "profile_hierarchy.json",
            tmp_path / "web",
            demo_password="old",
        )
    )
    assert demo.post("/api/auth/login", json={"password": "old"}, headers=HEADERS).status_code == 200
    info = demo.get("/api/run").json()
    assert info["demo"] is True and info["path"] is None
    response = demo.put("/api/auth/session-length", json={"seconds": 8 * 3600}, headers=HEADERS)
    assert response.json()["code"] == "demo_settings_fixed"
