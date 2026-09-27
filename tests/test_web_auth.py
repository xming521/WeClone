import sqlite3

import pytest
from fastapi.testclient import TestClient

from weclone.server import auth
from weclone.server.profile_review import create_app

HEADERS = {"X-WeClone-Request": "1"}


@pytest.fixture
def web(tmp_path, capsys):
    source = tmp_path / "input.json"
    source.write_text('{"dimensions":[],"facts":[],"sources":{}}')
    options = {"database": tmp_path / "profile.sqlite3", "source": source, "static_dir": tmp_path / "web"}
    client = TestClient(create_app(**options))
    password = capsys.readouterr().out.strip().split("：")[-1]
    return client, password, options


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/profile"), ("GET", "/api/avatar-profile?download=true"),
    ("GET", "/api/facts/example/history"), ("POST", "/api/facts"),
    ("PATCH", "/api/facts/example"), ("POST", "/api/reviews/batch"),
])
def test_private_routes_reject_anonymous(web, method, path):
    client, _, _ = web
    response = client.request(method, path, headers=HEADERS)
    assert response.status_code == 401
    assert response.headers["cache-control"] == "no-store"


def test_session_persists_without_sliding_and_expires(web, monkeypatch, capsys):
    client, password, options = web
    now = 1_800_000_000
    monkeypatch.setattr(auth.time, "time", lambda: now)
    response = client.post("/api/auth/login", json={"password": password}, headers=HEADERS)
    assert response.status_code == 200
    expires = response.json()["expires_at"]
    assert expires == now + 30 * 24 * 60 * 60
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie and "Max-Age=2592000" in cookie
    token = client.cookies.get(auth.COOKIE)
    auth_path = options["database"].with_suffix(".auth.sqlite3")
    assert auth_path.stat().st_mode & 0o777 == 0o600
    assert password.encode() not in auth_path.read_bytes()
    assert token.encode() not in auth_path.read_bytes()

    restarted = TestClient(create_app(**options))
    assert capsys.readouterr().out == ""
    restarted.cookies.update(client.cookies)
    now = expires - 1
    assert restarted.get("/api/profile").status_code == 200
    assert restarted.get("/api/auth/session").json()["expires_at"] == expires
    now = expires
    assert restarted.get("/api/profile").status_code == 401


def test_logout_and_password_reset_revoke_sessions(web, capsys):
    client, password, options = web
    assert client.post("/api/auth/login", json={"password": password}, headers=HEADERS).status_code == 200
    old = dict(client.cookies)
    assert client.post("/api/auth/logout", json={}, headers=HEADERS).status_code == 200
    client.cookies.update(old)
    assert client.get("/api/profile").status_code == 401
    assert client.post("/api/auth/login", json={"password": password}, headers=HEADERS).status_code == 200
    auth.AuthStore(options["database"], reset=True)
    replacement = capsys.readouterr().out.strip().split("：")[-1]
    assert replacement != password
    assert client.get("/api/profile").status_code == 401
    assert client.post("/api/auth/login", json={"password": password}, headers=HEADERS).status_code == 401
    assert client.post("/api/auth/login", json={"password": replacement}, headers=HEADERS).status_code == 200


def test_login_rate_limit_csrf_and_secure_cookie(web, monkeypatch):
    client, password, options = web
    assert client.post("/api/auth/login", json={"password": password}).status_code == 403
    for _ in range(5):
        assert client.post("/api/auth/login", json={"password": "wrong"}, headers=HEADERS).status_code == 401
    assert client.post("/api/auth/login", json={"password": password}, headers=HEADERS).status_code == 429
    future = auth.time.time() + 61
    monkeypatch.setattr(auth.time, "time", lambda: future)
    secure = TestClient(create_app(**options), base_url="https://testserver")
    response = secure.post("/api/auth/login", json={"password": password}, headers=HEADERS)
    assert response.status_code == 200
    assert "Secure" in response.headers["set-cookie"]
    assert secure.post("/api/facts", json={}).status_code == 403
    with sqlite3.connect(options["database"].with_suffix(".auth.sqlite3")) as db:
        assert db.execute("SELECT failures FROM credentials").fetchone()[0] == 0
