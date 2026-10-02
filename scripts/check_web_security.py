"""Exercise shared-password web flows with temporary synthetic encrypted data."""

import json
import os
import tempfile
import time
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from httpx import Cookies

from weclone.server import api_service, auth
from weclone.server.app import create_app
from weclone.utils import secure_storage as secure


class WebSecurityChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="weclone-web-check-")
        self.root = Path(self.directory.name)
        self.old_config = os.environ.get("WECLONE_CONFIG_PATH")
        config = self.root / "settings.jsonc"
        config.write_text(
            json.dumps(
                {
                    "security_args": {
                        "storage_mode": "encrypted",
                        "state_dir": str(self.root / "state"),
                    }
                }
            )
        )
        secure.configure(config)
        secure.lock()
        self.options = {
            "database": self.root / "review.sqlite3",
            "source": self.root / "profile.json",
            "static_dir": self.root / "web",
        }
        self.headers = {"X-WeClone-Request": "1"}

    def tearDown(self):
        secure.lock()
        if self.old_config is None:
            os.environ.pop("WECLONE_CONFIG_PATH", None)
        else:
            os.environ["WECLONE_CONFIG_PATH"] = self.old_config
        self.directory.cleanup()

    def setup_owner(self, client):
        client.headers.update(self.headers)
        response = client.post("/api/auth/setup", json={"password": "x", "confirmation": "x"})
        self.assertEqual(response.status_code, 200, response.text)
        secure.unlock("x")
        secure.write_json(
            self.options["source"],
            {
                "dimensions": [
                    {
                        "dim": 1,
                        "name": "Synthetic",
                        "groups": [
                            {
                                "name": "Group",
                                "items": [{"attr": "Preference", "fact_indices": [0]}],
                            }
                        ],
                    }
                ],
                "facts": [
                    {"dim": 1, "attr": "Preference", "value": "synthetic-private-fact", "source_ids": ["s1"]}
                ],
                "sources": {"s1": {"id": "s1", "content": "synthetic-private-source"}},
            },
        )
        secure.lock()
        return response

    def test_setup_review_encryption_restart_and_expiry(self):
        with TestClient(create_app(**self.options)) as client:
            self.assertFalse(client.get("/api/auth/status").json()["initialized"])
            self.assertEqual(client.get("/api/profile").status_code, 401)
            self.assertFalse(secure.encrypted_path(self.options["database"]).exists())
            response = self.setup_owner(client)
            expires = response.json()["expires_at"]
            self.assertAlmostEqual(expires - time.time(), 7200, delta=5)
            self.assertIn("Max-Age=7200", response.headers["set-cookie"])
            profile = client.get("/api/profile")
            self.assertEqual(profile.status_code, 200, profile.text)
            first = profile.json()["facts"][0]
            approved = client.post(
                "/api/reviews/batch",
                json={
                    "items": [{"id": first["id"], "version": first["version"]}],
                    "status": "approved",
                },
            )
            self.assertEqual(approved.status_code, 200, approved.text)
            self.assertEqual(
                client.get("/api/avatar-profile").json()["facts"][0]["value"], "synthetic-private-fact"
            )
            self.assertFalse(self.options["database"].exists())
            ciphertext = secure.encrypted_path(self.options["database"]).read_bytes()
            self.assertTrue(ciphertext.startswith(secure.MAGIC))
            self.assertNotIn(b"synthetic-private", ciphertext)
            self.assertNotIn(
                b"synthetic-private", self.options["database"].with_suffix(".auth.sqlite3").read_bytes()
            )
            cookies = Cookies(client.cookies)
            self.assertEqual(client.post("/api/auth/logout", json={}).status_code, 200)
            client.cookies.update(cookies)
            self.assertEqual(client.get("/api/profile").status_code, 401)
            self.assertEqual(client.post("/api/auth/login", json={"password": "x"}).status_code, 200)
            cookies = Cookies(client.cookies)
        secure.encrypted_path(self.options["source"]).unlink()
        with TestClient(create_app(**self.options)) as restarted:
            restarted.headers.update(self.headers)
            restarted.cookies.update(cookies)
            self.assertEqual(restarted.get("/api/profile").status_code, 401)
            response = restarted.post("/api/auth/login", json={"password": "x"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(restarted.get("/api/profile").status_code, 200)
            with patch.object(auth.time, "time", return_value=response.json()["expires_at"]):
                self.assertEqual(restarted.get("/api/profile").status_code, 401)
                self.assertFalse(restarted.app.state.auth_store._keys)

    def test_rate_limit_csrf_password_change_and_reset(self):
        with TestClient(create_app(**self.options), base_url="https://testserver") as client:
            self.assertEqual(
                client.post("/api/auth/setup", json={"password": "x", "confirmation": "x"}).status_code, 403
            )
            self.assertIn("Secure", self.setup_owner(client).headers["set-cookie"])
            for _ in range(5):
                self.assertEqual(client.post("/api/auth/login", json={"password": "wrong"}).status_code, 401)
            self.assertEqual(client.post("/api/auth/login", json={"password": "x"}).status_code, 429)
            with patch.object(auth.time, "time", return_value=time.time() + 61):
                self.assertEqual(client.post("/api/auth/login", json={"password": "x"}).status_code, 200)
            secure.change_password("x", "y")
            self.assertEqual(client.get("/api/profile").status_code, 401)
            self.assertEqual(client.post("/api/auth/login", json={"password": "x"}).status_code, 401)
            self.assertEqual(client.post("/api/auth/login", json={"password": "y"}).status_code, 200)
            self.assertEqual(client.get("/api/profile").status_code, 200)
            secure.reset_password("z")
            secure.lock()
            self.assertEqual(client.get("/api/profile").status_code, 401)
            self.assertEqual(client.post("/api/auth/login", json={"password": "z"}).status_code, 200)
            self.assertEqual(client.get("/api/profile").status_code, 409)

    def test_inference_loads_after_login_and_cleans_up_on_shutdown(self):
        events = []

        class Model:
            def __init__(self, config):
                secure.get_unlocked_key()
                events.append("load")
                self.engine = SimpleNamespace(name="synthetic")

        def inference_factory(model):
            @asynccontextmanager
            async def lifespan(app):
                self.assertEqual(model.engine.name, "synthetic")
                events.append("start")
                yield
                events.append("stop")

            app = FastAPI(lifespan=lifespan)

            @app.get("/v1/models")
            def models():
                return {"model": model.engine.name}

            @app.post("/v1/chat/completions")
            def chat():
                def stream():
                    secure.get_unlocked_key()
                    yield "data: synthetic\n\n"

                return StreamingResponse(stream(), media_type="text/event-stream")

            return app

        api_module, chat_module = ModuleType("llamafactory.api.app"), ModuleType("llamafactory.chat")
        api_module.create_app = inference_factory
        chat_module.ChatModel = Model
        base = SimpleNamespace(common_args=SimpleNamespace(adapter_name_or_path=None))
        config = Mock()
        config.model_dump.return_value = {}
        with (
            patch.dict("sys.modules", {"llamafactory.api.app": api_module, "llamafactory.chat": chat_module}),
            patch.object(api_service, "load_base_config", return_value=base),
            patch.object(api_service, "create_config_by_arg_type", return_value=config),
        ):
            with TestClient(create_app(**self.options, inference=True)) as client:
                self.assertEqual(client.get("/v1/models").status_code, 401)
                self.assertEqual(events, [])
                self.setup_owner(client)
                response = client.get("/v1/models")
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(events, ["load", "start"])
                self.assertIn("data: synthetic", client.post("/v1/chat/completions", json={}).text)
            self.assertEqual(events, ["load", "start", "stop"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
