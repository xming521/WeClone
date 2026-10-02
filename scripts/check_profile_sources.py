"""Check source-chat lookup and profile review using isolated synthetic data."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from weclone.server.profile_review import create_app
from weclone.utils import secure_storage as secure


class ProfileSourceChecks(unittest.TestCase):
    def test_source_chat_and_review(self):
        for mode in ("plaintext", "encrypted"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                old_config = os.environ.get("WECLONE_CONFIG_PATH")
                try:
                    self.check_flow(Path(directory), mode)
                finally:
                    secure.lock()
                    if old_config is None:
                        os.environ.pop("WECLONE_CONFIG_PATH", None)
                    else:
                        os.environ["WECLONE_CONFIG_PATH"] = old_config

    def check_flow(self, root, mode):
        config = root / "settings.jsonc"
        config.write_text(
            json.dumps({"security_args": {"storage_mode": mode, "state_dir": str(root / "state")}})
        )
        secure.configure(config)
        secure.lock()
        profile_path, chat_path = root / "profile.json", root / "chat.json"
        sample = {
            "id": "stable-sample",
            "chat_with": "合成联系人",
            "time": "2026-10-01T20:14:00",
            "messages": [
                {"role": "system", "content": "not a chat message"},
                {"role": "user", "content": "先说原因吗？", "time": "2026-10-01T20:14:01"},
                {"role": "assistant", "content": "先看结论。\n\r\n再说理由。<script>text only</script>"},
            ],
            "state_memories": [{"content": "喜欢先看结论", "confidence": 4, "importance": 3}],
            "event_memories": {"surface_events": [{"surface_event": "讨论表达方式"}]},
        }
        origin = {
            "file": str(chat_path),
            "sample_id": "stable-sample",
            "sample_index": 0,
            "kind": "S",
            "memory_index": 0,
        }
        sources = {
            "S:1": {"content": "喜欢先看结论", "confidence": 4, "importance": 3, "origins": [origin]},
            "ES:2": {"content": "讨论表达方式", "origins": [{**origin, "kind": "ES"}]},
            "legacy": {"content": "旧来源"},
        }
        profile = {
            "dimensions": [
                {
                    "dim": 8,
                    "name": "偏好",
                    "groups": [{"name": "沟通", "items": [{"attr": "交流偏好", "fact_indices": [0]}]}],
                }
            ],
            "facts": [
                {
                    "dim": 8,
                    "attr": "交流偏好",
                    "value": "喜欢先看结论",
                    "source_ids": list(sources),
                    "source_confidence_mean": 4,
                    "source_importance_mean": 3,
                    "support_count": 1,
                }
            ],
            "sources": sources,
        }
        with TestClient(create_app(root / "review.sqlite3", profile_path, root / "no-static")) as client:
            url = "/api/sources/S:1/chat"
            self.assertEqual(client.get(url).status_code, 401)
            client.headers["X-WeClone-Request"] = "1"
            self.assertEqual(
                client.post("/api/auth/setup", json={"password": "test", "confirmation": "test"}).status_code,
                200,
            )
            secure.unlock("test")
            secure.write_json(chat_path, [sample])
            secure.write_json(profile_path, profile)
            secure.lock()

            response = client.get(url)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers["cache-control"], "no-store")
            data = response.json()
            self.assertEqual(len(data["messages"]), 3)
            self.assertEqual(data["messages"][0]["speaker"], "合成联系人")
            self.assertEqual(data["messages"][1]["speaker"], "本人")
            self.assertIsNone(data["messages"][1]["time"])
            self.assertEqual(data["messages"][1]["content"], "先看结论。")
            self.assertEqual(data["messages"][2]["content"], "再说理由。<script>text only</script>")
            self.assertEqual(data["messages"][2]["speaker"], "本人")
            self.assertEqual(data["messages"][2]["role"], "assistant")
            self.assertEqual(len({message["id"] for message in data["messages"]}), 3)
            self.assertNotIn(str(chat_path), response.text)
            self.assertEqual(client.get("/api/sources/ES:2/chat").status_code, 200)
            self.assertEqual(client.get("/api/sources/legacy/chat").status_code, 404)
            self.assertEqual(client.get("/api/sources/unknown/chat").status_code, 404)

            fact = client.get("/api/profile").json()["facts"][0]
            self.assertEqual(fact["source_confidence_mean"], 4)
            for status in ("approved", "rejected", "pending"):
                response = client.post(
                    "/api/reviews/batch",
                    json={"items": [{"id": fact["id"], "version": fact["version"]}], "status": status},
                )
                self.assertEqual(response.status_code, 200, response.text)
                exported = client.get("/api/avatar-profile").json()
                self.assertEqual(len(exported["facts"]), int(status == "approved"))
                fact = client.get("/api/profile").json()["facts"][0]
            edited = client.patch(
                f"/api/facts/{fact['id']}",
                json={
                    **{k: fact[k] for k in ("value", "attr", "group_id", "version")},
                    "value": "修改后的画像",
                },
            )
            self.assertEqual(edited.status_code, 200)
            self.assertEqual(edited.json()["source_confidence_mean"], 4)
            self.assertEqual(client.get(url).json(), data)

            # Stable IDs survive ordering changes; changed extraction content is rejected.
            secure.unlock("test")
            secure.write_json(chat_path, [{"id": "other", "messages": []}, sample])
            secure.lock()
            self.assertEqual(client.get(url).status_code, 200)
            secure.unlock("test")
            sample["state_memories"][0]["content"] = "另一条记忆"
            secure.write_json(chat_path, [sample])
            secure.lock()
            self.assertEqual(client.get(url).status_code, 404)
            secure.resolve_path(chat_path).unlink()
            self.assertEqual(client.get(url).status_code, 404)
            self.assertEqual(client.post("/api/auth/logout", json={}).status_code, 200)
            self.assertEqual(client.get(url).status_code, 401)


if __name__ == "__main__":
    unittest.main()
