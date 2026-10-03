"""Check profile discovery and review persistence with isolated synthetic data."""

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from weclone.server import profile_review as review
from weclone.utils import secure_storage as secure

PASSWORD = "profile-path-check"


class ProfilePathChecks(unittest.TestCase):
    @contextmanager
    def project(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "settings.jsonc"
            config.write_text(
                json.dumps({"security_args": {"storage_mode": mode, "state_dir": str(root / "state")}})
            )
            with (
                patch.dict(os.environ, {"WECLONE_CONFIG_PATH": str(config)}),
                patch.object(review, "ROOT", root),
            ):
                secure.configure()
                secure.ensure_initialized(PASSWORD)
                secure.lock()
                try:
                    yield root / "dataset/res_csv/agent"
                finally:
                    secure.lock()

    def write_profile(self, source, value):
        data = {
            "dimensions": [
                {
                    "dim": 8,
                    "name": "偏好",
                    "groups": [{"name": "阅读", "items": [{"attr": "偏好", "fact_indices": [0]}]}],
                }
            ],
            "facts": [{"dim": 8, "attr": "偏好", "value": value, "source_ids": []}],
            "sources": {},
        }
        secure.unlock(PASSWORD)
        try:
            return secure.write_json(source, data)
        finally:
            secure.lock()

    def login(self, client):
        client.headers["X-WeClone-Request"] = "1"
        self.assertEqual(client.get("/api/profile").status_code, 401)
        response = client.post("/api/auth/login", json={"password": PASSWORD})
        self.assertEqual(response.status_code, 200, response.text)

    def profile(self, client, expected):
        response = client.get("/api/profile")
        self.assertEqual(response.status_code, 200, response.text)
        facts = response.json()["facts"]
        self.assertEqual([fact["value"] for fact in facts], [expected])
        return facts[0]

    def test_latest_complete_run_and_snapshot_without_source(self):
        for mode in ("plaintext", "encrypted"):
            with self.subTest(mode=mode), self.project(mode) as directory:
                runs = directory / "profile_runs"
                self.write_profile(
                    runs / "20260101_000000" / "profile_hierarchy/profile_hierarchy.json", "旧画像"
                )
                source = runs / "20260102_000000" / "profile_hierarchy/profile_hierarchy.json"
                physical_source = self.write_profile(source, "新画像")
                (runs / "20260103_000000" / "profile_hierarchy").mkdir(parents=True)
                database = source.parent / "profile_review.sqlite3"
                self.assertEqual(review.review_paths(), (database, source))
                with TestClient(review.create_app()) as client:
                    self.login(client)
                    fact = self.profile(client, "新画像")
                    response = client.post(
                        "/api/reviews/batch",
                        json={
                            "items": [{"id": fact["id"], "version": fact["version"]}],
                            "status": "approved",
                        },
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(secure.file_exists(database))
                physical_source.unlink()
                self.assertEqual(review.review_paths(), (database, source))
                with TestClient(review.create_app()) as client:
                    self.login(client)
                    persisted = self.profile(client, "新画像")
                    self.assertEqual(persisted["id"], fact["id"])
                    self.assertEqual(persisted["status"], "approved")

    def test_legacy_profile_and_snapshot_take_precedence(self):
        for mode in ("plaintext", "encrypted"):
            with self.subTest(mode=mode), self.project(mode) as directory:
                source = directory / "memory_organization/profile_hierarchy.json"
                physical_source = self.write_profile(source, "已有画像")
                self.write_profile(
                    directory / "profile_runs/20260102_000000/profile_hierarchy/profile_hierarchy.json",
                    "新运行",
                )
                database = source.parent / "profile_review.sqlite3"
                self.assertEqual(review.review_paths(), (database, source))
                with TestClient(review.create_app()) as client:
                    self.login(client)
                    self.profile(client, "已有画像")
                physical_source.unlink()
                self.assertEqual(review.review_paths(), (database, source))
                with TestClient(review.create_app()) as client:
                    self.login(client)
                    self.profile(client, "已有画像")

    def test_explicit_sources_have_separate_databases(self):
        for mode in ("plaintext", "encrypted"):
            with self.subTest(mode=mode), self.project(mode) as directory:
                for name in ("first", "second"):
                    source = directory / name / "profile.json"
                    physical_source = self.write_profile(source, name)
                    database = source.parent / "profile_review.sqlite3"
                    self.assertEqual(review.review_paths(source=physical_source), (database, physical_source))
                    with TestClient(review.create_app(source=physical_source)) as client:
                        self.login(client)
                        self.profile(client, name)
                    self.assertTrue(secure.file_exists(database))
                    override = directory / "custom.sqlite3"
                    self.assertEqual(
                        review.review_paths(override, physical_source), (override, physical_source)
                    )

    def test_missing_profile_does_not_create_an_empty_review_database(self):
        for mode in ("plaintext", "encrypted"):
            with self.subTest(mode=mode), self.project(mode) as directory:
                database, _ = review.review_paths()
                with TestClient(review.create_app()) as client:
                    self.login(client)
                    response = client.get("/api/profile")
                    self.assertEqual(response.status_code, 404)
                self.assertFalse(secure.file_exists(database))
                source = directory / "profile_runs/20260102_000000/profile_hierarchy/profile_hierarchy.json"
                self.write_profile(source, "随后生成")
                self.assertEqual(review.review_paths()[1], source)
                with TestClient(review.create_app()) as client:
                    self.login(client)
                    self.profile(client, "随后生成")


if __name__ == "__main__":
    unittest.main()
