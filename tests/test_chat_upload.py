import io
import subprocess
import sys
import zipfile

import pytest
from fastapi.testclient import TestClient

from weclone.data import chat_csv
from weclone.data.agent import profile_pipeline as pipeline
from weclone.server import chat_upload, profile_build
from weclone.server.profile_review import create_app
from weclone.utils import secure_storage
from tests.test_profile_build import PASSWORD, configure

ENDPOINT = {"base_url": "u", "llm_api_key": "k", "model_name": "m"}
CSV = "id,MsgSvrID,type_name,is_sender,talker,msg,src,CreateTime\n1,1,文本,0,alice,你好,,2026-01-01 10:00:00\n"


@pytest.fixture
def web(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_upload, "STAGING_DIR", tmp_path / "uploads")

    def start(mode="plaintext", platform="chat"):
        configure(
            tmp_path,
            monkeypatch,
            mode,
            {
                "agent_distill_args": {"llm_provider": "api"},
                "make_dataset_args": {"platform": platform, "csv_folder": str(tmp_path / "csv"), **ENDPOINT},
            },
        )
        client = TestClient(create_app(static_dir=tmp_path / "no-static"))
        client.headers["X-WeClone-Request"] = "1"
        if not secure_storage.is_initialized():
            assert client.post("/api/auth/setup", json=PASSWORD).status_code == 200
        else:
            assert client.post("/api/auth/login", json={"password": PASSWORD["password"]}).status_code == 200
        return client

    return start


def upload(client, files: dict[str, bytes]) -> tuple[str, list[int]]:
    upload_id = client.post("/api/data/uploads").json()["id"]
    kept = []
    for path, content in files.items():
        response = client.put(f"/api/data/uploads/{upload_id}/files", params={"path": path}, content=content)
        assert response.status_code == 200, response.text
        kept.append(response.json()["kept"])
    return upload_id, kept


def zipped(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        for name, content in members.items():
            zip_file.writestr(name, content)
    return buffer.getvalue()


def test_an_upload_replaces_the_chats_uploaded_before(web, tmp_path):
    client = web()
    upload_id, kept = upload(
        client,
        {
            "csv/alice/alice_0_1.csv": CSV.encode(),
            "csv/alice/users.json": b"{}",
            "csv/alice/photo.jpg": b"jpg",
            "csv/.DS_Store": b"",
            "csv/qq/qq_0_1.csv": CSV.encode(),
        },
    )
    assert kept == [1, 1, 0, 0, 1]
    assert client.post(f"/api/data/uploads/{upload_id}/done").json() == {"chats": 2}
    uploaded = tmp_path / "uploaded"
    assert sorted(path.name for path in (uploaded / "alice").iterdir()) == ["alice_0_1.csv", "users.json"]
    assert not (tmp_path / "uploads" / upload_id).exists()

    upload_id, _ = upload(client, {"alice/alice_2_3.csv": CSV.encode()})
    assert client.post(f"/api/data/uploads/{upload_id}/done").json() == {"chats": 1}
    assert [path.name for path in uploaded.iterdir()] == ["alice"]
    assert [path.name for path in (uploaded / "alice").iterdir()] == ["alice_2_3.csv"]
    assert not (tmp_path / "csv").exists()


def test_an_upload_refused_during_a_build_finishes_after_it(web, tmp_path, monkeypatch):
    client = web()
    upload_id, _ = upload(client, {"alice/alice_0_1.csv": CSV.encode()})
    monkeypatch.setattr(profile_build.Builds, "running", lambda self: True)
    assert client.post(f"/api/data/uploads/{upload_id}/done").json()["code"] == "build_running"
    monkeypatch.setattr(profile_build.Builds, "running", lambda self: False)
    assert client.post(f"/api/data/uploads/{upload_id}/done").json() == {"chats": 1}
    assert [path.name for path in (tmp_path / "uploaded" / "alice").iterdir()] == ["alice_0_1.csv"]


def test_a_zip_names_loose_chunks_after_itself_and_rejects_paths_out(web, tmp_path):
    client = web()
    upload_id, kept = upload(client, {"alice.zip": zipped({"alice_0_1.csv": CSV.encode(), "a.png": b""})})
    assert kept == [1]
    assert client.post(f"/api/data/uploads/{upload_id}/done").json()["chats"] == 1
    assert (tmp_path / "uploaded" / "alice" / "alice_0_1.csv").is_file()

    upload_id = client.post("/api/data/uploads").json()["id"]
    response = client.put(
        f"/api/data/uploads/{upload_id}/files",
        params={"path": "chats.zip"},
        content=zipped({"../../escape/x.csv": b""}),
    )
    assert response.json()["code"] == "upload_path_invalid"
    response = client.put(f"/api/data/uploads/{upload_id}/files", params={"path": "../x.csv"}, content=b"")
    assert response.json()["code"] == "upload_path_invalid"
    assert not (tmp_path / "escape").exists()


def test_chunks_outside_a_chat_folder_are_not_a_chat(web):
    client = web()
    upload_id, _ = upload(client, {"alice_0_1.csv": CSV.encode()})
    assert client.post(f"/api/data/uploads/{upload_id}/done").json()["code"] == "upload_no_chats"
    upload_id, _ = upload(client, {"a/alice/x.csv": CSV.encode(), "b/alice/y.csv": CSV.encode()})
    assert client.post(f"/api/data/uploads/{upload_id}/done").json()["code"] == "upload_duplicate_chat"


def test_an_encrypted_upload_stores_ciphertext_the_build_reads(web, tmp_path):
    client = web("encrypted")
    upload_id, _ = upload(
        client, {"alice/alice_0_1.csv": CSV.encode(), "qq.zip": zipped({"qq/qq_0_1.csv": CSV.encode()})}
    )
    assert client.post(f"/api/data/uploads/{upload_id}/done").json()["chats"] == 2
    stored = tmp_path / "uploaded" / "qq" / "qq_0_1.csv.enc"
    assert stored.read_bytes().startswith(secure_storage.MAGIC)
    assert not (tmp_path / "uploaded" / "alice" / "alice_0_1.csv").exists()
    with secure_storage.use_key(secure_storage.unlock_key(PASSWORD["password"])):
        files = chat_csv.list_csv_files(tmp_path / "uploaded")
        assert len(files) == 2 and chat_csv.read_csv(files[0])["msg"].tolist() == ["你好"]


def test_a_build_reads_the_uploaded_chats_unless_asked_for_the_configured_ones(web, tmp_path, monkeypatch):
    client = web(platform="telegram")
    query = {"max_context_tokens": 64000}
    assert client.get("/api/build/check", params=query).json()["chats"]["source"] == "config"
    upload_id, _ = upload(client, {"alice/alice_0_1.csv": CSV.encode()})
    client.post(f"/api/data/uploads/{upload_id}/done")

    found = client.get("/api/build/check", params=query).json()
    assert found["problems"] == []
    assert found["chats"] == {
        "source": "upload",
        "upload": {"count": 1},
        "config": {"platform": "telegram", "folder": "dataset/telegram", "count": 0},
    }
    found = client.get("/api/build/check", params={**query, "chats": "config"}).json()
    assert found["chats"]["source"] == "config" and "telegram_id_missing" in [p["code"] for p in found["problems"]]

    spawned = []

    def spawn(self, run, args):
        spawned.append(args)
        return subprocess.Popen([sys.executable, "-c", ""])

    monkeypatch.setattr(profile_build.Builds, "spawn", spawn)
    body = {"purpose": "assistant", "consent": True, **query}
    assert client.post("/api/build", json=body).status_code == 202
    assert spawned[0][-2:] == ["--csv-folder", str(pipeline.UPLOAD_DIR)]


@pytest.mark.parametrize("size", [0, secure_storage.CHUNK_SIZE, 2 * secure_storage.CHUNK_SIZE + 7])
def test_a_file_written_in_pieces_reads_back_whole(web, tmp_path, size):
    web("encrypted")
    data = bytes(range(256)) * (size // 256) + b"x" * (size % 256)
    with secure_storage.use_key(secure_storage.unlock_key(PASSWORD["password"])):
        with secure_storage.open_write(tmp_path / "piece.bin") as output:
            for start in range(0, size, 300_001):
                output.write(data[start : start + 300_001])
        assert secure_storage.read_bytes(tmp_path / "piece.bin") == data
