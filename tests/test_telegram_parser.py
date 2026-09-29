import json
from types import SimpleNamespace

from weclone.data.chat_parsers.telegram_parser import process_telegram_dataset


def make_config():
    return SimpleNamespace(
        telegram_args=SimpleNamespace(my_id="user-1"),
        include_type=[],
    )


def test_process_full_telegram_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    export_dir = tmp_path / "dataset" / "telegram" / "ChatExport"
    export_dir.mkdir(parents=True)
    (export_dir / "result.json").write_text(
        json.dumps(
            {
                "chats": {
                    "list": [
                        {
                            "name": "Alice",
                            "type": "personal_chat",
                            "id": 101,
                            "messages": [
                                {
                                    "id": 1,
                                    "type": "message",
                                    "date": "2025-01-01T12:00:00",
                                    "from": "me",
                                    "from_id": "user-1",
                                    "text": "hello",
                                }
                            ],
                        },
                        {
                            "name": "Team",
                            "type": "group",
                            "id": 202,
                            "messages": [
                                {
                                    "id": 2,
                                    "type": "message",
                                    "date": "2025-01-01T12:01:00",
                                    "from": "Alice",
                                    "from_id": "user-2",
                                    "text": [{"type": "plain", "text": "hi"}],
                                }
                            ],
                        },
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    process_telegram_dataset(make_config())

    alice_csv = tmp_path / "dataset" / "csv" / "Alice-personal_chat-101" / "Alice-personal_chat-101.csv"
    team_csv = tmp_path / "dataset" / "csv" / "Team-group-202" / "Team-group-202.csv"
    assert alice_csv.exists()
    assert team_csv.exists()
    assert "hello" in alice_csv.read_text(encoding="utf-8")
    assert "hi" in team_csv.read_text(encoding="utf-8")


def test_process_individual_chat_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    export_dir = tmp_path / "dataset" / "telegram" / "SingleChat"
    export_dir.mkdir(parents=True)
    (export_dir / "result.json").write_text(
        json.dumps(
            {
                "name": "Bob",
                "type": "personal_chat",
                "id": 303,
                "messages": [
                    {
                        "id": 1,
                        "type": "message",
                        "date": "2025-01-01T12:00:00",
                        "from": "Bob",
                        "from_id": "user-3",
                        "text": "hello from Bob",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    process_telegram_dataset(make_config())

    output = tmp_path / "dataset" / "csv" / "Bob-personal_chat-303" / "Bob-personal_chat-303.csv"
    assert output.exists()
    assert "hello from Bob" in output.read_text(encoding="utf-8")
