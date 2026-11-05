"""Chat exports under csv_folder: one folder per chat, holding CSV chunks and an optional users.json.

make-dataset and build-profile's chat snapshot both read chats through here, so they see the same files and the same
chat partner.
"""

import json
import os
import re
from io import BytesIO
from pathlib import Path

import pandas as pd

from weclone.utils import secure_storage
from weclone.utils.log import logger

CHUNK_START = re.compile(r"_(\d+)_\d+\.csv$")


def list_csv_files(csv_folder: str | Path, parsed_files: list[str] | None = None) -> list[str]:
    """CSV chunks of every chat folder (or the given parsed files), sorted by the starting sequence number in the name."""
    csv_files = list(parsed_files) if parsed_files is not None else []
    if parsed_files is None:
        for chat_folder in os.listdir(csv_folder):
            chat_folder_path = os.path.join(csv_folder, chat_folder)
            if not os.path.isdir(chat_folder_path):
                continue
            csv_files.extend(str(path) for path in secure_storage.iter_files(chat_folder_path, "*.csv"))

    def start(path: str) -> int:
        match = CHUNK_START.search(secure_storage.logical_path(path).name)
        return int(match.group(1)) if match else 0

    return sorted(csv_files, key=start)


def read_csv(file_path: str | Path) -> pd.DataFrame:
    imported_path = secure_storage.import_file(file_path)
    return pd.read_csv(
        BytesIO(secure_storage.read_bytes(imported_path)),
        encoding="utf-8",
        dtype={"msg": str, "src": str},
        escapechar=None,
        keep_default_na=False,
    )


def load_users_data(folder_path: str | Path) -> dict:
    users_json_path = os.path.join(folder_path, "users.json")
    if not secure_storage.file_exists(users_json_path):
        return {}
    try:
        imported_path = secure_storage.import_file(users_json_path)
        data = secure_storage.read_json(imported_path)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load users.json from {folder_path}: {e}")
        return {}


def find_partner_talker(folder_name: str, df: pd.DataFrame) -> str:
    """The non-self talker of a chat folder: the folder name if it received messages, else the only talker it
    received from, else the folder name."""
    received_talkers = []
    if "is_sender" in df.columns and "talker" in df.columns:
        is_received = df["is_sender"].astype(str) == "0"
        received_talkers = [
            str(talker).strip()
            for talker in df.loc[is_received, "talker"].dropna().astype(str).unique()
            if str(talker).strip()
        ]
    if folder_name not in received_talkers and len(received_talkers) == 1:
        return received_talkers[0]
    return folder_name
