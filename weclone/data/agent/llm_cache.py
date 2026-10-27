"""Model answers keyed by what was asked, shared by every profile run.

A key hashes the question itself: the prompt and output schema of an organize task, or the extraction rules and one
chat sample's text for distillation. A rerun asks only what is missing, so a failed run resumes by running again. A
new chat export pays for the samples whose text changed, and for profile samples that moved out of their chat's last
14 days, which are extracted without current states from then on. Only answers that passed validation are stored, so
failed requests are asked again. The key leaves out the model and request settings, so
switching models midway keeps the answers given so far. Delete the cache file to ask everything again.
"""

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from weclone.utils import secure_storage

DEFAULT_PATH = Path("dataset/res_csv/agent/llm_cache.sqlite")
_QUERY_CHUNK = 500


def cache_key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(list(parts), ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _create_table(db) -> None:
    db.execute(
        "CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, task TEXT NOT NULL, answer TEXT NOT NULL, "
        "model TEXT NOT NULL, created_at TEXT NOT NULL)"
    )


class LLMCache:
    def __init__(self, path: Path | str | None = None):
        # DEFAULT_PATH is read here rather than bound as a default, so tests can redirect it.
        self.path = Path(path) if path is not None else DEFAULT_PATH

    def get_many(self, keys: Iterable[str]) -> dict[str, Any]:
        keys = list(dict.fromkeys(keys))
        if not keys or not secure_storage.file_exists(self.path):
            return {}
        found = {}
        with secure_storage.encrypted_sqlite(self.path) as db:
            _create_table(db)
            for start in range(0, len(keys), _QUERY_CHUNK):
                chunk = keys[start : start + _QUERY_CHUNK]
                rows = db.execute(
                    f"SELECT key, answer FROM answers WHERE key IN ({','.join('?' * len(chunk))})", chunk
                )
                found.update((key, json.loads(answer)) for key, answer in rows)
        return found

    def put_many(self, rows: Iterable[tuple[str, str, Any, str]]) -> None:
        """Store (key, task, answer, model) rows. A key answered before keeps its first answer."""
        at = datetime.now().isoformat(timespec="seconds")
        values = [
            (key, task, json.dumps(answer, ensure_ascii=False), model or "", at)
            for key, task, answer, model in rows
        ]
        if not values:
            return
        with secure_storage.encrypted_sqlite(self.path) as db:
            _create_table(db)
            db.executemany(
                "INSERT OR IGNORE INTO answers (key, task, answer, model, created_at) VALUES (?, ?, ?, ?, ?)",
                values,
            )
