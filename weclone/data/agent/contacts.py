"""Stable contact ids.

A contact is keyed by a hash of its source account (platform and partner talker), never by remark name or by the
order contacts appear in. The registry gives each key a P id once and only grows, so a contact keeps its id across
chat exports and rebuilds, and prompts, facts and style groups that name P3 keep meaning the same person.
"""

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

from filelock import FileLock

from weclone.utils import secure_storage

DEFAULT_REGISTRY_PATH = Path("dataset/res_csv/agent/contacts.json")
CONTACT_ID = re.compile(r"P[1-9]\d*")


def contact_key(platform: str, talker: str) -> str:
    if not talker:
        raise ValueError("A contact key needs the partner talker")
    return hashlib.sha256(f"{platform}:{talker}".encode()).hexdigest()[:16]


def _load_registry(path: Path) -> dict[str, str]:
    if not secure_storage.file_exists(path):
        return {}
    registry = secure_storage.read_json(path)
    if not isinstance(registry, dict) or not all(
        isinstance(key, str) and isinstance(cid, str) and CONTACT_ID.fullmatch(cid) for key, cid in registry.items()
    ):
        raise ValueError(f"Contact registry must map keys to P ids: {path}")
    if len(set(registry.values())) != len(registry):
        raise ValueError(f"Contact registry reuses a P id: {path}")
    return registry


def assign_contact_ids(keys: Iterable[str], registry_path: Path | str = DEFAULT_REGISTRY_PATH) -> dict[str, str]:
    """P ids for the keys. Known keys keep theirs, new keys take the next numbers in the given order."""
    keys = list(dict.fromkeys(keys))
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(f"{path}.lock"):
        registry = _load_registry(path)
        number = max((int(cid[1:]) for cid in registry.values()), default=0)
        new_keys = [key for key in keys if key not in registry]
        for key in new_keys:
            number += 1
            registry[key] = f"P{number}"
        if new_keys:
            secure_storage.write_json(path, registry)
    return {key: registry[key] for key in keys}
