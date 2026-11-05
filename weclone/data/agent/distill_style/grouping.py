"""Contact or relationship groups for conditioned style statistics."""

import json
import re
from collections import Counter
from pathlib import Path

from .data import PRIMARY_CORPUS

DEFAULT_MIN_GROUP_MESSAGES = 300
GROUP_PREFIX = "group-"


def nonempty_messages(row):
    return sum(bool(m["text"].strip()) for m in row["messages"])


def load_group_map(path):
    if path is None:
        return None
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in value.items()
    ):
        raise ValueError("--contact-groups must map contact_id strings to nonempty group labels")
    return value


def group_key(row, mapping):
    contact = row.get("contact_id")
    if contact is None:
        return None
    return mapping.get(contact, contact) if mapping else contact


def assign_groups(records, mapping=None, min_messages=DEFAULT_MIN_GROUP_MESSAGES, enabled=True):
    """Give each sufficiently large contact or mapped relationship group its own corpus."""
    messages, windows, contacts = Counter(), Counter(), {}
    for row in records:
        key = group_key(row, mapping)
        if key is None:
            continue
        messages[key] += nonempty_messages(row)
        windows[key] += 1
        contacts.setdefault(key, set()).add(row["contact_id"])
    names = {}
    for key in sorted(k for k, n in messages.items() if enabled and n >= min_messages):
        name = GROUP_PREFIX + re.sub(r"[^\w.-]", "_", key)
        if name in names.values() or name == PRIMARY_CORPUS:
            raise ValueError(f"Group label collides after sanitizing: {key}")
        names[key] = name
    ungrouped = [k for k in messages if k not in names]
    return {
        "enabled": enabled,
        "basis": "contact_groups mapping, unmapped contacts kept separate" if mapping else "contact_id",
        "min_messages": min_messages,
        "corpora": {
            names[k]: {
                "group": k,
                "contacts": sorted(contacts[k]),
                "windows": windows[k],
                "messages": messages[k],
            }
            for k in sorted(names)
        },
        "ungrouped": {
            "groups": len(ungrouped),
            "windows": sum(windows[k] for k in ungrouped),
            "messages": sum(messages[k] for k in ungrouped),
        },
        "windows_without_contact": sum(group_key(row, mapping) is None for row in records),
        "_names": names,
        "_mapping": mapping,
    }


def public(grouping):
    return {k: v for k, v in grouping.items() if not k.startswith("_")}


def corpus_of(row, grouping):
    return grouping["_names"].get(group_key(row, grouping["_mapping"]))


def build_corpora(records, grouping, selection=None):
    """Person corpus plus one corpus per group; a semantic selection keeps the pooled subsample unweighted."""
    if selection is None:
        corpora = {PRIMARY_CORPUS: list(records)}
        allowed = None
    else:
        pooled = set(selection["pooled_ids"])
        corpora = {PRIMARY_CORPUS: [r for r in records if r["sample_id"] in pooled]}
        allowed = {name: set(ids) for name, ids in selection["group_ids"].items()}
    for name in grouping["corpora"]:
        rows = [
            r
            for r in records
            if corpus_of(r, grouping) == name and (allowed is None or r["sample_id"] in allowed.get(name, ()))
        ]
        if rows:
            corpora[name] = rows
    return corpora


def corpus_scope(corpora):
    return {
        name: {"windows": len(rows), "messages": sum(nonempty_messages(r) for r in rows)}
        for name, rows in corpora.items()
    }
