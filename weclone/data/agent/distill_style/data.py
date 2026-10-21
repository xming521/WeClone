"""Raw-text records for style extraction."""

import hashlib
import json
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from weclone.utils import secure_storage

PRIMARY_CORPUS = "person"
DEFAULT_WINDOW_DAYS = 365
UNITS = {
    "message": "one target-speaker turn as stored in the input; chat input merges consecutive sends into one turn",
    "bubble": "nonblank newline-separated piece of a message; one sent message in chat input, which joins consecutive sends with a newline",
    "window": "one input record with all of its target messages; its size follows data windowing, not the speaker",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_rows(path):
    # Encrypted mode stores inputs as `<name>.enc`; secure storage resolves and decrypts them.
    text = secure_storage.read_text(path)
    if secure_storage.logical_path(path).suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError("Corpus JSON must be a list of records")
    return value


def write_json(path, value):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if secure_storage.is_encrypted_mode():
        secure_storage.write_text(path, text)
        return
    # Plaintext skips secure storage's per-file fsync, since the parse caches hold tens of thousands of files.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        file.write(text)
        temporary = Path(file.name)
    temporary.replace(path)


BUBBLE_BREAK = re.compile(r"\r\n|[\r\n]")


def bubble_spans(text):
    """(start, end) of each nonblank newline-separated piece, i.e. each sent message of a chat turn."""
    spans, start = [], 0
    for stop, resume in [(m.start(), m.end()) for m in BUBBLE_BREAK.finditer(text)] + [(len(text), None)]:
        if text[start:stop].strip():
            spans.append((start, stop))
        start = resume
    return spans


def visible_text(text):
    # These delimiters are the repository's completion protocol, not language heuristics.
    matches = re.findall(r"<\|message_start\|>(.*?)<\|message_end\|>", text, re.S)
    if matches:
        return "\n".join(matches)
    return re.sub(r"<\|(reply_thought|context)_start\|>.*?<\|\1_end\|>", "", text, flags=re.S)


def validate_records(records):
    seen = set()
    for row in records:
        for key in ("sample_id", "person_id", "conversation_id", "messages", "context", "boundary"):
            if key not in row:
                raise ValueError(f"Missing {key}")
        if row["sample_id"] in seen:
            raise ValueError("Duplicate sample_id")
        seen.add(row["sample_id"])
        if row["boundary"] not in ("response", "message"):
            raise ValueError("boundary must be response or message")
        ids = set()
        for msg in row["context"] + row["messages"]:
            if not isinstance(msg["text"], str) or msg["role"] not in ("user", "assistant"):
                raise ValueError("Expected visible user/assistant text")
            if msg["message_id"] in ids:
                raise ValueError("Duplicate message_id within record")
            ids.add(msg["message_id"])
        if not row["messages"] or any(m["role"] != "assistant" for m in row["messages"]):
            raise ValueError("messages must contain target speaker messages")
    return records


def load_training(path, person_id):
    rows = read_rows(path)
    if rows and "sample_id" in rows[0]:
        return validate_records(rows)
    records = []
    for row in rows:
        cid = str(row["id"])
        history = []
        for index, msg in enumerate(row["messages"]):
            if msg["role"] not in ("user", "assistant"):
                continue
            current = {"message_id": f"m{index}", "role": msg["role"], "text": visible_text(msg["content"])}
            if msg["role"] == "assistant":
                records.append(
                    {
                        "sample_id": f"{person_id}:{cid}:{index}",
                        "conversation_id": cid,
                        "person_id": person_id,
                        "time": row.get("time"),
                        "context": list(history),
                        "messages": [current],
                        "boundary": "response",
                    }
                )
            history.append(current)
    return validate_records(records)


def record_time(row):
    value = row.get("time")
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment.astimezone(timezone.utc).replace(tzinfo=None) if moment.tzinfo else moment


def apply_time_window(records, days=DEFAULT_WINDOW_DAYS):
    """Keep records within `days` before the latest record time; 0 keeps all history."""
    if not days:
        return records, {"days": None, "state": "disabled", "kept_records": len(records)}
    times = [record_time(row) for row in records]
    timed = [t for t in times if t is not None]
    if not timed:
        return records, {
            "days": days,
            "state": "not_applied",
            "reason": "No parseable record time",
            "kept_records": len(records),
        }
    end = max(timed)
    start = end - timedelta(days=days)
    kept = [row for row, t in zip(records, times, strict=True) if t is not None and t >= start]
    return kept, {
        "days": days,
        "state": "applied",
        "anchor": "latest record time",
        "start": start.isoformat(sep=" "),
        "end": end.isoformat(sep=" "),
        "input_records": len(records),
        "kept_records": len(kept),
        "untimed_records_dropped": len(times) - len(timed),
    }


def target_fingerprints(records):
    return {digest([m["text"] for m in r["messages"]]) for r in records}


def all_speaker_replies(records):
    """Make every B turn in a segment an extraction target."""
    result = []
    for row in records:
        dialogue = row.get("dialogue", row["context"] + row["messages"])
        result.append(
            dict(
                row,
                dialogue=dialogue,
                context=[m for m in dialogue if m["role"] == "user"],
                messages=[m for m in dialogue if m["role"] == "assistant"],
            )
        )
    return validate_records(result)


def load_profile_chats(path, person_id, limit=None):
    """Read the same source files and A/B turns as profile distillation."""
    from weclone.data.agent.distill_profile import iter_chat_files
    from weclone.data.agent.distill_windows import chat_messages

    path = Path(path)
    files = list(iter_chat_files(path)) if path.is_dir() else [path]
    records = []
    for source in files:
        for index, item in enumerate(read_rows(source)):
            dialogue = [
                {"message_id": f"m{i}", "role": role, "text": text}
                for i, (role, text) in enumerate(chat_messages(item))
            ]
            if not any(m["role"] == "assistant" for m in dialogue):
                continue
            cid = f"{source.name}:{item.get('id', index)}:{index}"
            records.append(
                {
                    "sample_id": f"{person_id}:{cid}",
                    "conversation_id": cid,
                    "person_id": person_id,
                    "time": item.get("time"),
                    "contact_id": item.get("chat_with_id"),
                    "context": [m for m in dialogue if m["role"] == "user"],
                    "messages": [m for m in dialogue if m["role"] == "assistant"],
                    "dialogue": dialogue,
                    "boundary": "response",
                    "source_file": str(source),
                }
            )
            if limit is not None and len(records) >= limit:
                return validate_records(records)
    return validate_records(records)
