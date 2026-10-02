"""Resolve a persisted memory origin to the chat sample used for extraction."""

from pathlib import Path

from fastapi import HTTPException

from weclone.utils import secure_storage as secure

ROOT = Path(__file__).resolve().parents[2]


def sample_id(sample: dict, index: int) -> str:
    for key in ("id", "local_id"):
        value = sample.get(key)
        if value is not None and str(value):
            return str(value)
    return str(index)


def matches_memory(sample: dict, origin: dict, source: dict) -> bool:
    kind = origin.get("kind", source.get("kind"))
    if kind == "S":
        memories, field = sample.get("state_memories"), "content"
    elif kind in {"ES", "EI"}:
        events = sample.get("event_memories")
        if not isinstance(events, dict):
            return False
        memories = events.get("surface_events" if kind == "ES" else "inferred_events")
        field = "surface_event" if kind == "ES" else "inferred_event"
    else:
        return False
    index = origin.get("memory_index")
    return (
        isinstance(memories, list)
        and type(index) is int
        and 0 <= index < len(memories)
        and isinstance(memories[index], dict)
        and isinstance(source.get("content"), str)
        and memories[index].get(field) == source["content"]
    )


def source_chat(source_id: str, source: dict) -> dict:
    # Paths come only from the authenticated snapshot, never from URL/query parameters.
    origins = source.get("origins")
    if not isinstance(origins, list) or not origins:
        raise HTTPException(404, "该来源未保存聊天定位信息，无法查看原文")
    for origin in origins:
        if not isinstance(origin, dict) or not isinstance(origin.get("file"), str):
            continue
        path = secure.logical_path(origin["file"])
        if path.suffix != ".json" or "archive" in path.parts:
            continue
        if not path.is_absolute():
            path = ROOT / path
        try:
            samples = secure.read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(samples, list):
            continue
        candidates = [
            sample
            for index, sample in enumerate(samples)
            if isinstance(sample, dict) and sample_id(sample, index) == str(origin.get("sample_id"))
        ]
        # Never use an index blindly: files may have been reordered or replaced.
        if len(candidates) != 1 or not matches_memory(candidates[0], origin, source):
            continue
        sample = candidates[0]
        raw_messages = sample.get("messages")
        if not isinstance(raw_messages, list):
            continue
        messages = []
        peer = str(sample.get("chat_with") or "对方")
        for index, message in enumerate(raw_messages):
            if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            messages.append(
                {
                    "id": str(index),
                    "role": message["role"],
                    "speaker": "本人" if message["role"] == "assistant" else peer,
                    "content": content,
                    "time": message.get("time") if isinstance(message.get("time"), str) else None,
                }
            )
        if messages:
            return {
                "source_id": source_id,
                "sample_id": str(origin["sample_id"]),
                "chat_with": peer,
                "sample_time": sample.get("time") if isinstance(sample.get("time"), str) else None,
                "messages": messages,
            }
    raise HTTPException(404, "原始聊天文件不存在或与来源记录不一致，请检查本次抽取的来源文件")
