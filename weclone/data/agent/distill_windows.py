from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from weclone.core.inference.llm_client import LLMRequest, LLMResponse
from weclone.core.inference.usage import UsageLimitReached
from weclone.data.agent.llm_cache import cache_key
from weclone.prompts.distill_memory import build_window_extract_prompt


@dataclass
class ChatSample:
    source_index: int
    sample_id: str
    item: dict[str, Any]
    include_current_state: bool


def chat_messages(item: dict[str, Any]):
    for message in item["messages"]:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            continue
        content = str(message.get("content", "")).replace("\r\n", "\n").strip()
        if content:
            yield message["role"], content


def content_chars(item: dict[str, Any]) -> int:
    return sum(len(content) for _, content in chat_messages(item))


def sample_body(item: dict[str, Any], *, target_role: str) -> str:
    return "\n".join(
        ("B:" if role == target_role else "A:") + content for role, content in chat_messages(item)
    )


def sample_header(sample_id: str) -> str:
    return f"#{sample_id}"


def render_sample(item: dict[str, Any], *, sample_id: str, target_role: str) -> str:
    body = sample_body(item, target_role=target_role)
    return f"{sample_header(sample_id)}\n{body}" if body else sample_header(sample_id)


def sample_cache_key(sample: ChatSample, *, task: str, target_role: str) -> str:
    """The rules this sample is extracted under and its text as the model sees it, without the positional id."""
    return cache_key(
        "distill",
        task,
        build_window_extract_prompt(task, include_current_state=sample.include_current_state),
        sample_body(sample.item, target_role=target_role),
    )


def group_samples(
    samples: list[ChatSample], *, max_samples: int, max_content_chars: int
) -> list[list[ChatSample]]:
    if max_samples < 1 or max_content_chars < 1:
        raise ValueError("Window sample and content limits must be positive")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("Duplicate input sample IDs")
    lengths = {sample.sample_id: content_chars(sample.item) for sample in samples}
    groups = []
    for recent in (False, True):
        current = []
        size = 0
        for sample in samples:
            length = lengths[sample.sample_id]
            if sample.include_current_state != recent or length > max_content_chars:
                continue
            if current and (len(current) == max_samples or size + length > max_content_chars):
                groups.append(current)
                current, size = [], 0
            current.append(sample)
            size += length
        if current:
            groups.append(current)
    groups.extend([sample] for sample in samples if lengths[sample.sample_id] > max_content_chars)
    return groups


def window_prompt(samples: list[ChatSample], *, task: str, target_role: str) -> str:
    prompt = build_window_extract_prompt(task, include_current_state=samples[0].include_current_state)
    body = "\n\n".join(render_sample(s.item, sample_id=s.sample_id, target_role=target_role) for s in samples)
    return prompt.replace("{{CHAT_SAMPLES}}", body)


def make_window_request(
    samples: list[ChatSample],
    *,
    task: str,
    source_path: Path,
    target_role: str,
    provider: str,
    model: str | None,
    effort: str | None,
    max_tokens: int | None,
) -> LLMRequest:
    return LLMRequest.from_prompt(
        window_prompt(samples, task=task, target_role=target_role),
        provider=provider,
        model=model,
        effort=effort,
        max_tokens=max_tokens,
        json_mode=True,
        metadata={
            "task": f"{task}_distill",
            "source_file": str(source_path),
            "sample_ids": [s.sample_id for s in samples],
            "samples": len(samples),
            "include_current_state": samples[0].include_current_state,
            "content_chars": sum(content_chars(s.item) for s in samples),
        },
    )


def validate_memory(memory: Any, *, content_field: str) -> None:
    if not isinstance(memory, dict):
        raise ValueError("Memory must be an object")
    if not isinstance(memory.get(content_field), str) or not memory[content_field].strip():
        raise ValueError(f"Missing or invalid {content_field}")
    tags = memory.get("tags")
    if (
        not isinstance(tags, list)
        or not tags
        or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
    ):
        raise ValueError("Missing or invalid tags")
    for name, values in (("importance", (1, 2, 3, 4)), ("confidence", (2, 3, 4))):
        if type(memory.get(name)) is not int or memory[name] not in values:
            raise ValueError(f"Invalid {name}")


def split_window_response(response: LLMResponse, samples: list[ChatSample], task: str) -> dict[str, Any]:
    if not response.ok or response.finish_reason in {"length", "content_filter"}:
        raise ValueError(response.error or f"Incomplete response: {response.finish_reason}")
    payload = response.parsed_json
    if (
        not isinstance(payload, dict)
        or set(payload) != {"results"}
        or not isinstance(payload["results"], list)
    ):
        raise ValueError("Expected a results array")
    expected = {s.sample_id: s for s in samples}
    # An answer names a sample by its ID or, as some models copy it, by its header line.
    names = {sample_header(sid): sid for sid in expected} | {sid: sid for sid in expected}
    parsed = {}
    field_name = f"{task}_memories"
    for item in payload["results"]:
        if not isinstance(item, dict) or set(item) != {"sample_id", field_name}:
            raise ValueError("Invalid per-sample result fields")
        sid = names.get(item["sample_id"]) if isinstance(item["sample_id"], str) else None
        if sid is None or sid in parsed:
            raise ValueError("Unknown or duplicate sample_id")
        value = item[field_name]
        if task == "state":
            if not isinstance(value, list):
                raise ValueError("state_memories must be an array")
            allowed = {"stable_fact", "preference", "goal"}
            if expected[sid].include_current_state:
                allowed.add("current_state")
            for memory in value:
                validate_memory(memory, content_field="content")
                if not isinstance(memory.get("type"), str) or memory["type"] not in allowed:
                    raise ValueError("Invalid profile type or current_state outside the time window")
            parsed[sid] = {"memories": value}
        else:
            if not isinstance(value, dict) or set(value) - {"surface_events"}:
                raise ValueError("Invalid event_memories object")
            events = value.get("surface_events", [])
            if not isinstance(events, list) or len(events) > 1:
                raise ValueError("Invalid surface_events array")
            for event in events:
                validate_memory(event, content_field="surface_event")
                types = event.get("event_types")
                if not isinstance(types, list) or not types or any(not isinstance(t, str) for t in types):
                    raise ValueError("Missing or invalid event_types")
            parsed[sid] = value
    if set(parsed) != set(expected):
        raise ValueError("Missing sample IDs")
    return parsed


@dataclass
class WindowResult:
    samples: list[ChatSample]
    attempts: list[LLMResponse] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def generate_window_batch(client: Any, batch: list[tuple[list[ChatSample], LLMRequest]], *, task: str):
    outcomes = [WindowResult(samples) for samples, _ in batch]
    pending = list(range(len(batch)))
    for _ in range(2):
        responses = list(client.generate_batch(batch[i][1] for i in pending))
        failed = []
        for position, i in enumerate(pending):
            response = (
                responses[position]
                if position < len(responses)
                else LLMResponse(ok=False, error="missing response")
            )
            outcome = outcomes[i]
            outcome.attempts.append(response)
            try:
                outcome.results = split_window_response(response, outcome.samples, task)
                outcome.error = ""
            except ValueError as error:
                outcome.error = str(error)
                outcome.attempts[-1] = replace(response, ok=False, error=outcome.error)
                failed.append(i)
        pending = failed
        # Past the usage limit a retry only raises, which would lose the answers this batch got.
        if not pending or isinstance(getattr(client, "limit", None), UsageLimitReached):
            break
    return outcomes
