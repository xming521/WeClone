"""Coalesce independent annotations while retaining per-input cache identities."""

from weclone.prompts.distill_style import (
    format_joint_prompt as format_joint_prompt,
    BATCH_PROMPT as BATCH_PROMPT,
    joint_prompt as joint_prompt,
)

import json
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .data import write_json
from .wire import COMMON_PROMPT, MODULES

DEFAULT_LLM_BATCH_SIZE = 8
DEFAULT_LLM_BATCH_INPUT_CHARS = 16000
DEFAULT_LLM_BATCH_OUTPUT_CHARS = 32000
DEFAULT_BATCH_WAIT_SECONDS = 0.5
BATCH_PROTOCOL = "style-chat-batch-v8"
# Keyed by module so protocol bumps carry over; JOINT_FIELDS maps the current protocol names.
JOINT_LABELS = {"register_style": "语体", "lexical_choice": "词汇", "rhetorical_devices": "修辞"}
JOINT_FIELDS = {protocol: JOINT_LABELS[name] for protocol, name in MODULES.items() if name in JOINT_LABELS}
JOINT_MODULES = set(JOINT_FIELDS)


def shared_input(payload):
    return {k: payload[k] for k in ("context", "messages", "dialogue") if k in payload}


def chat_routes(rows):
    identities, routes = {}, []
    for row in rows:
        identity = (row["conversation_id"], compact(shared_input(row["payload"])))
        sid = identities.setdefault(identity, f"s{len(identities)}")
        routes.append((sid, JOINT_FIELDS[row["protocol"]]))
    return routes


def batch_request(rows):
    rules = list(dict.fromkeys((r["protocol"], r["prompt"]) for r in rows))
    if len(rules) == 1 and not all(
        r["protocol"] in JOINT_MODULES and "messages" in r["payload"] and "failures" not in r["payload"]
        for r in rows
    ):
        return BATCH_PROMPT + rules[0][1], {
            "samples": [
                {"sample_id": f"s{i}", "input": {k: v for k, v in r["payload"].items() if k != "boundary"}}
                for i, r in enumerate(rows)
            ]
        }
    chats = {}
    for row, (sid, _) in zip(rows, chat_routes(rows), strict=True):
        chat = chats.setdefault(sid, {"sample_id": sid, **shared_input(row["payload"])})
        chat.update({k: v for k, v in row["payload"].items() if k not in {"context", "messages", "boundary"}})
    prompt = (
        joint_prompt(dict.fromkeys(JOINT_FIELDS[protocol] for protocol, _ in rules))
        + COMMON_PROMPT
        + "\n".join(
            f"{JOINT_FIELDS[protocol]}：\n{prompt.removeprefix(COMMON_PROMPT)}" for protocol, prompt in rules
        )
    )
    return prompt, {"chats": list(chats.values())}


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def dialogue_of(chat):
    return chat.get("dialogue", chat.get("context", []) + chat.get("messages", []))


def candidate_evidence(candidate, dialogue):
    mid, quote = candidate["message_id"], candidate["quote"]
    occurrence = 0
    for identity, _, text in dialogue:
        if identity == mid:
            occurrence += sum(text.startswith(quote, i) for i in range(candidate["start"]))
            return [quote, occurrence]
        occurrence += sum(text.startswith(quote, i) for i in range(len(text)))
    raise ValueError("Candidate message missing from dialogue")


def joint_transport(prompt, payload):
    """Render every extractor's chats without exposing internal routing IDs."""
    chats = payload.get("chats")
    if chats is None:
        chats = [s["input"] for s in payload["samples"]]
    delimiter = "---"
    while any(delimiter in m[2] for c in chats for m in dialogue_of(c)):
        delimiter += "-"
    blocks = []
    for chat in chats:
        dialogue = dialogue_of(chat)
        lines = [("B:" if role == "assistant" else "A:") + text for _, role, text in dialogue]
        if "degree_anchors" in chat:
            groups = [
                [
                    {k: a[k] for k in ("modifier", "target", "source")}
                    for a in chat["degree_anchors"]
                    if a["message_id"] == mid
                ]
                for mid, _, _ in chat["messages"]
            ]
            lines.append("degree_anchors=" + compact(groups))
        if "anchors" in chat:
            candidates = {
                key: [candidate_evidence(a, dialogue) for a in chat["anchors"][key]]
                for key in ("repeats", "emoji")
            }
            lines.append("anchors=" + compact(candidates))
        blocks.append("\n".join(lines))
    return format_joint_prompt(prompt, delimiter), ("\n" + delimiter + "\n").join(blocks)


def restore_chat_output(value, chat, protocol=None):
    value = restore_joint_evidence(value, dialogue_of(chat))
    if not isinstance(value, dict):
        raise ValueError("Expected a dialogue result object")
    if MODULES.get(protocol) in {"affective_style", "pragmatic_style"} and len(chat.get("messages", [])) > 1:
        details = value.get("items", {})
        if not isinstance(details, dict):
            raise ValueError("Invalid item details")
        for key, parts in details.items():
            if not isinstance(parts, list) or len(parts) != len(chat["messages"]):
                raise ValueError("Incomplete message detail coverage")
            if any(part is not None and not isinstance(part, dict) for part in parts):
                raise ValueError("Invalid message detail")
            details[key] = {
                m[0]: part for m, part in zip(chat["messages"], parts, strict=True) if part is not None
            }
    if MODULES.get(protocol) == "paralinguistic_style":
        for field, key in (("repeat_usage", "repeats"), ("emoji_usage", "emoji")):
            if field not in value:
                continue
            parts, candidates = value[field], chat["anchors"][key]
            if not isinstance(parts, list) or len(parts) != len(candidates):
                raise ValueError("Incomplete candidate coverage")
            for part, candidate in zip(parts, candidates, strict=True):
                if not isinstance(part, dict) or "anchor_id" in part:
                    raise ValueError("Invalid candidate detail")
                part["anchor_id"] = candidate["anchor_id"]
    return value


def restore_joint_evidence(value, messages):
    if isinstance(value, list):
        if len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int:
            quote, occurrence = value
            if not quote or occurrence < 0:
                raise ValueError("Invalid evidence occurrence")
            for mid, _, text in messages:
                positions = [i for i in range(len(text)) if text.startswith(quote, i)]
                if occurrence < len(positions):
                    return [mid, quote, occurrence]
                occurrence -= len(positions)
            raise ValueError("Evidence not in dialogue")
        return [restore_joint_evidence(v, messages) for v in value]
    if isinstance(value, dict):
        return {k: restore_joint_evidence(v, messages) for k, v in value.items()}
    return value


class BatchDispatcher:
    def __init__(self, generate, output, workers, max_samples, input_chars, output_chars):
        self.generate = generate
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "batches").mkdir(exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.lock = threading.RLock()
        self.groups, self.pending, self.timers = {}, {}, set()
        self.max_samples, self.input_chars, self.output_chars = max_samples, input_chars, output_chars
        self.stats = Counter()
        self.usage = Counter()
        self.started = time.monotonic()

    def count(self, key, n=1):
        with self.lock:
            self.stats[key] += n

    def submit(self, prompt, payload, cache_key, conversation_id, protocol):
        with self.lock:
            if cache_key in self.pending:
                self.stats["coalesced_samples"] += 1
                return self.pending[cache_key]
            future = Future()
            self.pending[cache_key] = future
            self.stats["submitted_samples"] += 1
            joint = protocol in JOINT_MODULES and "messages" in payload and "failures" not in payload
            # These three modules return target evidence, not historical context spans.
            evidence_payload = {k: v for k, v in payload.items() if k != "context"} if joint else payload
            estimate = max(1024, len(prompt) + 2 * len(compact(evidence_payload)))
            key = ("joint", None) if joint else (protocol, prompt)
            row = {
                "payload": payload,
                "protocol": protocol,
                "prompt": prompt,
                "cache_key": cache_key,
                "conversation_id": conversation_id,
                "future": future,
                "estimated_output": estimate,
            }
            groups = self.groups.setdefault(key, [])
            group = None
            for candidate in groups:
                request_prompt, request_payload = batch_request(candidate + [row])
                same_input = compact(shared_input(payload))
                inputs = {
                    (r["conversation_id"], compact(shared_input(r["payload"]))) for r in candidate + [row]
                }
                if (
                    conversation_id is None
                    or all(
                        r["conversation_id"] != conversation_id
                        or (
                            joint
                            and compact(shared_input(r["payload"])) == same_input
                            and r["protocol"] != protocol
                        )
                        for r in candidate
                    )
                ) and (
                    # A joint prompt labels rules by category only, so one category gets one rule per batch.
                    (not joint or all(r["protocol"] != protocol or r["prompt"] == prompt for r in candidate))
                    and (not joint or len(inputs) <= self.max_samples)
                    and len(request_prompt) + len(compact(request_payload)) <= self.input_chars
                    and sum(r["estimated_output"] for r in candidate) + estimate <= self.output_chars
                ):
                    group = candidate
                    break
            if group is None:
                group = []
                groups.append(group)
            group.append(row)
            if len(group) == 1:
                timer = threading.Timer(DEFAULT_BATCH_WAIT_SECONDS, self.flush, args=(key, group))
                timer.daemon = True
                self.timers.add(timer)
                timer.start()
            if len(group) >= self.max_samples * (len(JOINT_MODULES) if joint else 1):
                self._flush(key, group)
            return future

    def flush(self, key, expected):
        with self.lock:
            self.timers.discard(threading.current_thread())
            if any(group is expected for group in self.groups.get(key, [])):
                self._flush(key, expected)

    def _flush(self, key, rows):
        self.groups[key] = [group for group in self.groups[key] if group is not rows]
        if not self.groups[key]:
            del self.groups[key]
        self.pool.submit(self._run, key, rows)

    def _run(self, key, rows):
        protocol, _ = key
        try:
            pending = rows
            for attempt in range(2):
                prompt, payload = batch_request(pending)
                batch_id = uuid.uuid4().hex
                with self.lock:
                    self.stats["requests"] += 1
                    self.stats["retry_requests"] += int(attempt > 0)
                    self._save()
                sent_prompt, sent_payload = joint_transport(prompt, payload)
                response = self.generate(sent_prompt, sent_payload)
                write_json(
                    self.output / "batches" / f"{batch_id}.json",
                    {
                        "protocol": protocol,
                        "task_protocols": [r["protocol"] for r in pending],
                        "sample_cache_keys": [r["cache_key"] for r in pending],
                        "attempt": attempt,
                        "response": response,
                    },
                )
                data = response.get("data")
                results = data.get("results") if isinstance(data, dict) else None
                joint = "chats" in payload
                routes = chat_routes(pending) if joint else [(f"s{i}", None) for i in range(len(pending))]
                expected_fields = {}
                for sid, field in routes:
                    expected_fields.setdefault(sid, set()).add(field)
                if isinstance(results, list):
                    chats = (
                        payload["chats"]
                        if joint
                        else [dict(s["input"], sample_id=s["sample_id"]) for s in payload["samples"]]
                    )
                    if len(results) != len(chats):
                        results = None
                    else:
                        aligned = []
                        for index, (chat, item) in enumerate(zip(chats, results, strict=True)):
                            try:
                                restored = restore_chat_output(
                                    item, chat, None if joint else pending[index]["protocol"]
                                )
                            except ValueError:
                                restored = None
                            aligned.append({"sample_id": chat["sample_id"], "data": restored})
                        results = aligned
                mapped, duplicate = {}, set()
                if response["state"] == "ok" and isinstance(results, list):
                    for item in results:
                        if not isinstance(item, dict) or not isinstance(item.get("sample_id"), str):
                            continue
                        sid = item["sample_id"]
                        if sid in mapped:
                            duplicate.add(sid)
                        mapped[sid] = item
                    if set(mapped) - set(expected_fields):
                        mapped.clear()
                failed = []
                for row, (sid, field) in zip(pending, routes, strict=True):
                    item = mapped.get(sid)
                    result = item.get("data") if item is not None else None
                    if joint and isinstance(result, dict):
                        # Unrequested category keys are ignored, unknown keys mean a malformed result.
                        result = result.get(field) if set(result) <= set(JOINT_LABELS.values()) else None
                    if sid not in duplicate and isinstance(result, dict):
                        value = {
                            "state": "ok",
                            "data": result,
                            "raw": compact(result),
                            "model": response.get("model"),
                            "provider": response.get("provider"),
                            "cache_key": row["cache_key"],
                            "batch_id": batch_id,
                            "usage": None,
                        }
                        row["future"].set_result(value)
                        self.count("completed_samples")
                    else:
                        failed.append(row)
                with self.lock:
                    self.stats["completed_requests"] += 1
                    self.stats["input_chars"] += len(sent_prompt) + (
                        len(sent_payload) if isinstance(sent_payload, str) else len(compact(sent_payload))
                    )
                    self.stats["joint_requests"] += int(joint)
                    self.stats["output_chars"] += len(response.get("raw") or "")
                    self.stats["request_samples"] += len(pending)
                    self.stats["request_inputs"] += len(payload.get("chats", payload.get("samples", [])))
                    self.usage.update(
                        {k: v for k, v in (response.get("usage") or {}).items() if type(v) is int}
                    )
                    self._save()
                pending = failed
                if not pending:
                    break
            for row in pending:
                row["future"].set_result(
                    {
                        "state": "failure",
                        "error": "InvalidBatchSample",
                        "cache_key": row["cache_key"],
                        "batch_id": batch_id,
                        "raw": None,
                        "model": response.get("model"),
                        "usage": None,
                    }
                )
                self.count("failed_samples")
        except Exception as exc:
            for row in rows:
                if not row["future"].done():
                    row["future"].set_exception(exc)
        finally:
            with self.lock:
                for row in rows:
                    self.pending.pop(row["cache_key"], None)
                self._save()

    def _save(self):
        write_json(
            self.output / "progress.json",
            dict(
                self.stats,
                usage=dict(self.usage),
                elapsed_seconds=round(time.monotonic() - self.started, 1),
            ),
        )

    def close(self):
        with self.lock:
            for timer in self.timers:
                timer.cancel()
            for key in list(self.groups):
                for group in list(self.groups[key]):
                    self._flush(key, group)
        self.pool.shutdown(wait=True)
        with self.lock:
            self._save()
