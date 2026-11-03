"""Shared LTP parser, cached LLM client and original-text evidence lookup."""

import concurrent.futures
import json
import re
import uuid
from copy import copy, deepcopy
from pathlib import Path

from weclone.utils import secure_storage

from .data import digest, write_json
from .batching import (
    BATCH_PROMPT,
    BATCH_PROTOCOL,
    BatchDispatcher,
    compact,
    joint_transport,
    restore_chat_output,
)
from .wire import (
    MODULES,
    WIRE_VERSION,
    schema_for,
    format_prompt,
    pack_input,
    expand_output,
    expand_evidence,
    unique_object,
)

DEFAULT_BATCH_SIZE = 32


DEFAULT_WORKERS = 64


DEFAULT_TIMEOUT = 120
# LLM batches are filled in queue order, so the queue is shuffled with this seed before submission.
QUEUE_SEED = 0
# LTP.pipeline fixes its tokenizer max_length to 512, including special tokens.
DEFAULT_LTP_MAX_TOKENS = 512
# The model sometimes types a space into a state string and appends `.replace(" ","")` to undo it.
STATE_REPLACE = re.compile(r'"([paunk ]+)"\.replace(?:\(\s*" "\s*,\s*""\s*\))?')


def repair_state_strings(raw):
    return STATE_REPLACE.sub(lambda m: '"' + m.group(1).replace(" ", "") + '"', raw)


def locate(evidence, messages):
    evidence = expand_evidence(evidence)
    if not isinstance(evidence, dict):
        raise ValueError("Missing evidence object")
    text = {m["message_id"]: m["text"] for m in messages}[evidence["message_id"]]
    quote, occurrence = evidence["quote"], evidence["occurrence"]
    if not isinstance(quote, str) or not quote or type(occurrence) is not int or occurrence < 0:
        raise ValueError("Invalid quote or occurrence")
    starts = [m.start() for m in re.finditer("(?=" + re.escape(quote) + ")", text)]
    if occurrence >= len(starts):
        raise ValueError("Evidence not found in original text")
    return dict(evidence, start=starts[occurrence], end=starts[occurrence] + len(quote))


def merge_parses(parts):
    result = {key: [] for key in ("cws", "pos", "starts", "ends", "chunks")}
    syntax = "dep" in parts[0][2]
    if syntax:
        result.update(dep={"head": [], "label": []}, sdp={"head": [], "label": []})
    for offset, text, value in parts:
        base = len(result["cws"])
        for key in ("cws", "pos"):
            result[key].extend(value[key])
        for key in ("starts", "ends"):
            result[key].extend(offset + position for position in value[key])
        result["chunks"].append({"start": offset, "end": offset + len(text)})
        if syntax:
            for key in ("dep", "sdp"):
                result[key]["head"].extend(h + base if h else 0 for h in value[key]["head"])
                result[key]["label"].extend(value[key]["label"])
    return result


class Judge:
    def __init__(self, settings, cache_dir, model=None, base_url=None, *, protocol, provider=None):
        import pyjson5

        config = pyjson5.loads(Path(settings).read_text())
        self.protocol = protocol
        self.batcher = None
        self.conversation_id = None
        self.owns_client = True
        self.provider = provider or config.get("agent_distill_args", {}).get("llm_provider", "codex_exec")
        if self.provider not in ("codex_exec", "api"):
            raise ValueError(f"Unknown LLM provider: {self.provider}")
        self.backend_options = {}
        if self.provider == "codex_exec":
            from weclone.core.inference.llm_client import CodexExecClient

            if base_url is not None:
                raise ValueError("--base-url requires --llm-provider api")
            options = config["codex_exec_args"]
            self.model = model or options["model"]
            self.url = None
            self.backend_options = {k: options[k] for k in ("effort", "timeout", "command", "sandbox")}
            self.client = CodexExecClient(
                model=self.model,
                effort=options["effort"],
                timeout=options["timeout"],
                command=options["command"],
                sandbox=options["sandbox"],
                isolated_cwd=True,
                config_path=settings,
            )
        else:
            options = config["make_dataset_args"]
            self.model = model or options["model_name"]
            self.url = (base_url or options["base_url"]).rstrip("/")
            self.key = options["llm_api_key"]
            from weclone.core.inference import OpenAICompatibleClient, RetryPolicy

            self.client = OpenAICompatibleClient(
                api_key=self.key,
                base_url=self.url,
                model=self.model,
                timeout=DEFAULT_TIMEOUT,
                retry_policy=RetryPolicy(max_retries=0),
            )
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)

    def configure_batches(self, output, workers, max_samples, input_chars, output_chars):
        if max_samples > 1:
            self.batcher = BatchDispatcher(
                self._generate, output, workers, max_samples, input_chars, output_chars
            )

    def for_record(self, row):
        judge = copy(self)
        judge.conversation_id = (row["person_id"], row["conversation_id"])
        judge.dialogue = row.get("dialogue")
        return judge

    def call(self, prompt, payload, refresh=False, attempt=0):
        """Cached LLM call. `attempt` keys a repair or retry apart from the first call, so it is cached too;
        `refresh` draws a new uncached sample (QA repeats)."""
        if (
            isinstance(payload, dict)
            and "messages" in payload
            and getattr(self, "dialogue", None) is not None
        ):
            payload = dict(payload, dialogue=self.dialogue)
        schema = schema_for(self.protocol, payload)
        wire = self.protocol in MODULES and isinstance(payload, dict) and "context" in payload
        if schema is not None:
            prompt = format_prompt(schema) + prompt
        if wire:
            full_schema = schema_for(self.protocol, {k: v for k, v in payload.items() if k != "failures"})
            payload = pack_input(payload, full_schema)
        identity = {
            "protocol": self.protocol,
            "model": self.model,
            "url": self.url,
            "prompt": prompt,
            "payload": payload,
            "temperature": 0,
            "transport": BATCH_PROTOCOL,
        }
        if wire:
            identity["wire_protocol"] = WIRE_VERSION
        if self.provider == "codex_exec":
            identity.update(provider=self.provider, backend_options=self.backend_options)
        if self.batcher is not None:
            identity["batch_protocol"] = BATCH_PROTOCOL
        if attempt:
            identity["attempt"] = attempt
        cache_key = digest(identity)
        path = self.cache / f"{cache_key}.json"
        if refresh:
            cache_key = digest({"request_key": cache_key, "repeat_id": uuid.uuid4().hex})
            path = self.cache / f"{cache_key}.json"
        if secure_storage.file_exists(path) and not refresh:
            value = secure_storage.read_json(path)
            if value["state"] == "ok" and not value.get("validation_failed", False):
                if self.batcher is not None:
                    self.batcher.count("cache_hits")
                return value
        discovery = MODULES.get(self.protocol) == "lexical_choice" and isinstance(payload, list)
        if discovery:
            chats = [pack_input({"context": [], "messages": r["messages"]}) for r in payload]
            sent_prompt, sent_payload = joint_transport(prompt, {"chats": chats})
            value = self._generate(sent_prompt, sent_payload, cache_key)
            if value["state"] == "ok":
                try:
                    results = value["data"]["results"]
                    if not isinstance(results, list) or len(results) != len(chats):
                        raise ValueError("Incomplete discovery dialogue coverage")
                    candidates = []
                    for row, chat, result in zip(payload, chats, results, strict=True):
                        result = expand_evidence(restore_chat_output(result, chat))
                        if not isinstance(result.get("candidates"), list):
                            raise ValueError("Missing discovery candidates")
                        for candidate in result["candidates"]:
                            candidate["evidence"]["sample_id"] = row["sample_id"]
                            candidates.append(candidate)
                    value["data"] = {"candidates": candidates}
                except (KeyError, TypeError, ValueError) as exc:
                    value.update(state="failure", error=f"InvalidChatOutput: {exc}")
                    value.pop("data", None)
        elif self.batcher is not None:
            value = deepcopy(
                self.batcher.submit(
                    prompt,
                    payload,
                    cache_key,
                    self.conversation_id,
                    self.protocol,
                ).result()
            )
        else:
            if wire:
                sent_prompt, sent_payload = joint_transport(BATCH_PROMPT + prompt, {"chats": [payload]})
                value = self._generate(sent_prompt, sent_payload, cache_key)
                if value["state"] == "ok":
                    try:
                        results = value["data"].get("results") if isinstance(value["data"], dict) else None
                        if not isinstance(results, list) or len(results) != 1:
                            raise ValueError("Expected one dialogue result")
                        value["data"] = restore_chat_output(results[0], payload, self.protocol)
                    except ValueError as exc:
                        value.update(state="failure", error=f"InvalidChatOutput: {exc}")
                        value.pop("data", None)
            else:
                value = self._generate(prompt, payload, cache_key)
        if wire and value["state"] == "ok":
            try:
                output_schema = schema
                data = value["data"]
                if (
                    schema is not None
                    and schema != full_schema
                    and isinstance(data, dict)
                    and (
                        isinstance(data.get("states"), dict)
                        and set(data["states"]) == set(full_schema["keys"])
                        or isinstance(data.get("states"), str)
                        and len(data["states"])
                        == len(full_schema["keys"])
                        * (len(full_schema["message_ids"]) if full_schema["per_message"] else 1)
                    )
                ):
                    output_schema = full_schema
                value["data"] = expand_output(data, output_schema) if output_schema else expand_evidence(data)
            except (ValueError, KeyError, TypeError) as exc:
                value.update(state="failure", error=f"InvalidCompactOutput: {exc}")
                value.pop("data", None)
        write_json(path, value)
        return value

    def _generate(self, prompt, payload, cache_key=None):
        if self.provider == "codex_exec":
            from weclone.core.inference.llm_client import LLMRequest

            raw = None
            try:
                response = self.client.generate(
                    LLMRequest(
                        messages=[
                            {"role": "system", "content": prompt},
                            {
                                "role": "user",
                                "content": payload if isinstance(payload, str) else compact(payload),
                            },
                        ],
                        model=self.model,
                        provider=self.provider,
                        json_mode=True,
                        effort=self.backend_options["effort"],
                        timeout=self.backend_options["timeout"],
                    )
                )
                raw = response.text
                value = {
                    "state": "ok" if response.ok else "failure",
                    "raw": raw,
                    "model": response.model,
                    "provider": self.provider,
                    "usage": response.metadata.get("usage"),
                    "cache_key": cache_key,
                }
                if response.ok:
                    value["data"] = json.loads(raw, object_pairs_hook=unique_object)
                elif raw and (repaired := repair_state_strings(raw)) != raw:
                    value.update(state="ok", raw=repaired)
                    value["data"] = json.loads(repaired, object_pairs_hook=unique_object)
                else:
                    value["error"] = "CodexExecFailure"
            except Exception as exc:
                value = {
                    "state": "failure",
                    "error": type(exc).__name__,
                    "raw": raw,
                    "model": self.model,
                    "provider": self.provider,
                    "cache_key": cache_key,
                }
            return value
        response = self.client.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": payload if isinstance(payload, str) else compact(payload)},
            ],
            temperature=0,
            json_mode=True,
            max_tokens=None,
        )
        if response.ok:
            value = {
                "state": "ok",
                "data": response.parsed_json,
                "raw": response.text,
                "model": response.model,
                "usage": response.metadata.get("usage"),
                "cache_key": cache_key,
            }
            try:
                value["data"] = json.loads(response.text, object_pairs_hook=unique_object)
            except (ValueError, TypeError) as exc:
                value.update(state="failure", error=f"InvalidJSON: {exc}")
                value.pop("data", None)
        else:
            value = {
                "state": "failure",
                "error": response.metadata.get("error_type", "InferenceFailure"),
                "cache_key": cache_key,
                "http_status": response.metadata.get("http_status"),
                "raw": response.text,
                "model": response.model if response.metadata.get("http_status") == 200 else None,
                "usage": response.metadata.get("usage"),
            }
        return value

    def invalidate(self, *cache_keys):
        for key in dict.fromkeys(k for k in cache_keys if k is not None):
            path = self.cache / f"{key}.json"
            value = secure_storage.read_json(path)
            value["validation_failed"] = True
            write_json(path, value)
            if self.batcher is not None:
                self.batcher.count("invalidated_samples")

    def for_protocol(self, protocol):
        """A judge for another group on this judge's client, so launch pacing and cooldown stay shared."""
        judge = copy(self)
        judge.protocol, judge.batcher, judge.owns_client = protocol, None, False
        return judge

    def close(self):
        if self.batcher is not None:
            self.batcher.close()
        if self.owns_client:
            self.client.close()


class Parser:
    def __init__(self, model_path, cache_dir, device="cpu", batch_size=DEFAULT_BATCH_SIZE):
        self.path = Path(model_path)
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.device, self.batch_size = device, batch_size
        self.model = None
        self.identity = digest(
            {
                "protocol": "style-ltp-offset-chunks-v3",
                "max_tokens": DEFAULT_LTP_MAX_TOKENS,
                "model": {
                    p.name: {"size": p.stat().st_size, "mtime": p.stat().st_mtime_ns}
                    for p in self.path.iterdir()
                    if p.is_file()
                },
            }
        )

    def load_model(self):
        if self.model is None:
            from ltp import LTP

            self.model = LTP(str(self.path)).to(self.device)

    def split_text(self, text):
        self.load_model()
        tokenizer = self.model.tokenizer
        budget = DEFAULT_LTP_MAX_TOKENS - tokenizer.num_special_tokens_to_add(pair=False)
        parts, offset = [], 0
        while offset < len(text):
            remaining = text[offset:]
            encoded = tokenizer(
                remaining,
                truncation=False,
                add_special_tokens=False,
                return_offsets_mapping=True,
                verbose=False,
            )
            if len(encoded["input_ids"]) <= budget:
                parts.append((offset, remaining))
                break
            end = encoded["offset_mapping"][budget - 1][1]
            boundaries = list(re.finditer(r"[。！？!?；;\n.]", remaining[:end]))
            if boundaries:
                end = boundaries[-1].end()
            if not 0 < end < len(remaining):
                raise ValueError("Cannot split LTP input at a token-aligned character boundary")
            parts.append((offset, remaining[:end]))
            offset += end
        return parts

    def pipeline(self, texts, tasks):
        self.load_model()
        encoded = self.model.tokenizer(texts, truncation=False, padding=False, verbose=False)
        for index, ids in enumerate(encoded["input_ids"]):
            if len(ids) > DEFAULT_LTP_MAX_TOKENS:
                raise ValueError(
                    f"LTP input {index} has {len(ids)} tokens including special tokens; "
                    f"limit is {DEFAULT_LTP_MAX_TOKENS}. Split long input explicitly before parsing."
                )
        return self.model.pipeline(texts, tasks=tasks)

    def parse(self, texts):
        result, missing = {}, []
        for text in dict.fromkeys(texts):
            path = self.cache / (digest({"parser": self.identity, "text": text}) + ".json")
            if secure_storage.file_exists(path):
                result[text] = secure_storage.read_json(path)
            elif not text.strip():
                result[text] = {"cws": [], "pos": [], "starts": [], "ends": []}
            else:
                chunks = self.split_text(text)
                if len(chunks) > 1:
                    values = self.parse([part for _, part in chunks])
                    value = merge_parses([(offset, part, values[part]) for offset, part in chunks])
                    result[text] = value
                    write_json(path, value)
                else:
                    missing.append((text, path))
        for i in range(0, len(missing), self.batch_size):
            batch = missing[i : i + self.batch_size]
            output = self.pipeline([t for t, p in batch], tasks=["cws", "pos"])
            for j, (text, path) in enumerate(batch):
                starts, ends, cursor, words, tags = [], [], 0, [], []
                for word, tag in zip(output.cws[j], output.pos[j], strict=True):
                    start = text.find(word, cursor)
                    if start < 0:
                        raise ValueError("Parser tokens cannot be aligned with original text")
                    cursor = start + len(word)
                    if word.strip():
                        starts.append(start + len(word) - len(word.lstrip()))
                        ends.append(cursor - (len(word) - len(word.rstrip())))
                        words.append(word.strip())
                        tags.append(tag)
                # A truncated parse must never count as a complete short reply.
                covered = {k for s, e in zip(starts, ends) for k in range(s, e)}
                if any(not c.isspace() and k not in covered for k, c in enumerate(text)):
                    raise ValueError("Parser dropped characters; split long input explicitly")
                value = {"cws": words, "pos": tags, "starts": starts, "ends": ends}
                if len(value["cws"]) != len(value["pos"]):
                    raise ValueError("Parser token/POS length mismatch")
                result[text] = value
                write_json(path, value)
        return result


def queue_order(values, key):
    """`values` in a fixed pseudo-random order by `key`.

    Input files run contact by contact and in time order. Submitting them as they are would fill each LLM
    batch with one contact's adjacent windows, and relative judgments (politeness, intensity, register,
    monologue) would be calibrated against those neighbours.
    """
    return sorted(values, key=lambda value: (digest(["queue", QUEUE_SEED, key(value)]), key(value)))


def parallel_map(fn, values, workers=DEFAULT_WORKERS, *, judge=None):
    # Waiting samples must not consume the concurrency budget of whole LLM batches.
    batch_size = judge.batcher.max_samples if judge is not None and judge.batcher is not None else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers * batch_size) as pool:
        return list(pool.map(fn, values))


def annotate_corpora(corpora, fn, workers=DEFAULT_WORKERS, *, judge=None, kinds=None):
    """Annotate every distinct record once; corpora sharing a record share its annotation.

    With `kinds`, all kinds of one record are queued together so joint batches stay joint.
    """
    unique = {}
    for rows in corpora.values():
        for row in rows:
            unique.setdefault(row["sample_id"], row)
    rows = queue_order(unique.values(), lambda row: row["sample_id"])
    jobs = [(row, kind) for row in rows for kind in (kinds or (None,))]
    values = parallel_map(
        (lambda job: fn(job[0])) if kinds is None else (lambda job: fn(*job)), jobs, workers, judge=judge
    )
    done = {(row["sample_id"], kind): value for (row, kind), value in zip(jobs, values, strict=True)}
    if kinds is None:
        return {name: [done[(r["sample_id"], None)] for r in rows] for name, rows in corpora.items()}
    return {
        name: {kind: [done[(r["sample_id"], kind)] for r in rows] for kind in kinds}
        for name, rows in corpora.items()
    }


def distinct_records(corpora):
    return len({row["sample_id"] for rows in corpora.values() for row in rows})
