"""Lexical Choice: metrics, prompts, annotation and discovery."""

from weclone.prompts.distill_style import (
    build_annotation_prompt as build_annotation_prompt,
    LEXICAL_CHOICE_ITEMS as ITEMS,
    LEXICAL_CHOICE_ANNOTATION_PROMPT as ANNOTATION_PROMPT,  # noqa: F401 -- public prompt metadata
    TOPIC_PROMPT as TOPIC_PROMPT,
    DISCOVERY_PROMPT as DISCOVERY_PROMPT,
)

import math
import re
import statistics
import unicodedata
from collections import Counter
from datetime import datetime

import regex

from .data import PRIMARY_CORPUS, bubble_spans, digest, target_fingerprints, write_json
from .extraction import annotate_corpora, distinct_records, locate, parallel_map, queue_order
from .stats import DEFAULT_TOP_K, by_length, message_texts, missing_messages, ratio, reliability, with_missing

DEFAULT_BLOCK_SIZE = 200

DEFAULT_HDD_SAMPLE = 42

DEFAULT_MTLD_THRESHOLD = 0.72

FUNCTION_POS = frozenset({"p", "c", "u", "e", "o", "d"})

CONTENT_POS = frozenset({"n", "nh", "ni", "nl", "ns", "nt", "nz", "v", "a", "i", "j"})

# Closed grammatical inventory, intersected with LTP's r tag; never substring matching.
PERSONAL_PRONOUNS = frozenset(
    "我 你 您 他 她 它 咱 俺 我们 你们 您们 他们 她们 它们 咱们 俺们 自己 本人 人家".split()
)

PROTOCOL_VERSION = "lexical-v5"
SCOPE = "Lexical Choice 001-024; excludes 005/013/014"


DEFAULT_DISCOVERY_BATCH = 32

DEFAULT_MIN_CONVERSATIONS = 2


def word_token(token, pos):
    return pos != "wp" and any(unicodedata.category(c)[0] in "LN" for c in token)


# Laughter, crying and hesitation strings: a run of one sound character, whatever LTP tagged it.
# Closed set of sound syllables; single interjections (啊, 嗯, 哦) stay lexical.
VOCAL_RUN = regex.compile(
    r"([哈呵嘿嘻嘤呜嗷噗咯嘎桀])\1+|(?i)(?<![a-z])(?:a*h{2,}a*|(?:ha){2,}h?|e+m{2,}|m{3,})(?![a-z])"
)


def vocal_token(token):
    """Token that writes a vocal sound: at least half its nonspace characters in sound runs (哈哈, hh, emmm)."""
    nonspace = sum(not c.isspace() for c in token)
    return bool(nonspace) and 2 * sum(len(m.group()) for m in VOCAL_RUN.finditer(token)) >= nonspace


def lexical_token(token, pos):
    """Word token for the lexical statistics; vocal strings belong to 099-105."""
    return word_token(token, pos) and not vocal_token(token)


def mtld(tokens, threshold=DEFAULT_MTLD_THRESHOLD):
    if not tokens:
        return None

    def direction(seq):
        types, length, factors = set(), 0, 0.0
        for token in seq:
            types.add(token)
            length += 1
            if len(types) / length <= threshold:
                factors += 1
                types, length = set(), 0
        if length:
            factors += (1 - len(types) / length) / (1 - threshold)
        return len(tokens) / factors if factors else None

    values = [direction(tokens), direction(reversed(tokens))]
    return statistics.mean(values) if all(v is not None for v in values) else None


def hdd(tokens, sample=DEFAULT_HDD_SAMPLE):
    n = len(tokens)
    if n < sample:
        return None
    denominator = math.comb(n, sample)
    return (
        sum(
            1 - (math.comb(n - count, sample) / denominator if n - count >= sample else 0)
            for count in Counter(tokens).values()
        )
        / sample
    )


def diversity(tokens, block_size=DEFAULT_BLOCK_SIZE):
    blocks = [tokens[i : i + block_size] for i in range(0, len(tokens) - block_size + 1, block_size)]
    values = [{"ttr": len(set(b)) / block_size, "mtld": mtld(b), "hdd": hdd(b)} for b in blocks]
    return {
        "block_size": block_size,
        "n_blocks": len(blocks),
        "discarded_tokens": len(tokens) % block_size,
        "blocks": values,
        "mean": {
            k: statistics.mean(v[k] for v in values if v[k] is not None)
            if any(v[k] is not None for v in values)
            else None
            for k in ("ttr", "mtld", "hdd")
        },
    }


def zipf(tokens):
    ranked = sorted(Counter(tokens).items(), key=lambda p: (-p[1], p[0]))
    points = [
        {"rank": i, "word": w, "frequency": f, "log_rank": math.log(i), "log_frequency": math.log(f)}
        for i, (w, f) in enumerate(ranked, 1)
    ]
    if len(points) < 2:
        return {"points": points, "slope": None, "intercept": None, "r_squared": None}
    x, y = [p["log_rank"] for p in points], [p["log_frequency"] for p in points]
    xm, ym = statistics.mean(x), statistics.mean(y)
    slope = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / sum((a - xm) ** 2 for a in x)
    intercept = ym - slope * xm
    variance = sum((b - ym) ** 2 for b in y)
    r2 = 1 - sum((b - intercept - slope * a) ** 2 for a, b in zip(x, y)) / variance if variance else None
    return {"points": points, "slope": slope, "intercept": intercept, "r_squared": r2}


def legacy_patterns(text):
    """The old raw-string mixed-pattern extractor.

    Extraction no longer reports it; WC-exp's style marker script still imports it.
    """

    def emoji(c):
        return unicodedata.category(c) == "So" or any(
            lo <= c <= hi
            for lo, hi in (
                ("\U0001f600", "\U0001f64f"),
                ("\U0001f300", "\U0001f5ff"),
                ("\U0001f680", "\U0001f6ff"),
                ("\U0001f1e0", "\U0001f1ff"),
                ("\u2600", "\u26ff"),
                ("\u2700", "\u27bf"),
                ("\U0001f900", "\U0001f9ff"),
                ("\U0001fa70", "\U0001faff"),
            )
        )

    result, i = [], 0
    while i < len(text):
        start, c = i, text[i]
        if "\u4e00" <= c <= "\u9fff":
            while i < len(text) and (text[i].isalpha() or text[i].isdigit()):
                i += 1
            seq = text[start:i]
            if re.fullmatch(r"[\u4e00-\u9fff]+", seq):
                result.extend(seq[j : j + n] for n in range(1, 6) for j in range(len(seq) - n + 1))
            else:
                result.append(seq)
        elif c.isalnum():
            while i < len(text) and text[i].isalnum():
                i += 1
            result.append(text[start:i])
        elif emoji(c):
            while i < len(text) and (emoji(text[i]) or text[i] in "\ufe0f\u200d"):
                i += 1
            result.append(text[start:i])
        elif c == "[" and text.find("]", i) - i >= 2:
            i = text.find("]", i) + 1
            result.append(text[start:i])
        elif c in "？！。，；：.":
            while i < len(text) and text[i] in "？！。，；：.":
                i += 1
            result.append(text[start:i])
        elif not c.isspace():
            while i < len(text) and text[i] == c:
                i += 1
            result.append(text[start:i])
        else:
            i += 1
    return result


def script_stats(text):
    spans = [
        {
            "start": m.start(),
            "end": m.end(),
            "text": m.group(),
            "script": "han" if regex.match(r"\p{Script=Han}", m.group()) else "latin",
        }
        for m in regex.finditer(r"\p{Script=Han}+|\p{Script=Latin}+", text)
    ]
    return {
        "spans": spans,
        "mixed": len({p["script"] for p in spans}) == 2,
        "switches": sum(a["script"] != b["script"] for a, b in zip(spans, spans[1:])),
    }


def summarize(records, parsed, block_size=DEFAULT_BLOCK_SIZE):
    words, functions, pronouns, pos, lengths = (Counter() for _ in range(5))
    ngrams = {str(n): Counter() for n in range(1, 5)}
    chars = {str(n): Counter() for n in range(1, 6)}
    tokens, per_message, vocal = [], [], Counter()
    for row in records:
        for msg in row["messages"]:
            p = parsed[msg["text"]]
            kept = [(t, s) for t, s in zip(p["cws"], p["pos"], strict=True) if lexical_token(t, s)]
            vocal.update(
                t for t, s in zip(p["cws"], p["pos"], strict=True) if word_token(t, s) and vocal_token(t)
            )
            seq = [t for t, s in kept]
            tokens.extend(seq)
            words.update(seq)
            pos.update(s for t, s in kept)
            lengths.update(str(len(t)) for t in seq)
            functions.update(t for t, s in kept if s in FUNCTION_POS)
            pronouns.update(t for t, s in kept if s == "r" and t in PERSONAL_PRONOUNS)
            # Break n-grams at punctuation, vocal strings and original newline boundaries.
            runs, run = [], []
            last_end = 0
            for t, s, start, end in zip(p["cws"], p["pos"], p["starts"], p["ends"], strict=True):
                if not lexical_token(t, s) or "\n" in msg["text"][last_end:start]:
                    if run:
                        runs.append(run)
                    run = []
                if lexical_token(t, s):
                    run.append(t)
                last_end = end
            if run:
                runs.append(run)
            for n in range(1, 5):
                ngrams[str(n)].update(json_key(r[i : i + n]) for r in runs for i in range(len(r) - n + 1))
            for chunk in regex.findall(r"\p{Script=Han}+", msg["text"]):
                for n in range(1, 6):
                    chars[str(n)].update(chunk[i : i + n] for i in range(len(chunk) - n + 1))
            per_message.append(
                {
                    "sample_id": row["sample_id"],
                    "message_id": msg["message_id"],
                    "word_count": len(seq),
                    "function_count": sum(s in FUNCTION_POS for t, s in kept),
                    "personal_pronoun_count": sum(s == "r" and t in PERSONAL_PRONOUNS for t, s in kept),
                    "script": script_stats(msg["text"]),
                }
            )
    total = len(tokens)
    return {
        "n_windows": len(records),
        "n_messages": len(per_message),
        "word_tokens": total,
        "excluded_vocal_tokens": sum(vocal.values()),
        "excluded_vocal_forms": vocal,
        "words": words,
        "function_words": functions,
        "function_rate": ratio(sum(functions.values()), total),
        "personal_pronouns": pronouns,
        "personal_pronoun_rate": ratio(sum(pronouns.values()), total),
        "pos": pos,
        "word_lengths": lengths,
        "mean_word_length": ratio(sum(len(t) for t in tokens), total),
        "top_words": sorted(words.items(), key=lambda p: (-p[1], p[0]))[:DEFAULT_TOP_K],
        "top_mass": {str(k): ratio(sum(v for w, v in words.most_common(k)), total) for k in (50, 100)},
        "diversity": diversity(tokens, block_size),
        "zipf": zipf(tokens),
        "word_ngrams": ngrams,
        "han_character_ngrams": chars,
        "per_message": per_message,
        "_tokens": tokens,
    }


def json_key(parts):
    import json

    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def correction_applicable(boundary, messages):
    """015 needs two adjacent own sends: two lines of one chat turn, or two messages when boundary=message."""
    return (boundary == "message" and len(messages) >= 2) or any(
        len(bubble_spans(m["text"])) >= 2 for m in messages
    )


def adjacent_sends(row):
    """Number of (earlier send, next send) pairs a 015 correction can span."""
    lines = sum(max(0, len(bubble_spans(m["text"])) - 1) for m in row["messages"])
    return lines + (max(0, len(row["messages"]) - 1) if row["boundary"] == "message" else 0)


def bubble_of(text, start, end):
    return next((i for i, (s, e) in enumerate(bubble_spans(text)) if s <= start and end <= e), None)


def validate_item(item, value, row, tokens):
    if value["state"] not in ("present", "absent", "unknown", "na"):
        raise ValueError("Invalid item state")
    events = value["events"]
    if not isinstance(events, list) or (value["state"] == "present") != bool(events):
        raise ValueError("Inconsistent state/events")
    seen = set()
    for event in events:
        if event["label"] not in ITEMS[item][1] or not isinstance(event["attributes"], dict):
            raise ValueError("Invalid label/attributes")
        event["evidence"] = locate(event["evidence"], row["messages"])
        e = event["evidence"]
        key = (e["message_id"], e["start"], e["end"])
        if key in seen:
            raise ValueError("Duplicate event")
        seen.add(key)
        required = {
            "008": ("domain",),
            "009": ("domain",),
            "010": ("expansion",),
            "012": ("region",),
        }.get(item, ())
        if set(event["attributes"]) != set(required):
            raise ValueError("Unexpected or missing attributes")
        if any(v is not None and not isinstance(v, str) for v in event["attributes"].values()):
            raise ValueError("Invalid attribute value")
        if item == "015":
            source = event["source"] = locate(event["source"], row["messages"])
            ids = [m["message_id"] for m in row["messages"]]
            if source["message_id"] == e["message_id"]:
                text = row["messages"][ids.index(e["message_id"])]["text"]
                before = bubble_of(text, source["start"], source["end"])
                after = bubble_of(text, e["start"], e["end"])
                if before is None or after is None or after - before != 1:
                    raise ValueError("Correction must refer to the immediately preceding line of the turn")
            elif (
                row["boundary"] != "message"
                or ids.index(e["message_id"]) - ids.index(source["message_id"]) != 1
            ):
                raise ValueError("Correction must refer to immediately preceding own message")
        elif event["source"] is not None:
            raise ValueError("Unexpected correction source")
    if item == "011":
        for msg in row["messages"]:
            spans = sorted((start, end) for mid, start, end in seen if mid == msg["message_id"])
            if (
                spans
                and len(
                    {
                        part["script"]
                        for start, end in spans
                        for part in script_stats(msg["text"][start:end])["spans"]
                    }
                )
                != 2
            ):
                raise ValueError("Mixed-language events require both Chinese and English in the same message")
            if any(right[0] < left[1] for left, right in zip(spans, spans[1:])):
                raise ValueError("Overlapping language spans would double-count switches")
    return value


def annotate(row, parsed, judge, refresh=False):
    judge = judge.for_record(row)
    tokens = []
    requested = list(ITEMS)
    na_correction = not correction_applicable(row["boundary"], row["messages"])
    if na_correction:
        requested.remove("015")
    prompt = build_annotation_prompt(requested)
    payload = {
        "context": row["context"],
        "messages": row["messages"],
        "boundary": row["boundary"],
    }
    response = judge.call(prompt, payload, refresh=refresh)
    result = {}
    data = response.get("data", {})
    # Observed JSON-mode output sometimes omits only the envelope. Never fill missing items.
    direct_item_map = isinstance(data, dict) and set(data) == set(requested)
    item_values = data if direct_item_map else data.get("items", {}) if isinstance(data, dict) else {}
    for item in requested:
        try:
            if response["state"] != "ok":
                raise ValueError(response.get("error", "call_failed"))
            value = item_values[item]
            result[item] = validate_item(item, value, row, tokens)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            result[item] = {"state": "failure", "events": [], "reason": str(exc)}
    failures = {key: value["reason"] for key, value in result.items() if value["state"] == "failure"}
    repair = None
    if failures:
        repair = judge.call(
            prompt,
            dict(payload, failures=failures),
            refresh=refresh,
            attempt=1,
        )
        fixed = repair.get("data", {})
        fixed = fixed.get("items", {}) if isinstance(fixed, dict) else {}
        for item in failures:
            try:
                result[item] = validate_item(item, fixed[item], row, tokens)
            except (KeyError, ValueError, TypeError, IndexError) as exc:
                result[item] = {"state": "failure", "events": [], "reason": str(exc)}
    if na_correction:
        result["015"] = {"state": "na", "events": [], "reason": "No two adjacent own sends"}
    if any(value["state"] == "failure" for value in result.values()):
        judge.invalidate(response["cache_key"], repair["cache_key"] if repair else None)
    return {
        "sample_id": row["sample_id"],
        "items": result,
        "cache_key": response["cache_key"],
        "repair_cache_key": repair["cache_key"] if repair else None,
        "initial_failures": failures,
        "remaining_failures": {
            key: value["reason"] for key, value in result.items() if value["state"] == "failure"
        },
        "item_cache_keys": {
            key: repair["cache_key"] if key in failures and repair else response["cache_key"]
            for key in result
        },
        "normalized_direct_item_map": direct_item_map,
    }


def classify_topic(context, judge):
    if not context:
        return {"label": "unknown", "evidence": None, "state": "unknown"}
    response = judge.call(TOPIC_PROMPT, {"context": context})
    try:
        value = response["data"]
        if value["label"] not in (
            "daily",
            "work",
            "study",
            "technology",
            "relationship",
            "entertainment",
            "transaction",
            "other",
            "unknown",
        ):
            raise ValueError("Invalid topic")
        if value["label"] != "unknown":
            value["evidence"] = locate(value["evidence"], context)
        value["state"] = "unknown" if value["label"] == "unknown" else "present"
        return value
    except (KeyError, ValueError, TypeError, IndexError):
        judge.invalidate(response["cache_key"])
        return {"label": "unknown", "evidence": None, "state": "failure"}


def discover(
    records, judge, workers, min_conversations=DEFAULT_MIN_CONVERSATIONS, batch_size=DEFAULT_DISCOVERY_BATCH
):
    by_id = {r["sample_id"]: r for r in records}
    queued = queue_order(records, lambda r: r["sample_id"])
    batches = [queued[i : i + batch_size] for i in range(0, len(queued), batch_size)]

    def call(batch):
        return judge.call(
            DISCOVERY_PROMPT, [{"sample_id": r["sample_id"], "messages": r["messages"]} for r in batch]
        )

    replies = parallel_map(call, batches, workers)
    proposed, invalid, failed = {}, 0, 0
    for batch, response in zip(batches, replies, strict=True):
        data = response.get("data")
        if (
            response["state"] != "ok"
            or not isinstance(data, dict)
            or not isinstance(data.get("candidates"), list)
        ):
            judge.invalidate(response["cache_key"])
            failed += 1
            continue
        allowed_ids = {r["sample_id"] for r in batch}
        invalid_before = invalid
        for candidate in response["data"]["candidates"]:
            try:
                evidence = candidate["evidence"]
                if evidence["sample_id"] not in allowed_ids:
                    raise ValueError("Evidence outside batch")
                evidence = locate(evidence, by_id[evidence["sample_id"]]["messages"])
                if candidate["item"] != "004,006" or candidate["kind"] not in (
                    "expression",
                    "entity",
                    "topic_term",
                ):
                    raise ValueError("Invalid candidate")
                if candidate["expression"] != evidence["quote"]:
                    raise ValueError("Expression is not exact evidence")
                key = (candidate["item"], candidate["expression"], candidate["kind"])
                proposed[key] = dict(candidate, evidence=evidence)
            except (KeyError, ValueError, TypeError, IndexError):
                invalid += 1
        if invalid > invalid_before:
            judge.invalidate(response["cache_key"])
    candidates = []
    for (item, expression, kind), candidate in sorted(proposed.items()):
        conversations = set()
        occurrences = 0
        for row in records:
            counts = sum(
                len(list(re.finditer("(?=" + re.escape(expression) + ")", m["text"])))
                for m in row["messages"]
            )
            if counts:
                conversations.add(row["conversation_id"])
                occurrences += counts
        candidates.append(
            dict(
                candidate,
                train_occurrences=occurrences,
                train_conversations=len(conversations),
                accepted=kind == "expression" and len(conversations) >= min_conversations,
            )
        )
    value = {
        "version": 1,
        "state": "complete" if not failed and not invalid else "incomplete",
        "person_ids": sorted({r["person_id"] for r in records}),
        "train_digest": digest(records),
        "train_conversation_ids": sorted({r["conversation_id"] for r in records}),
        "train_target_hashes": sorted(target_fingerprints(records)),
        "min_conversations": min_conversations,
        "discovery_prompt_digest": digest(DISCOVERY_PROMPT),
        "model": judge.model,
        "batches": len(batches),
        "failed_batches": failed,
        "invalid_candidates": invalid,
        "candidates": candidates,
    }
    value["artifact_digest"] = digest(value)
    return value


def validate_frozen(frozen, records):
    value = {k: v for k, v in frozen.items() if k != "artifact_digest"}
    if digest(value) != frozen["artifact_digest"]:
        raise ValueError("Frozen expression artifact was modified")
    if frozen["state"] != "complete":
        raise ValueError("Discovery incomplete; retry failed batches before extraction")
    if set(frozen["person_ids"]) != {r["person_id"] for r in records}:
        raise ValueError("Personal expression owner mismatch")


def expression_stats(records, frozen):
    result = {}
    for item in ("004,006",):
        candidates = sorted(
            {
                c["expression"]
                for c in frozen["candidates"]
                if c["item"] in (item, "004", "006") and c["accepted"]
            }
        )
        counts = Counter(dict.fromkeys(candidates, 0))
        message_counts = Counter(dict.fromkeys(candidates, 0))
        window_hits = hit_messages = 0
        messages = [m for r in records for m in r["messages"] if m["text"].strip()]
        chars = sum(len(m["text"]) for r in records for m in r["messages"])
        hit_keys, expression_keys = set(), {expression: set() for expression in candidates}
        for row in records:
            found = False
            for message in row["messages"]:
                hit = False
                key = (row["sample_id"], message["message_id"])
                for expression in candidates:
                    count = len(list(re.finditer("(?=" + re.escape(expression) + ")", message["text"])))
                    counts[expression] += count
                    message_counts[expression] += count > 0
                    hit = hit or count > 0
                    if count:
                        expression_keys[expression].add(key)
                hit_messages += hit
                if hit:
                    hit_keys.add(key)
                found = found or hit
            window_hits += found
        texts = message_texts(records)
        result[item] = {
            "n_candidates": len(candidates),
            "counts": counts,
            "message_counts": message_counts,
            "nonempty_messages": len(messages),
            "hit_messages": hit_messages,
            "message_occurrence_rate": ratio(hit_messages, len(messages)),
            "message_rates": {k: ratio(v, len(messages)) for k, v in message_counts.items()},
            "by_length": {
                "message_occurrence_rate": by_length(texts, hit_keys),
                "message_rates": {k: by_length(texts, keys) for k, keys in expression_keys.items()},
            },
            "reliability": {
                "message_occurrence_rate": reliability(texts, hit_keys),
                "message_rates": {k: reliability(texts, keys) for k, keys in expression_keys.items()},
            },
            "n_windows": len(records),
            "window_hits": window_hits,
            "window_occurrence_rate": ratio(window_hits, len(records)),
            "occurrences_per_1000_chars": 1000 * sum(counts.values()) / chars if chars else None,
            "state": "ok" if candidates else "na",
            "reason": "" if candidates else "No recurring candidates",
        }
    return result


def summarize_semantic(records, annotations, topics, parsed):
    result = {}
    rows = {r["sample_id"]: r for r in records}
    for item in ITEMS:
        states = Counter()
        labels, expressions, attributes = Counter(), Counter(), Counter()
        eligible_chars = eligible_tokens = events_n = eligible_messages = 0
        hits, texts = set(), {}
        mixed_messages = switches = language_messages = 0
        gaps, seconds, pairs, same_turn = [], [], 0, 0
        confirmed = uncertain = 0
        by_topic = {}
        for annotation in annotations:
            row = rows[annotation["sample_id"]]
            value = annotation["items"][item]
            states[value["state"]] += 1
            if value["state"] not in ("present", "absent"):
                continue
            nonempty = sum(bool(m["text"].strip()) for m in row["messages"])
            hit_ids = {e["evidence"]["message_id"] for e in value["events"]}
            eligible_messages += nonempty
            hits.update((row["sample_id"], message_id) for message_id in hit_ids)
            texts.update(message_texts([row]))
            if item == "011":
                language_messages += nonempty
            if item == "015":
                pairs += adjacent_sends(row)
            chars = sum(len(m["text"]) for m in row["messages"])
            nwords = sum(
                sum(
                    word_token(t, s)
                    for t, s in zip(parsed[m["text"]]["cws"], parsed[m["text"]]["pos"], strict=True)
                )
                for m in row["messages"]
            )
            eligible_chars += chars
            eligible_tokens += nwords
            topic = topics[row["sample_id"]]["label"]
            if topic not in by_topic:
                by_topic[topic] = {
                    "windows": 0,
                    "present_windows": 0,
                    "messages": 0,
                    "hit_messages": 0,
                    "word_tokens": 0,
                    "events": 0,
                    "labels": Counter(),
                    "expressions": Counter(),
                    "attributes": Counter(),
                }
            group = by_topic[topic]
            group["windows"] += 1
            group["present_windows"] += value["state"] == "present"
            group["messages"] += nonempty
            group["hit_messages"] += len(hit_ids)
            group["word_tokens"] += nwords
            for event in value["events"]:
                events_n += 1
                label = event["label"]
                quote = event["evidence"]["quote"]
                labels[label] += 1
                expressions[quote] += 1
                group["events"] += 1
                group["labels"][label] += 1
                group["expressions"][quote] += 1
                attributes.update(f"{k}={v}" for k, v in event["attributes"].items() if v is not None)
                group["attributes"].update(
                    f"{k}={v}" for k, v in event["attributes"].items() if v is not None
                )
                if item == "015":
                    messages = {m["message_id"]: m for m in row["messages"]}
                    src = messages[event["source"]["message_id"]]
                    dst = messages[event["evidence"]["message_id"]]
                    gaps.append(1)
                    same_turn += src is dst
                    if src.get("timestamp") is not None and dst.get("timestamp") is not None:
                        dt = (
                            datetime.fromisoformat(dst["timestamp"])
                            - datetime.fromisoformat(src["timestamp"])
                        ).total_seconds()
                        if dt < 0:
                            raise ValueError("Correction timestamps out of order")
                        seconds.append(dt)
                    confirmed += label == "confirmed_error"
                    uncertain += label == "uncertain_prior_error"
            if item == "011":
                for msg in row["messages"]:
                    es = sorted(
                        (
                            e["evidence"]
                            for e in value["events"]
                            if e["evidence"]["message_id"] == msg["message_id"]
                        ),
                        key=lambda e: e["start"],
                    )
                    if es:
                        # Reset at excluded gaps, so code/link removal cannot invent transitions.
                        parts = []
                        for e in es:
                            if parts and not msg["text"][parts[-1]["end"] : e["start"]].strip():
                                parts[-1]["text"] += msg["text"][parts[-1]["end"] : e["start"]] + e["quote"]
                                parts[-1]["end"] = e["end"]
                            else:
                                parts.append({"text": e["quote"], "end": e["end"]})
                        stats = [script_stats(p["text"]) for p in parts]
                        mixed_messages += len({s["script"] for stat in stats for s in stat["spans"]}) == 2
                        switches += sum(s["switches"] for s in stats)
        n = len(annotations)
        valid = states["present"] + states["absent"]
        unknown, failure = missing_messages(
            (rows[a["sample_id"]], a["items"][item]["state"]) for a in annotations
        )
        result[item] = {
            "states": {s: states[s] for s in ("present", "absent", "unknown", "na", "failure")},
            "protocol": PROTOCOL_VERSION,
            "valid_messages": eligible_messages,
            "hit_messages": len(hits),
            "message_occurrence_rate": ratio(len(hits), eligible_messages),
            "by_length": {"message_occurrence_rate": by_length(texts, hits)},
            "reliability": {
                "message_occurrence_rate": with_missing(reliability(texts, hits), unknown, failure)
            },
            "n_windows": n,
            "valid_windows": valid,
            "coverage": ratio(valid, n),
            "unknown_rate": ratio(states["unknown"], n),
            "failure_rate": ratio(states["failure"], n),
            "window_occurrence_rate": ratio(states["present"], valid),
            "events": events_n,
            "eligible_characters": eligible_chars,
            "eligible_word_tokens": eligible_tokens,
            "events_per_100_words": 100 * events_n / eligible_tokens if eligible_tokens else None,
            "labels": labels,
            "expressions": expressions,
            "attributes": attributes,
            "by_topic": by_topic,
            "label_events_per_100_words": {
                k: 100 * v / eligible_tokens if eligible_tokens else None for k, v in labels.items()
            },
        }
        if item == "011":
            result[item].update(
                denominator="valid nonempty target messages, including monolingual messages",
                language_messages=language_messages,
                mixed_messages=mixed_messages,
                mixed_rate=ratio(mixed_messages, language_messages),
                switches=switches,
                switches_per_language_message=ratio(switches, language_messages),
            )
        if item == "015":
            result[item].update(
                message_gaps=gaps,
                adjacent_sends=pairs,
                corrections_per_adjacent_send=ratio(len(gaps), pairs),
                same_turn_corrections=same_turn,
                time_gaps_seconds=seconds,
                n_timed=len(seconds),
                confirmed_error_corrections=confirmed,
                uncertain_prior_error_corrections=uncertain,
            )
    return result


def extract(
    corpora,
    parsed,
    judge,
    output,
    workers,
    qa_runner=None,
    block_size=DEFAULT_BLOCK_SIZE,
    frozen=None,
):
    topics, annotations, semantic = {}, {}, {}
    if judge:
        records = list({r["sample_id"]: r for rows in corpora.values() for r in rows}.values())
        print("Classifying shared visible contexts", flush=True)
        queued = queue_order(records, lambda r: r["sample_id"])
        values = parallel_map(
            lambda r: classify_topic(r["context"], judge.for_record(r)), queued, workers, judge=judge
        )
        found = {r["sample_id"]: v for r, v in zip(queued, values, strict=True)}
        topics = {r["sample_id"]: found[r["sample_id"]] for r in records}
        write_json(output / "topics.json", topics)
        print(f"Annotating {distinct_records(corpora)} windows with shared worker pool", flush=True)
        annotations = annotate_corpora(
            corpora, lambda row: annotate(row, parsed, judge), workers, judge=judge
        )
    for name, rows in corpora.items():
        dest = output / name
        dest.mkdir()
        summary = summarize(rows, parsed, block_size)
        write_json(dest / "lexical.json", {k: v for k, v in summary.items() if k != "_tokens"})
        write_json(dest / "records.json", rows)
        if frozen:
            write_json(dest / "personal.json", expression_stats(rows, frozen))
        if judge:
            write_json(dest / "annotations.json", annotations[name])
            semantic[name] = summarize_semantic(rows, annotations[name], topics, parsed)
            write_json(dest / "semantic.json", semantic[name])
            if qa_runner and name == PRIMARY_CORPUS:
                write_json(dest / "qa.json", qa_runner(rows, annotations[name]))
    return semantic


def extraction_coverage(semantic, frozen, judge):
    coverage = {f"{i:03}": {"state": "implemented"} for i in range(1, 25)}
    for i in ("004", "006"):
        coverage.pop(i)
    for i in ("005", "013", "014"):
        coverage[i] = {"state": "deferred", "reason": "High difficulty"}
    for i in ("004,006",):
        coverage[i] = {
            "state": "extracted" if frozen else "not_run",
            "reason": "" if frozen else "Requires a discovered expression artifact (--frozen)",
        }
    for i, reason in {
        "003": "Content-word semantic classification disabled",
        "019": "Omitted; overall register is measured by 233",
    }.items():
        coverage[i] = {"state": "not_run", "reason": reason}
    for i in ITEMS:
        coverage[i] = {
            "state": "extracted" if judge else "not_run",
            "corpora": {name: value[i]["states"] for name, value in semantic.items()},
        }
    return coverage
