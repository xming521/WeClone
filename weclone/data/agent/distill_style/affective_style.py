"""Observable affect, independently conditioned happiness and closed-class degree modifiers."""

from weclone.prompts.distill_style import (
    AFFECTIVE_STYLE_CONTEXT_PROMPT as CONTEXT_PROMPT,
    AFFECTIVE_STYLE_ANNOTATION_PROMPT as ANNOTATION_PROMPT,
)

from collections import Counter
from copy import copy

from .data import PRIMARY_CORPUS, digest, write_json
from .extraction import annotate_corpora, distinct_records, locate, parallel_map, queue_order
from .lexical_choice import word_token
from .orthographic_style import punctuation_events
from . import paralinguistic_style
from .pragmatic_style import labels_summary
from .stats import length_strata, missing_entries, nonspace_length, ratio, window_reliability, with_missing

PROTOCOL_VERSION = "affective-style-v3"
SCOPE = "Shared affective extraction: 151,152,153+160,155,158,159,161,163"
DEFAULT_FREQUENCY_SCALE = 100
MEASURES = ("151", "152", "153,160", "155", "158", "159", "161", "163")
DEFERRED = ("154", "156", "157", "162")
ITEMS = {
    "151": "情感词",
    "152": "表达强度",
    "153,160": "情绪表达方式",
    "158": "抱怨",
    "159": "吐槽",
    "161": "表达基调",
    "163": "程度修饰的情绪用途",
}
TONES = ("positive", "negative", "mixed", "calm", "unmarked", "unknown")
MODES = ("direct", "symbolic", "implicit")


# Degree adverbs (before the head, dep ADV) and degree complements (after it, dep CMP) form closed
# classes; SRL ARGM-DGR never fires on chat text. "fixed" members always grade the head, "ambiguous"
# ones also have frequency, manner, focus, quantity or result uses that the annotation resolves.
DEGREE_ADVERBS = {
    "fixed": frozenset(
        "很 挺 太 非常 十分 特 相当 极 极其 极为 极度 格外 分外 异常 尤为 超 超级 巨 贼 蛮 颇 忒 "
        "最 最为 更 更加 越 越发 愈 愈发 愈加 比较 较 较为 稍 稍微 稍稍 略 略微 有点 有点儿 有些 "
        "过于 太过 多么".split()
    ),
    "ambiguous": frozenset("好 真 可 老 多 怪 够 还 这么 那么 这样 那样 如此 实在 简直 狂 特别 尤其".split()),
}
DEGREE_COMPLEMENTS = {
    "fixed": frozenset("很 极 不行 要命 要死".split()),
    "ambiguous": frozenset("死 坏 透 多 一点 一点儿 些".split()),
}
# These grade only adjectives; before verbs they mean "still" or "in this way".
ADJECTIVE_ONLY = frozenset("还 这样 那样 如此".split())


def degree_source(word, relation, head_pos):
    table = DEGREE_ADVERBS if relation == "ADV" else DEGREE_COMPLEMENTS if relation == "CMP" else {}
    if relation == "ADV" and word in ADJECTIVE_ONLY and head_pos != "a":
        return None
    return next((source for source, forms in table.items() if word in forms), None)


def degree_events(msg, parsed):
    """Closed-class degree adverbs/complements attached to a valid head; fixed members are degree by rule."""
    p = parsed[msg["text"]]
    candidates = []
    for i, word in enumerate(p["cws"]):
        head, relation = p["dep"]["head"][i] - 1, p["dep"]["label"][i]
        if not 0 <= head < len(p["cws"]) or head == i or p["pos"][head] == "wp":
            continue
        source = degree_source(word, relation, p["pos"][head])
        if source is None:
            continue
        candidates.append(
            {
                "anchor_id": f"{msg['message_id']}:dgr:{i}:{head}",
                "message_id": msg["message_id"],
                "token_index": i,
                "start": p["starts"][i],
                "end": p["ends"][i],
                "quote": msg["text"][p["starts"][i] : p["ends"][i]],
                "relation": relation,
                "source": source,
                "target": {
                    "token_index": head,
                    "start": p["starts"][head],
                    "end": p["ends"][head],
                    "quote": msg["text"][p["starts"][head] : p["ends"][head]],
                },
            }
        )
    return {"events": [c for c in candidates if c["source"] == "fixed"], "candidates": candidates}


def observations(row, parsed):
    result = []
    for msg in row["messages"]:
        p = parsed[msg["text"]]
        emoji, unknown, labels = paralinguistic_style.emoji_events(msg)
        result.append(
            {
                "sample_id": row["sample_id"],
                "message_id": msg["message_id"],
                "characters": len(msg["text"]),
                "nonempty": bool(msg["text"].strip()),
                "words": sum(word_token(w, t) for w, t in zip(p["cws"], p["pos"], strict=True)),
                "exclamations": [e for e in punctuation_events(msg["text"]) if e["label"] == "exclamation"],
                "emoji": emoji,
                "unknown_emoji": unknown,
                "unverified_labels": labels,
                "degree": degree_events(msg, parsed),
            }
        )
    return result


def validate_condition(value, row):
    if value["happiness"] not in {"happy", "not_happy", "unknown"}:
        raise ValueError("Invalid independent condition")
    if not isinstance(value["evidence"], list):
        raise ValueError("Condition evidence must be a list")
    evidence = [locate(e, [m for m in row["context"] if m["role"] == "assistant"]) for e in value["evidence"]]
    if bool(evidence) != (value["happiness"] != "unknown"):
        raise ValueError("Known condition requires pre-response speaker evidence")
    return dict(value, evidence=evidence)


def condition(row, judge):
    judge = judge.for_record(row) if judge else None
    supplied = row.get("affect_scenario")
    if supplied is not None:
        if (
            not isinstance(supplied, dict)
            or supplied.get("source") != "independent_scenario"
            or supplied.get("happiness") not in {"happy", "not_happy", "unknown"}
        ):
            raise ValueError("affect_scenario must declare independent_scenario and happiness")
        if any(
            not isinstance(supplied.get(k), str) or not supplied[k].strip()
            for k in ("scene_id", "description")
        ):
            raise ValueError("Independent scenario needs scene_id and description")
        return dict(
            supplied,
            state="known" if supplied["happiness"] != "unknown" else "unknown",
            evidence=[],
            cache_key=None,
        )
    scene = digest(row["context"]) if row["context"] else None
    base = {"source": "pre_response_context", "scene_id": scene}
    if not judge or not any(m["role"] == "assistant" and m["text"].strip() for m in row["context"]):
        return dict(
            base,
            state="unknown",
            happiness="unknown",
            evidence=[],
            reason="No independent speaker state evidence or context judge disabled",
            cache_key=None,
        )
    payload = {"context": row["context"]}
    first = judge.call(CONTEXT_PROMPT, payload)
    error, repair_key = None, None
    for attempt in range(2):
        response = first if attempt == 0 else judge.call(CONTEXT_PROMPT, payload, attempt=attempt)
        if attempt:
            repair_key = response["cache_key"]
        try:
            value = validate_condition(response.get("data", {}), row)
            return dict(
                base,
                **value,
                state="known" if value["happiness"] != "unknown" else "unknown",
                cache_key=first["cache_key"],
                repair_cache_key=repair_key,
            )
        except (KeyError, TypeError, ValueError) as exc:
            error = str(exc)
    judge.invalidate(first["cache_key"], repair_key)
    return dict(
        base,
        state="failure",
        happiness="unknown",
        evidence=[],
        reason=error,
        cache_key=first["cache_key"],
        repair_cache_key=repair_key,
    )


def validate_item(key, values, row, anchors):
    if not isinstance(values, list):
        raise ValueError(
            f"items[{key!r}] must be an array of message objects, received {type(values).__name__}"
        )
    if len(values) != len(row["messages"]):
        raise ValueError(
            f"items[{key!r}] must contain each target message once: {[m['message_id'] for m in row['messages']]}"
        )
    mapping = {v["message_id"]: v for v in values}
    if len(mapping) != len(values) or set(mapping) != {m["message_id"] for m in row["messages"]}:
        raise ValueError("Duplicate or unexpected message")
    result = []
    for msg in row["messages"]:
        value = mapping[msg["message_id"]]
        state, events = value["state"], value["events"]
        candidates = {a["anchor_id"]: a for a in anchors if a["message_id"] == msg["message_id"]}
        allowed = (
            {"na"}
            if not msg["text"].strip() or (key == "163" and not candidates)
            else {"present"}
            if key == "163"
            else {"present", "absent", "unknown"}
        )
        if state not in allowed or not isinstance(events, list) or (state == "present") != bool(events):
            raise ValueError("State, events or applicability mismatch")
        checked, seen = [], set()
        for event in events:
            e = dict(event, evidence=locate(event["evidence"], [msg]))
            span = (e["evidence"]["start"], e["evidence"]["end"])
            if key != "163" and any(span[0] < end and span[1] > start for start, end in seen):
                raise ValueError("Overlapping same-item evidence")
            if key == "151" and e["polarity"] not in {"positive", "negative"}:
                raise ValueError("Invalid sentiment polarity")
            if key == "151" and not word_token(e["evidence"]["quote"], "n"):
                raise ValueError("Sentiment words require lexical characters, not a standalone symbol")
            if key == "153,160" and (
                not isinstance(e["modes"], list)
                or not e["modes"]
                or len(set(e["modes"])) != len(e["modes"])
                or not set(e["modes"]) <= set(MODES)
            ):
                raise ValueError("Invalid expression modes")
            if key == "163":
                aid = e["anchor_id"]
                if aid not in candidates or aid in seen or e["degree"] not in {"yes", "no", "unknown"}:
                    raise ValueError("Invalid degree anchor annotation")
                emotion_labels = (
                    {"na"}
                    if e["degree"] == "no"
                    else {"unknown"}
                    if e["degree"] == "unknown"
                    else {"yes", "no", "unknown"}
                )
                if e["emotion"] not in emotion_labels or (
                    candidates[aid]["source"] == "fixed" and e["degree"] != "yes"
                ):
                    raise ValueError("Degree and emotional-use labels conflict")
                target = candidates[aid]["target"]
                if not (span[0] < target["end"] and span[1] > target["start"]):
                    raise ValueError("Degree use must locate the modified target")
                seen.add(aid)
            else:
                seen.add(span)
            checked.append(e)
        if key == "163" and seen != set(candidates):
            raise ValueError("Every degree candidate needs one annotation")
        if key == "152":
            intensity = value["intensity"]
            if (
                (state == "present" and (type(intensity) is not int or intensity not in (1, 2, 3)))
                or (state == "absent" and (type(intensity) is not int or intensity != 0))
                or (state in {"na", "unknown"} and intensity is not None)
            ):
                raise ValueError("Invalid intensity/state relation")
        if key == "161":
            label = value["tone"]
            if (
                label not in TONES
                or (state == "present" and label not in TONES[:4])
                or (state == "absent" and label != "unmarked")
                or (state in {"unknown", "na"} and label != "unknown")
            ):
                raise ValueError("Invalid tone/state relation")
        result.append(dict(value, events=checked))
    return result


def validate_response(value, row, anchors):
    valid, failures = {}, {}
    items = value.get("items", {}) if isinstance(value, dict) else {}
    for key in ITEMS:
        try:
            valid[key] = validate_item(key, items[key], row, anchors)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            failures[key] = str(exc)
    for a, b in (("152", "153,160"), ("152", "161"), ("151", "152"), ("151", "153,160")):
        if a not in valid or b not in valid:
            continue
        for x, y in zip(valid[a], valid[b], strict=True):
            conflict = (a == "152" and {x["state"], y["state"]} == {"present", "absent"}) or (
                a == "151" and x["state"] == "present" and y["state"] == "absent"
            )
            if conflict:
                failures[a] = failures[b] = f"Inconsistent observed affect between {a} and {b}"
    return {k: v for k, v in valid.items() if k not in failures}, failures


def annotate(row, parsed, judge, refresh=False):
    judge = judge.for_record(row)
    anchors = [e for o in observations(row, parsed) for e in o["degree"]["candidates"]]
    payload = {"context": row["context"], "messages": row["messages"], "degree_anchors": anchors}
    first = judge.call(ANNOTATION_PROMPT, payload, refresh=refresh)
    valid, failures = validate_response(first.get("data", {}), row, anchors)
    initial, origins, repair_key = dict(failures), dict.fromkeys(valid, first["cache_key"]), None
    if failures:
        repair = judge.call(
            ANNOTATION_PROMPT,
            dict(payload, failures=failures),
            refresh=refresh,
            attempt=1,
        )
        repair_key = repair["cache_key"]
        data = repair.get("data", {})
        replacements = data.get("items", {}) if isinstance(data, dict) else {}
        merged = dict(valid)
        if isinstance(replacements, dict):
            merged.update({k: replacements[k] for k in failures if k in replacements})
        valid, failures = validate_response({"items": merged}, row, anchors)
        for key in valid:
            origins.setdefault(key, repair_key)
    if failures:
        judge.invalidate(first["cache_key"], repair_key)
    for key, error in failures.items():
        valid[key] = [
            dict(
                message_id=m["message_id"],
                state="failure",
                events=[],
                reason=error,
                **({"intensity": None} if key == "152" else {"tone": "unknown"} if key == "161" else {}),
            )
            for m in row["messages"]
        ]
    return {
        "sample_id": row["sample_id"],
        "items": valid,
        "cache_key": first["cache_key"],
        "repair_cache_key": repair_key,
        "item_cache_keys": origins,
        "initial_failures": initial,
        "remaining_failures": failures,
    }


def summarize(records, parsed):
    rows = [o for row in records for o in observations(row, parsed)]
    degree = [e for o in rows for e in o["degree"]["events"]]
    candidates = [e for o in rows for e in o["degree"]["candidates"]]
    words = sum(o["words"] for o in rows)
    return {
        "n_windows": len(records),
        "n_messages": len(rows),
        "messages": rows,
        "163": {
            "fixed_events": len(degree),
            "forms": Counter(e["quote"] for e in degree),
            "candidate_count": len(candidates),
            "candidate_forms": {
                source: Counter(c["quote"] for c in candidates if c["source"] == source)
                for source in ("fixed", "ambiguous")
            },
            "relations": Counter(c["relation"] for c in candidates),
            "fixed_candidate_fraction": ratio(len(degree), len(candidates)),
            "word_denominator": words,
            "fixed_per_100_words": DEFAULT_FREQUENCY_SCALE * len(degree) / words if words else None,
            "interpretation": "fixed members are a lower bound; ambiguous members require contextual degree disambiguation",
        },
    }


def message_entries(records, annotations, raw, key):
    sizes = {(o["sample_id"], o["message_id"]): o for o in raw["messages"]}
    result = []
    for row, a in zip(records, annotations, strict=True):
        if a["sample_id"] != row["sample_id"]:
            raise ValueError("Annotations and records must align")
        texts = {m["message_id"]: m["text"] for m in row["messages"]}
        for m in a["items"][key]:
            result.append(
                dict(
                    m,
                    sample_id=row["sample_id"],
                    observation=sizes[(row["sample_id"], m["message_id"])],
                    nonspace_characters=nonspace_length(texts[m["message_id"]]),
                )
            )
    return result


def message_labels(entries, field, stringify=False):
    counts = Counter(
        str(m[field]) if stringify else m[field] for m in entries if m["state"] in {"present", "absent"}
    )
    return {
        "counts": counts,
        "denominator": sum(counts.values()),
        "proportions": {k: ratio(v, sum(counts.values())) for k, v in counts.items()},
    }


def window_occurrence(annotations, key):
    counts = Counter()
    for a in annotations:
        states = [m["state"] for m in a["items"][key]]
        label = (
            "failure"
            if "failure" in states
            else "unknown"
            if "unknown" in states
            else "na"
            if all(s == "na" for s in states)
            else "present"
            if "present" in states
            else "absent"
        )
        counts[label] += 1
    known = counts["present"] + counts["absent"]
    return {
        "states": counts,
        "window_denominator": known,
        "window_occurrence_rate": ratio(counts["present"], known),
        "coverage": ratio(known, len(annotations)),
    }


def sentiment_words(entries, records, parsed):
    texts = {(row["sample_id"], m["message_id"]): m["text"] for row in records for m in row["messages"]}
    counts, aligned = Counter(), []
    for m in entries:
        if m["state"] not in {"present", "absent"}:
            continue
        p = parsed[texts[(m["sample_id"], m["message_id"])]]
        for i, (word, tag) in enumerate(zip(p["cws"], p["pos"], strict=True)):
            if not word_token(word, tag):
                continue
            labels = {
                e["polarity"]
                for e in m["events"]
                if p["starts"][i] < e["evidence"]["end"] and p["ends"][i] > e["evidence"]["start"]
            }
            if labels:
                label = next(iter(labels)) if len(labels) == 1 else "mixed_alignment"
                counts[label] += 1
                aligned.append(
                    {
                        "sample_id": m["sample_id"],
                        "message_id": m["message_id"],
                        "token_index": i,
                        "word": word,
                        "polarity": label,
                    }
                )
    known_words = sum(m["observation"]["words"] for m in entries if m["state"] in {"present", "absent"})
    return {
        "counts": counts,
        "affective_word_count": sum(counts.values()),
        "word_denominator": known_words,
        "fraction_of_all_words": {k: ratio(v, known_words) for k, v in counts.items()},
        "polarity_share": {k: ratio(v, sum(counts.values())) for k, v in counts.items()},
        "aligned_tokens": aligned,
    }


def degree_summary(entries):
    candidates = {
        e["anchor_id"] + ":" + m["sample_id"]: e
        for m in entries
        for e in m["observation"]["degree"]["candidates"]
    }
    decisions, uses, forms, sources = Counter(), Counter(), Counter(), Counter()
    eligible_words = eligible_messages = positive_messages = 0
    events, eligible = [], []
    missing = {"unknown": 0, "failure": 0}
    for m in entries:
        for e in m["events"]:
            decisions[e["degree"]] += 1
        complete = m["state"] in {"present", "na"} and all(e["degree"] != "unknown" for e in m["events"])
        if not complete:
            if m["observation"]["nonempty"]:
                missing["failure" if m["state"] == "failure" else "unknown"] += 1
            continue
        if m["observation"]["nonempty"]:
            eligible_messages += 1
            eligible.append(
                (m["sample_id"], m["nonspace_characters"], any(e["degree"] == "yes" for e in m["events"]))
            )
        eligible_words += m["observation"]["words"]
        positive_messages += any(e["degree"] == "yes" for e in m["events"])
        for e in m["events"]:
            if e["degree"] != "yes":
                continue
            raw = candidates[e["anchor_id"] + ":" + m["sample_id"]]
            forms[raw["quote"]] += 1
            sources[raw["source"]] += 1
            uses[e["emotion"]] += 1
            events.append(
                dict(raw, sample_id=m["sample_id"], emotion=e["emotion"], use_evidence=e["evidence"])
            )
    known_use = uses["yes"] + uses["no"]
    return {
        "degree_decisions": decisions,
        "candidate_count": len(candidates),
        "annotated_candidate_coverage": ratio(sum(decisions.values()), len(candidates)),
        "determinate_candidate_coverage": ratio(decisions["yes"] + decisions["no"], len(candidates)),
        "unknown_fraction": ratio(decisions["unknown"], len(candidates)),
        "forms": forms,
        "sources": sources,
        "event_count": sum(forms.values()),
        "word_denominator": eligible_words,
        "message_denominator": eligible_messages,
        "message_rate": ratio(positive_messages, eligible_messages),
        "by_length": {"message_rate": length_strata((n, hit) for _, n, hit in eligible)},
        "reliability": {
            "message_rate": with_missing(
                window_reliability((w, hit) for w, _, hit in eligible), missing["unknown"], missing["failure"]
            )
        },
        "per_100_words": DEFAULT_FREQUENCY_SCALE * sum(forms.values()) / eligible_words
        if eligible_words
        else None,
        "emotion_uses": uses,
        "emotion_use_denominator": known_use,
        "emotional_use_rate": ratio(uses["yes"], known_use),
        "events": events,
    }


def summarize_semantic(records, annotations, parsed, raw, conditions):
    result = {}
    for key in ITEMS:
        entries = message_entries(records, annotations, raw, key)
        states = Counter(m["state"] for m in entries)
        valid = [m for m in entries if m["state"] in {"present", "absent"}]
        events = [e for m in valid for e in m["events"]]
        chars = sum(m["observation"]["characters"] for m in valid)
        value = {
            "states": states,
            "n_messages": len(entries),
            "valid_messages": len(valid),
            "coverage": ratio(states["present"] + states["absent"] + states["na"], len(entries)),
            "unknown_rate": ratio(states["unknown"], len(entries)),
            "failure_rate": ratio(states["failure"], len(entries)),
            "message_rate": ratio(states["present"], len(valid)),
            "by_length": {
                "message_rate": length_strata(
                    (m["nonspace_characters"], m["state"] == "present") for m in valid
                )
            },
            "reliability": {
                "message_rate": with_missing(
                    window_reliability((m["sample_id"], m["state"] == "present") for m in valid),
                    *missing_entries(entries),
                )
            },
            "event_count": len(events),
            "character_denominator": chars,
            "per_100_characters": DEFAULT_FREQUENCY_SCALE * len(events) / chars if chars else None,
            "forms": Counter(e["evidence"]["quote"] for e in events),
            "initial_failure_windows": sum(key in a["initial_failures"] for a in annotations),
            "remaining_failure_windows": sum(key in a["remaining_failures"] for a in annotations),
        }
        if key == "151":
            value["span_polarity"] = labels_summary(entries, lambda e: [e["polarity"]])
            value["word_polarity"] = sentiment_words(entries, records, parsed)
        elif key == "152":
            value["intensity"] = message_labels(entries, "intensity", True)
        elif key == "153,160":
            value["modes"] = labels_summary(entries, lambda e: e["modes"])
        elif key in {"158", "159"}:
            value["windows"] = window_occurrence(annotations, key)
        elif key == "161":
            value["tone"] = message_labels(entries, "tone")
            groups = {}
            missing = 0
            for m in valid:
                scene = conditions[m["sample_id"]]["scene_id"]
                if scene is None:
                    missing += 1
                else:
                    groups.setdefault(scene, Counter())[m["tone"]] += 1
            value["context_groups"] = groups
            value["unmatched_context_messages"] = missing
        elif key == "163":
            # Candidate presence is not degree-modifier occurrence.
            for field in (
                "message_rate",
                "by_length",
                "reliability",
                "event_count",
                "forms",
                "per_100_characters",
                "character_denominator",
            ):
                value.pop(field)
            value.update(degree_summary(entries))
        result[key] = value
    return result


def happiness_summary(records, raw, conditions, laughter):
    happy = {row["sample_id"] for row in records if conditions[row["sample_id"]]["happiness"] == "happy"}
    observed = [m for m in raw["messages"] if m["sample_id"] in happy and m["nonempty"]]
    texts = {(row["sample_id"], m["message_id"]): m["text"] for row in records for m in row["messages"]}

    def length(m):
        return nonspace_length(texts[(m["sample_id"], m["message_id"])])

    chars = sum(m["characters"] for m in observed)
    result = {
        "condition_counts": Counter(conditions[row["sample_id"]]["happiness"] for row in records),
        "condition_failure_count": sum(conditions[row["sample_id"]]["state"] == "failure" for row in records),
        "eligible_windows": len(happy),
        "eligible_messages": len(observed),
        "character_denominator": chars,
        "state": "extracted" if observed else "unavailable",
        "reason": None if observed else "No independently evidenced happy condition",
    }
    for key in ("exclamations", "emoji"):
        n = sum(len(m[key]) for m in observed)
        result[key] = {
            "events": n,
            "message_rate": ratio(sum(bool(m[key]) for m in observed), len(observed)),
            "by_length": {"message_rate": length_strata((length(m), bool(m[key])) for m in observed)},
            "reliability": {
                "message_rate": window_reliability((m["sample_id"], bool(m[key])) for m in observed)
            },
            "per_100_characters": DEFAULT_FREQUENCY_SCALE * n / chars if chars else None,
        }
    known, events = set(), []
    for sid in happy:
        text = laughter.get(sid, {}).get("text", {})
        if text.get("state") not in {"present", "absent"}:
            continue
        es = text["events"]
        if any(
            e["paralinguistic"] == "unknown" or (e["paralinguistic"] == "yes" and e["sound"] == "unknown")
            for e in es
        ):
            continue
        known.add(sid)
        events.extend(
            dict(e, sample_id=sid) for e in es if e["paralinguistic"] == "yes" and e["sound"] == "laughter"
        )
    eligible = [m for m in observed if m["sample_id"] in known]
    laughter_states = {sid: laughter.get(sid, {}).get("text", {}).get("state") for sid in happy}
    # Happy windows whose laughter is unknown or failed, not those where it does not apply.
    pending = [m for m in observed if m["sample_id"] not in known and laughter_states[m["sample_id"]] != "na"]
    pending_failed = sum(laughter_states[m["sample_id"]] in {None, "failure"} for m in pending)
    laugh_chars = sum(m["characters"] for m in eligible)
    hits = {(e["sample_id"], e["message_id"]) for e in events}
    result["laughter"] = {
        "events": len(events),
        "valid_windows": len(known),
        "coverage": ratio(len(known), len(happy)),
        "message_denominator": len(eligible),
        "message_rate": ratio(len(hits), len(eligible)),
        "by_length": {
            "message_rate": length_strata(
                (length(m), (m["sample_id"], m["message_id"]) in hits) for m in eligible
            )
        },
        "reliability": {
            "message_rate": with_missing(
                window_reliability(
                    (m["sample_id"], (m["sample_id"], m["message_id"]) in hits) for m in eligible
                ),
                len(pending) - pending_failed,
                pending_failed,
            )
        },
        "character_denominator": laugh_chars,
        "per_100_characters": DEFAULT_FREQUENCY_SCALE * len(events) / laugh_chars if laugh_chars else None,
        "forms": Counter(e["quote"] for e in events),
    }
    if observed and len(known) != len(happy):
        result["state"] = "partial"
    return result


def extract(corpora, parsed, judge, output, workers, qa_runner=None):
    rows = [row for records in corpora.values() for row in records]
    unique = {}
    for row in rows:
        unique.setdefault(digest({"context": row["context"], "scenario": row.get("affect_scenario")}), row)
    print(f"Resolving {len(unique)} independent pre-response conditions", flush=True)
    queued = queue_order(unique, lambda key: key)
    tagged = parallel_map(lambda key: condition(unique[key], judge), queued, workers, judge=judge)
    by_context = dict(zip(queued, tagged, strict=True))
    conditions = {
        row["sample_id"]: by_context[
            digest({"context": row["context"], "scenario": row.get("affect_scenario")})
        ]
        for row in rows
    }
    write_json(output / "conditions.json", conditions)
    raw, semantic, annotated, laughter = {}, {}, {}, {}
    for name, records in corpora.items():
        (output / name).mkdir()
        raw[name] = summarize(records, parsed)
        write_json(output / name / "affective_style.json", raw[name])
        write_json(output / name / "records.json", records)
    if judge:
        print(f"Annotating affective expression for {distinct_records(corpora)} windows", flush=True)
        annotated = annotate_corpora(corpora, lambda row: annotate(row, parsed, judge), workers, judge=judge)
        happy = {
            name: [row for row in records if conditions[row["sample_id"]]["happiness"] == "happy"]
            for name, records in corpora.items()
        }
        if any(happy.values()):
            reuse = copy(judge)
            reuse.protocol = paralinguistic_style.PROTOCOL_VERSION
            values = annotate_corpora(
                happy, lambda row: paralinguistic_style.annotate(row, parsed, reuse), workers, judge=judge
            )
            laughter = {
                name: {row["sample_id"]: value for row, value in zip(rows, values[name], strict=True)}
                for name, rows in happy.items()
                if rows
            }
        for name, records in corpora.items():
            semantic[name] = summarize_semantic(records, annotated[name], parsed, raw[name], conditions)
            write_json(output / name / "annotations.json", annotated[name])
            if qa_runner and name == PRIMARY_CORPUS:
                write_json(output / name / "qa.json", qa_runner(records, annotated[name]))
    for name, records in corpora.items():
        semantic.setdefault(name, {})["155"] = happiness_summary(
            records, raw[name], conditions, laughter.get(name, {})
        )
        write_json(output / name / "semantic.json", semantic[name])
        if laughter.get(name):
            write_json(output / name / "laughter_annotations.json", laughter[name])
    return semantic


def extraction_coverage(semantic, judge):
    coverage = {
        key: {
            "state": "extracted" if judge else "partial" if key == "163" else "not_run",
            "corpora": {name: values[key]["states"] for name, values in semantic.items()} if judge else {},
        }
        for key in ITEMS
    }
    available = [v["155"]["state"] for v in semantic.values()]
    coverage["155"] = {
        "state": "unavailable"
        if not available or all(s == "unavailable" for s in available)
        else "partial"
        if "partial" in available
        else "extracted",
        "corpora": {
            name: {k: values["155"][k] for k in ("state", "eligible_windows", "condition_counts")}
            for name, values in semantic.items()
        },
        "reason": "Requires independent happy condition; target reply never defines condition",
    }
    coverage.update({key: {"state": "deferred", "reason": "high difficulty"} for key in DEFERRED})
    return coverage


def protocol():
    return {
        "implemented": MEASURES,
        "deferred_high_difficulty": DEFERRED,
        "tones": TONES,
        "expression_modes": MODES,
        "intensity": {0: "unexpressed", 1: "mild", 2: "clear and strong", 3: "extreme expression"},
        "frequency_scale": DEFAULT_FREQUENCY_SCALE,
        "message_denominator": "present+absent; blank NA, unknown/failure excluded and reported",
        "word_alignment": "Any positive character overlap with LTP lexical token; unique token per message, competing polarities mixed_alignment",
        "scene_policy": "Replies grouped by identical visible context or declared independent scenario ID; no-context samples excluded from context groups",
        "happiness_condition": "Independent scenario metadata or context-only prior target-speaker statement; no target reply passed to condition classifier",
        "scenario_schema": {
            "affect_scenario": {
                "source": "independent_scenario",
                "scene_id": "shared scenario identifier",
                "happiness": "happy/not_happy/unknown",
                "description": "independent pre-response setup and provenance",
            }
        },
        "degree_rule": "closed degree class: adverb with dep=ADV or complement with dep=CMP on a valid non-punctuation head; fixed members are degree, ambiguous ones need degree-use disambiguation",
        "degree_forms": {
            "adverbs": {k: sorted(v) for k, v in DEGREE_ADVERBS.items()},
            "complements": {k: sorted(v) for k, v in DEGREE_COMPLEMENTS.items()},
            "adjective_head_only": sorted(ADJECTIVE_ONLY),
        },
        "degree_boundary": "SRL ARGM-DGR not used (never produced on chat text); ambiguous candidates never counted automatically; emotional use separate",
        "laughter_reuse": paralinguistic_style.PROTOCOL_VERSION,
        "emoji": paralinguistic_style.protocol()["emoji_data"],
        "labels": "Fixed functional categories, no sample-derived sentiment wordlists; degree adverbs are a closed grammatical class; observable expression not latent mental state",
    }
