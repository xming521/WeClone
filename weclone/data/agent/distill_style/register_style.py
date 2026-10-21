"""Register and sociolinguistic style, with reused internet-expression measures."""

from weclone.prompts.distill_style import (
    REGISTER_STYLE_ANNOTATION_PROMPT as ANNOTATION_PROMPT,
)

from collections import Counter
from copy import copy

from . import lexical_choice, rhetorical_devices
from .data import PRIMARY_CORPUS, digest, write_json
from .extraction import annotate_corpora, distinct_records, locate
from .stats import (
    WINDOW_LENGTH_EDGES,
    by_length,
    length_strata,
    message_texts,
    missing_messages,
    nonspace_length,
    ratio,
    reliability,
    with_missing,
)

PROTOCOL_VERSION = "register-style-v2"
DEFAULT_FREQUENCY_SCALE = 100
ITEMS = {"233": "书面与聊天语体", "234": "文言化用法", "245": "粗口用法"}
MEASURES = {**ITEMS, "244": "互联网表达分项统计"}
DEFERRED = ("237", "246")
SCOPE = "Register & Sociolinguistic Style: medium 233/234/244/245; high 237/246 deferred"
MODES = ("written", "spoken", "internet_chat")
REUSE = {
    "lexical": (lexical_choice, ("007", "010")),
    "rhetorical": (rhetorical_devices, ("178-179",)),
}
COMPONENTS = {
    "internet_usage": ("lexical", "007"),
    "abbreviation": ("lexical", "010"),
    "meme": ("rhetorical", "178-179"),
}


def validate_item(key, value, row):
    if not isinstance(value, dict):
        raise ValueError("Missing item")
    state, events = value.get("state"), value.get("events")
    if state not in ("present", "absent", "unknown", "na"):
        raise ValueError(
            f"Invalid state {state!r}; use present/absent/unknown/na. Register names belong in label."
        )
    if not isinstance(events, list):
        raise ValueError("events must be an array")
    if (state == "present") != bool(events):
        raise ValueError("State/events conflict")
    empty = not any(m["text"].strip() for m in row["messages"])
    if (state == "na") != empty:
        raise ValueError("NA only for blank reply")
    if key == "233" and state == "absent":
        raise ValueError("An unmarked grade is not absence")
    if key == "233":
        if (
            "label" not in value
            or (state == "present" and value["label"] not in (*MODES, "mixed"))
            or (state != "present" and value["label"] is not None)
        ):
            raise ValueError("Invalid register label")
    result = []
    seen = set()
    occupied = {}
    for event in events:
        e = locate(event["evidence"], row["messages"])
        if not e["quote"].strip():
            raise ValueError("Missing nonblank evidence/detail")
        e = {k: e[k] for k in ("message_id", "quote", "occurrence", "start", "end")}
        if key == "233" and event.get("mode") not in MODES:
            raise ValueError("Invalid evidence mode")
        if key in ("234", "245") and event.get("usage") not in ("own", "quoted_or_discussed"):
            raise ValueError("Invalid usage")
        if key == "245" and event.get("form") not in ("direct", "variant", "masked"):
            raise ValueError("Invalid profanity form")
        signature = (e["message_id"], e["start"], e["end"], event.get("mode"))
        if signature in seen:
            prior = next(
                x
                for x in result
                if (x["evidence"]["message_id"], x["evidence"]["start"], x["evidence"]["end"], x.get("mode"))
                == signature
            )
            if any(prior.get(f) != event.get(f) for f in ("usage", "form")):
                raise ValueError("Conflicting duplicate usage")
            continue
        seen.add(signature)
        if key in ("234", "245"):
            positions = set(range(e["start"], e["end"]))
            if occupied.get(e["message_id"], set()) & positions:
                raise ValueError("Overlapping usage events")
            occupied.setdefault(e["message_id"], set()).update(positions)
        result.append(dict(event, evidence=e))
    if key == "233" and state == "present":
        modes = {e["mode"] for e in result}
        if (value["label"] == "mixed" and len(modes) < 2) or (
            value["label"] != "mixed" and modes != {value["label"]}
        ):
            raise ValueError("Mixed requires multiple supported modes; single register must match mode")
    return dict(value, events=result)


def validate_response(value, row):
    raw = value.get("items", {}) if isinstance(value, dict) else {}
    items, failures = {}, {}
    for key in ITEMS:
        try:
            items[key] = validate_item(key, raw.get(key) if isinstance(raw, dict) else None, row)
        except (KeyError, TypeError, ValueError) as exc:
            failures[key] = str(exc)
    return items, failures


def annotate(row, parsed, judge, refresh=False):
    judge = judge.for_record(row)
    payload = {"context": row["context"], "messages": row["messages"]}
    first = judge.call(ANNOTATION_PROMPT, payload, refresh=refresh)
    original = first.get("data", {})
    items, failures = validate_response(original, row)
    initial, provenance = dict(failures), dict.fromkeys(items, first["cache_key"])
    repair_key = None
    if failures:
        repair = judge.call(ANNOTATION_PROMPT, dict(payload, failures=failures), refresh=refresh, attempt=1)
        repair_key = repair["cache_key"]
        data = repair.get("data", {})
        replacements = data.get("items", {}) if isinstance(data, dict) else {}
        merged = dict(items)
        if isinstance(replacements, dict):
            merged.update({k: replacements[k] for k in failures if k in replacements})
        items, failures = validate_response({"items": merged}, row)
        for key in items:
            provenance.setdefault(key, repair_key)
    if failures:
        judge.invalidate(first["cache_key"], repair_key)
    for key, error in failures.items():
        items[key] = {"state": "failure", "reason": error, "events": []}
    return {
        "sample_id": row["sample_id"],
        "items": items,
        "cache_key": first["cache_key"],
        "repair_cache_key": repair_key,
        "item_cache_keys": provenance,
        "initial_failures": initial,
        "remaining_failures": failures,
    }


def reused_annotation(kind, row, parsed, judge):
    module, keys = REUSE[kind]
    source_judge = copy(judge)
    source_judge.protocol = module.PROTOCOL_VERSION
    annotation = module.annotate(row, parsed, source_judge)
    return {
        "sample_id": annotation["sample_id"],
        "protocol": module.PROTOCOL_VERSION,
        "items": {k: annotation["items"][k] for k in keys},
        "cache_key": annotation["cache_key"],
        "item_cache_keys": {
            k: annotation.get("item_cache_keys", {}).get(k, annotation["cache_key"]) for k in keys
        },
        "repair_cache_key": annotation.get("repair_cache_key"),
    }


def summarize(records, parsed):
    return {
        "windows": len(records),
        "messages": sum(len(r["messages"]) for r in records),
        "characters": sum(len(m["text"]) for r in records for m in r["messages"]),
        "words": sum(
            sum(
                lexical_choice.word_token(t, p)
                for t, p in zip(parsed[m["text"]]["cws"], parsed[m["text"]]["pos"], strict=True)
            )
            for r in records
            for m in r["messages"]
        ),
    }


def event_spans(event):
    value = event["evidence"]
    return value if isinstance(value, list) else [value]


def usage_summary(records, items, own_only=False):
    states = Counter(item["state"] for item in items)
    valid = [
        (row, item)
        for row, item in zip(records, items, strict=True)
        if item["state"] in ("present", "absent")
    ]
    events = [
        dict(e, sample_id=row["sample_id"])
        for row, item in valid
        for e in item["events"]
        if not own_only or e["usage"] == "own"
    ]
    hits = {e["sample_id"] for e in events}
    hit_messages = {(e["sample_id"], s["message_id"]) for e in events for s in event_spans(e)}
    char_n = sum(len(m["text"]) for row, _ in valid for m in row["messages"] if m["text"].strip())
    msg_n = sum(bool(m["text"].strip()) for row, _ in valid for m in row["messages"])
    forms = Counter(" … ".join(s["quote"] for s in event_spans(e)) for e in events)
    unknown, failure = missing_messages(
        (row, item["state"]) for row, item in zip(records, items, strict=True)
    )
    return {
        "states": dict(states),
        "valid_messages": msg_n,
        "hit_messages": len(hit_messages),
        "message_occurrence_rate": ratio(len(hit_messages), msg_n),
        "by_length": {
            "message_occurrence_rate": by_length(message_texts(row for row, _ in valid), hit_messages)
        },
        "reliability": {
            "message_occurrence_rate": with_missing(
                reliability(message_texts(row for row, _ in valid), hit_messages), unknown, failure
            )
        },
        "valid_windows": len(valid),
        "coverage": ratio(len(valid), len(records)),
        "unknown_rate": ratio(states["unknown"], len(records)),
        "failure_rate": ratio(states["failure"], len(records)),
        "hit_windows": len(hits),
        "window_occurrence_rate": ratio(len(hits), len(valid)),
        "valid_characters": char_n,
        "event_count": len(events),
        "events_per_100_characters": ratio(DEFAULT_FREQUENCY_SCALE * len(events), char_n),
        "forms": dict(forms),
        "events": events,
    }


def label_summary(records, items, key):
    states = Counter(item["state"] for item in items)
    field = "label"
    labels = Counter(str(item[field]) for item in items if item["state"] == "present")
    # The label belongs to the window; weighting by its nonempty messages gives per-message exposure. Longer
    # windows show more modes and lean to mixed, so each message also carries its window's length.
    weighted, exposure = Counter(), []
    for row, item in zip(records, items, strict=True):
        if item["state"] == "present":
            texts = [m["text"] for m in row["messages"] if m["text"].strip()]
            weighted[str(item[field])] += len(texts)
            exposure += [(sum(map(nonspace_length, texts)), str(item[field]))] * len(texts)
    paired = {
        row["sample_id"]: {
            "context_id": digest(row["context"]) if any(m["text"].strip() for m in row["context"]) else None,
            "label": str(item[field]),
        }
        for row, item in zip(records, items, strict=True)
        if item["state"] == "present"
    }
    return {
        "states": dict(states),
        "labels": dict(labels),
        "valid_windows": sum(labels.values()),
        "window_proportions": {k: ratio(v, sum(labels.values())) for k, v in labels.items()},
        "labeled_messages": sum(weighted.values()),
        "message_label_counts": dict(weighted),
        "message_weighted_proportions": {k: ratio(v, sum(weighted.values())) for k, v in weighted.items()},
        "by_length": {
            "message_weighted_proportions": {
                label: length_strata(((n, own == label) for n, own in exposure), WINDOW_LENGTH_EDGES)
                for label in weighted
            }
        },
        "coverage": ratio(sum(labels.values()), len(records)),
        "unknown_rate": ratio(states["unknown"], len(records)),
        "failure_rate": ratio(states["failure"], len(records)),
        "sample_labels": paired,
    }


def summarize_semantic(records, annotations, reused):
    if [r["sample_id"] for r in records] != [a["sample_id"] for a in annotations]:
        raise ValueError("Annotation alignment mismatch")
    result = {}
    for key in ITEMS:
        items = [a["items"][key] for a in annotations]
        if key == "233":
            result[key] = label_summary(records, items, key)
        else:
            result[key] = usage_summary(records, items, True)
            result[key]["all_usage"] = usage_summary(records, items)
            result[key]["usage_counts"] = dict(Counter(e["usage"] for item in items for e in item["events"]))
            if key == "245":
                result[key]["own_form_counts"] = dict(Counter(e["form"] for e in result[key]["events"]))
    result["244"] = {"components": {}}
    for name, (kind, key) in COMPONENTS.items():
        values = reused[kind]
        if [r["sample_id"] for r in records] != [a["sample_id"] for a in values]:
            raise ValueError("Reused annotation alignment mismatch")
        component = usage_summary(records, [a["items"][key] for a in values])
        component["source"] = {
            "chapter": REUSE[kind][0].__name__.rsplit(".", 1)[-1],
            "item": key,
            "protocol": REUSE[kind][0].PROTOCOL_VERSION,
        }
        result["244"]["components"][name] = component
    return result


def extract(corpora, parsed, judge, output, workers, qa_runner=None):
    semantic = {}
    for name, records in corpora.items():
        dest = output / name
        dest.mkdir()
        write_json(dest / "records.json", records)
        write_json(dest / "register_style.json", summarize(records, parsed))
    if judge:
        print(
            f"Annotating register and reusing lexical/rhetorical sources for {distinct_records(corpora)} windows",
            flush=True,
        )

        def execute(row, kind):
            return (
                annotate(row, parsed, judge)
                if kind == "register"
                else reused_annotation(kind, row, parsed, judge)
            )

        groups = annotate_corpora(corpora, execute, workers, judge=judge, kinds=("register", *REUSE))
        for name, records in corpora.items():
            annotations = groups[name]["register"]
            reused = {kind: groups[name][kind] for kind in REUSE}
            semantic[name] = summarize_semantic(records, annotations, reused)
            write_json(output / name / "annotations.json", annotations)
            write_json(output / name / "reused_annotations.json", reused)
            write_json(output / name / "semantic.json", semantic[name])
            if qa_runner and name == PRIMARY_CORPUS:
                write_json(output / name / "qa.json", qa_runner(records, annotations))
    return semantic


def extraction_coverage(semantic, judge):
    result = {
        key: {"name": title, "state": "extracted" if judge else "not_run"} for key, title in MEASURES.items()
    }
    for key in ITEMS:
        result[key]["corpora"] = {name: values[key]["states"] for name, values in semantic.items()}
    result["244"]["components"] = {
        k: {name: values["244"]["components"][k]["states"] for name, values in semantic.items()}
        for k in COMPONENTS
    }
    result.update({k: {"state": "deferred", "reason": "high_difficulty"} for k in DEFERRED})
    return result


def protocol():
    return {
        "measures": MEASURES,
        "deferred": DEFERRED,
        "formality_levels": {"0": "casual", "1": "neutral", "2": "formal"},
        "register_labels": [*MODES, "mixed"],
        "label_basis": "whole-window expression conventions, not topic, length, politeness words or trigger dictionaries",
        "label_unit": "233 is labeled per window because single short messages rarely show register; message_weighted_proportions weight each window label by its nonempty messages",
        "usage": "234/245 separate own use from quoted/discussed material by current function, not quotation marks; own rate keeps quote-only windows as valid negatives",
        "denominator": "per-item nonempty messages of complete determinate windows (primary), windows and Unicode character lengths; unknown/failure excluded and reported",
        "reuse": {
            name: {
                "chapter": REUSE[kind][0].__name__.rsplit(".", 1)[-1],
                "item": key,
                "protocol": REUSE[kind][0].PROTOCOL_VERSION,
            }
            for name, (kind, key) in COMPONENTS.items()
        },
        "reuse_method": "existing annotate functions and original protocol/payload cache keys; full original shared extraction runs on cache miss; source items retain their own semantics and failures",
        "internet_components": "reported independently; abbreviations/emoticons are not asserted to be uniquely internet-specific; no union or weighted internet score",
        "frequency_scale": DEFAULT_FREQUENCY_SCALE,
        "qa": "repeatability QA covers newly extracted 233/234/245; reused sources retain original cache provenance",
    }
