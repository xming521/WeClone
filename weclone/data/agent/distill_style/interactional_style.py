"""Interactional style: message splitting, questions and contextual responses."""

from weclone.prompts.distill_style import (
    INTERACTIONAL_STYLE_ANNOTATION_PROMPT as ANNOTATION_PROMPT,
)

import re
from collections import Counter

from .data import PRIMARY_CORPUS, UNITS, bubble_spans, write_json
from .extraction import annotate_corpora, distinct_records, locate
from .stats import (
    LENGTH_BUCKETS,
    by_length,
    length_bucket,
    length_strata,
    message_texts,
    nonspace_length,
    missing_messages,
    quantiles,
    ratio,
    reliability,
    window_reliability,
    with_missing,
)

PROTOCOL_VERSION = "interactional-style-v2"
DEFAULT_LONG_CHARACTERS = 200
SCOPE = "Interactional Style: newline burst proxy and 11 medium items; high 220 deferred"
ITEMS = {
    "questions": "共享提问标注",
    "218": "延展形态",
    "219": "主动延展",
    "221": "结束表达",
    "227": "复述对方",
    "228": "情绪与事情回应",
    "229": "直接昵称",
    "231": "显式附和",
}
MEASURES = {
    "burst_pattern": "连发模式",
    "210": "追问",
    "212": "独立追问数",
    "214": "追问措辞",
    **{k: v for k, v in ITEMS.items() if k != "questions"},
}
LABELS = {
    "218": ("explanation", "monologue"),
    "219": ("same_topic", "new_branch"),
    "221": ("farewell", "departure", "wrap_up", "other"),
    "228": ("emotion", "matter"),
}
NEWLINE = re.compile(r"\r\n|[\r\n]")


def span(e, messages):
    e = locate(e, messages)
    if not e["quote"].strip():
        raise ValueError("Whitespace evidence")
    return {k: e[k] for k in ("message_id", "quote", "occurrence", "start", "end")}


def inside(a, b):
    return a["message_id"] == b["message_id"] and b["start"] <= a["start"] < a["end"] <= b["end"]


def identity(e):
    return e["message_id"], e["start"], e["end"]


def context_spans(values, row, user_only=False):
    if not isinstance(values, list):
        raise ValueError("Expected context_evidence array")
    return [span(e, [m for m in row["context"] if not user_only or m["role"] == "user"]) for e in values]


def state_check(value, row, states, array):
    if not isinstance(value, dict) or value.get("state") not in states:
        raise ValueError("Missing part or invalid state")
    if not isinstance(value.get(array), list):
        raise ValueError("Expected array")
    empty = not any(m["text"].strip() for m in row["messages"])
    if (value["state"] == "na") != empty:
        raise ValueError("NA only for blank reply")


def validate_questions(value, row):
    state_check(value, row, ("known", "unknown", "na"), "questions")
    if not isinstance(value.get("questions"), list):
        raise ValueError("Missing independent questions")
    if value["state"] != "known":
        if value["questions"]:
            raise ValueError("Unknown/NA must have empty arrays")
        return dict(value)
    questions, seen = [], set()
    for q in value["questions"]:
        e, focus = span(q["evidence"], row["messages"]), span(q["focus"], row["messages"])
        if not inside(focus, e) or identity(focus) in seen:
            raise ValueError("Distinct question focus must be inside expression")
        seen.add(identity(focus))
        if q["kind"] not in ("followup", "new", "unknown"):
            raise ValueError("Invalid independent question")
        if q["register"] not in ("colloquial", "neutral", "formal", "mixed", "unknown") or q[
            "restatement"
        ] not in ("yes", "no", "unknown"):
            raise ValueError("Invalid question attributes")
        if q["kind"] != "followup" and q["restatement"] != "unknown":
            raise ValueError("Restatement only applies to followups")
        context = context_spans(q["context_evidence"], row)
        if q["kind"] == "followup" and not context:
            raise ValueError("Followup needs prior information evidence")
        if not row["context"] and q["kind"] != "unknown":
            raise ValueError("Missing context cannot distinguish followup/new")
        if not isinstance(q["particles"], list):
            raise ValueError("Expected particle array")
        particles = [span(p, row["messages"]) for p in q["particles"]]
        if any(not inside(p, e) for p in particles) or len({identity(p) for p in particles}) != len(
            particles
        ):
            raise ValueError("Invalid particle evidence")
        questions.append(dict(q, evidence=e, focus=focus, context_evidence=context, particles=particles))
    return dict(value, questions=questions)


def validate_item(key, value, row):
    if key == "questions":
        return validate_questions(value, row)
    state_check(value, row, ("present", "absent", "unknown", "na"), "events")
    if (value["state"] == "present") != bool(value["events"]):
        raise ValueError("State/events mismatch")
    if key == "219" and not row["context"] and value["state"] not in ("unknown", "na"):
        raise ValueError("No context for requested versus additional topic")
    events, seen = [], set()
    for event in value["events"]:
        e = span(event["evidence"], row["messages"])
        context = context_spans(event["context_evidence"], row, key in ("227", "228", "231"))
        if key in ("219", "227", "228", "231") and not context:
            raise ValueError("Missing aligned prior evidence")
        if key in LABELS and event.get("label") not in LABELS[key]:
            raise ValueError("Invalid event label")
        signature = identity(e) + (event.get("label"),)
        if signature in seen:
            continue
        seen.add(signature)
        events.append(dict(event, evidence=e, context_evidence=context))
    return dict(value, events=events)


def validate_response(value, row):
    raw = value.get("items", {}) if isinstance(value, dict) else {}
    items, failures = {}, {}
    for key in ITEMS:
        try:
            items[key] = validate_item(key, raw.get(key) if isinstance(raw, dict) else None, row)
        except (ValueError, TypeError, KeyError) as exc:
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


def surface_row(row, long_characters):
    messages = []
    for m in row["messages"]:
        text = m["text"]
        breaks = list(NEWLINE.finditer(text))
        pieces = [
            {"start": start, "end": stop, "quote": text[start:stop], "characters": stop - start}
            for start, stop in bubble_spans(text)
        ]
        messages.append(
            {
                "message_id": m["message_id"],
                "characters": len(text),
                "nonspace_characters": nonspace_length(text),
                "nonempty": bool(text.strip()),
                "newline_count": len(breaks),
                "bubbles": pieces,
                "bubble_count": len(pieces),
            }
        )
    chars = sum(m["characters"] for m in messages)
    return {
        "sample_id": row["sample_id"],
        "characters": chars,
        "long_window": chars >= long_characters,
        "observed_message_count": len(messages),
        "nonempty_message_count": sum(m["nonempty"] for m in messages),
        "bubble_count": sum(m["bubble_count"] for m in messages),
        "newline_burst": any(m["bubble_count"] > 1 for m in messages),
        "messages": messages,
    }


def summarize(records, parsed, long_characters=DEFAULT_LONG_CHARACTERS):
    rows = [surface_row(r, long_characters) for r in records]
    nonempty = [r for r in rows if r["nonempty_message_count"]]
    messages = [m for r in rows for m in r["messages"] if m["nonempty"]]
    windows = [r["sample_id"] for r in rows for m in r["messages"] if m["nonempty"]]
    bubbles = [b["characters"] for m in messages for b in m["bubbles"]]

    def message_reliability(hit):
        return window_reliability(zip(windows, map(hit, messages), strict=True))

    buckets = Counter(length_bucket(m["nonspace_characters"]) for m in messages)
    return {
        "units": UNITS,
        "windows": len(rows),
        "per_window": rows,
        "long_characters": long_characters,
        "nonempty_messages": len(messages),
        "length_summary": {
            "message": quantiles([m["characters"] for m in messages]),
            "bubble": quantiles(bubbles),
            "window": quantiles([r["characters"] for r in nonempty]),
        },
        "message_lengths": [m["characters"] for r in rows for m in r["messages"]],
        "bubble_lengths": bubbles,
        "window_lengths": [r["characters"] for r in rows],
        "message_length_counts": {b: buckets[b] for b in LENGTH_BUCKETS[1:]},
        "message_length_shares": {b: ratio(buckets[b], len(messages)) for b in LENGTH_BUCKETS[1:]},
        "bubbles_per_message": dict(Counter(str(m["bubble_count"]) for m in messages)),
        "message_newline_burst_rate": ratio(sum(m["bubble_count"] > 1 for m in messages), len(messages)),
        "by_length": {
            "message_newline_burst_rate": length_strata(
                (m["nonspace_characters"], m["bubble_count"] > 1) for m in messages
            )
        },
        "reliability": {
            "message_newline_burst_rate": message_reliability(lambda m: m["bubble_count"] > 1),
            "message_long_rate": message_reliability(lambda m: m["characters"] >= long_characters),
            "message_length_shares": {
                b: message_reliability(lambda m, b=b: length_bucket(m["nonspace_characters"]) == b)
                for b in LENGTH_BUCKETS[1:]
            },
        },
        "message_long_rate": ratio(sum(m["characters"] >= long_characters for m in messages), len(messages)),
        "messages_per_window": dict(Counter(str(r["observed_message_count"]) for r in rows)),
        "bubbles_per_window": dict(Counter(str(r["bubble_count"]) for r in rows)),
        "window_newline_burst_rate": ratio(sum(r["newline_burst"] for r in nonempty), len(nonempty)),
        "window_long_rate": ratio(sum(r["long_window"] for r in nonempty), len(nonempty)),
    }


def base_summary(states, events, records):
    counts = Counter(states)
    valid = [r for r, s in zip(records, states, strict=True) if s in ("present", "absent")]
    chars = sum(len(m["text"]) for r in valid for m in r["messages"] if m["text"].strip())
    nmsg = sum(bool(m["text"].strip()) for r in valid for m in r["messages"])
    hits = {(e["sample_id"], e["evidence"]["message_id"]) for e in events}
    missing = missing_messages(zip(records, states, strict=True))
    return {
        "states": dict(counts),
        "valid_messages": nmsg,
        "hit_messages": len(hits),
        "message_occurrence_rate": ratio(len(hits), nmsg),
        "by_length": {"message_occurrence_rate": by_length(message_texts(valid), hits)},
        "reliability": {
            "message_occurrence_rate": with_missing(reliability(message_texts(valid), hits), *missing)
        },
        "valid_characters": chars,
        "events_per_100_characters": ratio(100 * len(events), chars),
        "valid_windows": len(valid),
        "coverage": ratio(len(valid), len(records)),
        "window_occurrence_rate": ratio(counts["present"], len(valid)),
        "unknown_rate": ratio(counts["unknown"], len(records)),
        "failure_rate": ratio(counts["failure"], len(records)),
        "event_count": len(events),
        "events": events,
    }


def response_order(events, row):
    order = {m["message_id"]: i for i, m in enumerate(row["messages"])}

    def ranges(label):
        return [
            (
                (order[e["evidence"]["message_id"]], e["evidence"]["start"]),
                (order[e["evidence"]["message_id"]], e["evidence"]["end"]),
            )
            for e in events
            if e["label"] == label
        ]

    emotion, matter = ranges("emotion"), ranges("matter")
    if not emotion and not matter:
        return "neither"
    if not emotion:
        return "matter_only"
    if not matter:
        return "emotion_only"
    if max(e[1] for e in emotion) <= min(e[0] for e in matter):
        return "emotion_first"
    if max(e[1] for e in matter) <= min(e[0] for e in emotion):
        return "matter_first"
    return "mixed_or_overlap"


def summarize_semantic(records, annotations):
    if [r["sample_id"] for r in records] != [a["sample_id"] for a in annotations]:
        raise ValueError("Annotation alignment mismatch")
    result = {}
    for key in ITEMS:
        if key == "questions":
            continue
        states = [a["items"][key]["state"] for a in annotations]
        events = [dict(e, sample_id=a["sample_id"]) for a in annotations for e in a["items"][key]["events"]]
        value = base_summary(states, events, records)
        if key in LABELS:
            value["labels"] = dict(Counter(e["label"] for e in events))
            label_hits = {
                label: {(e["sample_id"], e["evidence"]["message_id"]) for e in events if e["label"] == label}
                for label in LABELS[key]
            }
            value["label_message_rates"] = {
                label: ratio(len(hits), value["valid_messages"]) for label, hits in label_hits.items()
            }
            texts = message_texts(
                r for r, s in zip(records, states, strict=True) if s in ("present", "absent")
            )
            value["by_length"]["label_message_rates"] = {
                label: by_length(texts, hits) for label, hits in label_hits.items()
            }
            missing = missing_messages(zip(records, states, strict=True))
            value["reliability"]["label_message_rates"] = {
                label: with_missing(reliability(texts, hits), *missing) for label, hits in label_hits.items()
            }
            value["label_window_rates"] = {
                label: ratio(
                    len({e["sample_id"] for e in events if e["label"] == label}), value["valid_windows"]
                )
                for label in LABELS[key]
            }
        if key == "228":
            orders = Counter(
                response_order(a["items"][key]["events"], row)
                for row, a in zip(records, annotations, strict=True)
                if a["items"][key]["state"] in ("present", "absent")
            )
            both = sum(orders[k] for k in ("emotion_first", "matter_first", "mixed_or_overlap"))
            value.update(
                order_counts=dict(orders),
                both_response_windows=both,
                emotion_first_given_both=ratio(orders["emotion_first"], both),
            )
        result[key] = value
    qs = [a["items"]["questions"] for a in annotations]
    states = []
    followups = []
    counts = []
    attr = {k: Counter() for k in ("register", "restatement", "particles", "expressions")}
    lengths = []
    independent_counts = []
    for row, q in zip(records, qs, strict=True):
        sid = row["sample_id"]
        if q["state"] != "known":
            states.append(q["state"])
            counts.append(None)
            independent_counts.append(None)
            continue
        questions = q["questions"]
        independent_counts.append(len(questions))
        fs = [e for e in questions if e["kind"] == "followup"]
        unresolved = any(e["kind"] == "unknown" for e in questions)
        state = "unknown" if unresolved else "present" if fs else "absent"
        states.append(state)
        counts.append(len(fs) if state != "unknown" else None)
        if state != "unknown":
            followups.extend(dict(e, sample_id=sid) for e in fs)
            for e in fs:
                for field in ("register", "restatement"):
                    attr[field][e[field]] += 1
                attr["expressions"][e["evidence"]["quote"]] += 1
                attr["particles"].update(p["quote"] for p in e["particles"])
                lengths.append(len(e["evidence"]["quote"]))
    result["210"] = base_summary(states, followups, records)
    result["210"]["question_kind_counts"] = dict(
        Counter(e["kind"] for q in qs if q["state"] == "known" for e in q["questions"])
    )
    validcounts = [v for v in counts if v is not None]
    positive = [v for v in validcounts if v > 0]
    per_message = Counter((e["sample_id"], e["evidence"]["message_id"]) for e in followups)
    known_messages = [
        (row["sample_id"], m["message_id"])
        for row, c in zip(records, counts, strict=True)
        if c is not None
        for m in row["messages"]
        if m["text"].strip()
    ]
    result["212"] = {
        "states": result["210"]["states"],
        "coverage": result["210"]["coverage"],
        "valid_messages": len(known_messages),
        "mean_per_valid_message": ratio(sum(validcounts), len(known_messages)),
        "message_count_distribution": dict(Counter(str(per_message[k]) for k in known_messages)),
        "window_followup_counts": counts,
        "window_independent_question_counts": independent_counts,
        "window_count_distribution": dict(Counter(str(v) for v in validcounts)),
        "mean_per_valid_window": ratio(sum(validcounts), len(validcounts)),
        "mean_per_window_given_followup": ratio(sum(positive), len(positive)),
        "windows_with_followup": len(positive),
    }
    result["214"] = {
        "states": result["210"]["states"],
        "coverage": result["210"]["coverage"],
        "followup_count": len(followups),
        "attributes": {k: dict(v) for k, v in attr.items()},
        "expression_lengths": lengths,
        "attribute_coverage": {
            k: ratio(sum(v for label, v in attr[k].items() if label != "unknown"), len(followups))
            for k in ("register", "restatement")
        },
        "particle_question_rate": ratio(sum(bool(e["particles"]) for e in followups), len(followups)),
    }
    return result


def extract(corpora, parsed, judge, output, workers, qa_runner=None, long_characters=DEFAULT_LONG_CHARACTERS):
    raw, semantic, annotations = {}, {}, {}
    for name, records in corpora.items():
        dest = output / name
        dest.mkdir()
        raw[name] = summarize(records, parsed, long_characters)
        write_json(dest / "records.json", records)
        write_json(dest / "interactional_style.json", raw[name])
    if judge:
        print(f"Annotating interactional style for {distinct_records(corpora)} windows", flush=True)
        annotations = annotate_corpora(
            corpora, lambda row: annotate(row, parsed, judge), workers, judge=judge
        )
        for name, records in corpora.items():
            semantic[name] = summarize_semantic(records, annotations[name])
            write_json(output / name / "annotations.json", annotations[name])
            write_json(output / name / "semantic.json", semantic[name])
            if qa_runner and name == PRIMARY_CORPUS:
                write_json(output / name / "qa.json", qa_runner(records, annotations[name]))
    return semantic


def extraction_coverage(semantic, judge):
    return {
        **{
            k: {
                "name": v,
                "state": "extracted"
                if k == "burst_pattern" or judge
                else "partial"
                if k == "218"
                else "not_run",
                "corpora": {name: value[k]["states"] for name, value in semantic.items() if k in value},
            }
            for k, v in MEASURES.items()
        },
        "220": {"state": "deferred", "reason": "high_difficulty_requires_timing"},
    }


def protocol(long_characters=DEFAULT_LONG_CHARACTERS):
    return {
        "measures": MEASURES,
        "labels": LABELS,
        "shared_extraction": "questions serves 210/212/214; sentence forms reuse syntactic 052,211; all parts in one call with at most one repair",
        "units": UNITS,
        "newline_split": "LF/CRLF/CR; discard empty/whitespace-only pieces, retain original offsets; literal backslash-n not split; each nonblank piece is a bubble",
        "burst_interpretation": "newline-based split preference, not timestamp-verified consecutive sends; message-level rates are primary, window-level rates depend on window size",
        "long_characters": long_characters,
        "long_threshold_basis": "200 raw Unicode code points is a configurable descriptive threshold, not a validated monologue boundary; applied per message and per window, full lengths and quantiles retained",
        "sentence_unit": "LLM complete-form segments covering all nonspace characters, not question mark counts or independent question objects",
        "questions": "distinct semantic goals with distinct contained focus spans; reply counts null if followup/new unknown; lack of context cannot confirm either",
        "question_attributes": "conditional on confirmed followups; unknown register/restatement retained and excluded from determinate attribute counts; particles function-labeled without fixed vocabulary",
        "denominator": "per-item nonempty messages and characters of present+absent windows for message rates; the windows themselves only for window_* rates; question unit/message form rates separately report known coverage",
        "emotion_order": "only spans responding to user context; all emotion before all matter, reverse, or mixed/overlap; conditional rate only replies containing both",
        "closing": "explicit ending expression only, no inference of actual termination without later records",
        "context_alignment": "227/228/231 require user evidence; 219/confirmed followup require visible prior information; context never counted as target events",
        "no_aggregate_score": True,
    }
