"""Information density: additional advice relative to visible user requests."""

from weclone.prompts.distill_style import (
    INFORMATION_DENSITY_ANNOTATION_PROMPT as ANNOTATION_PROMPT,
)

from collections import Counter

from .data import PRIMARY_CORPUS, write_json
from .extraction import annotate_corpora, distinct_records, locate
from .stats import by_length, message_texts, missing_messages, ratio, reliability, with_missing

PROTOCOL_VERSION = "information-density-v1"
DEFAULT_FREQUENCY_SCALE = 100
ITEMS = {"256": "是否提供额外建议"}
DEFERRED = ("263",)
SCOPE = "Information Density: medium 256; high composite compression 263 deferred"
RELATIONS = ("requested", "extra", "unknown")


def spans(values, messages):
    if not isinstance(values, list):
        raise ValueError("Expected evidence array")
    order = {m["message_id"]: i for i, m in enumerate(messages)}
    result = []
    used = set()
    for value in values:
        e = locate(value, messages)
        if not e["quote"].strip():
            raise ValueError("Blank evidence")
        positions = {(e["message_id"], i) for i in range(e["start"], e["end"])}
        if positions & used:
            raise ValueError("Overlapping evidence within a unit")
        used.update(positions)
        result.append({k: e[k] for k in ("message_id", "quote", "occurrence", "start", "end")})
    return sorted(result, key=lambda e: (order[e["message_id"]], e["start"], e["end"]))


def validate_item(value, row):
    if not isinstance(value, dict):
        raise ValueError("Missing item 256")
    state, events = value.get("state"), value.get("events")
    if state not in ("present", "absent", "unknown", "na") or not isinstance(events, list):
        raise ValueError("Invalid state/events")
    if (state == "present") != bool(events):
        raise ValueError("State/events conflict")
    if (state == "na") != (not any(m["text"].strip() for m in row["messages"])):
        raise ValueError("NA only for blank reply")
    result = []
    seen = {}
    occupied = set()
    for event in events:
        if event["relation"] not in RELATIONS:
            raise ValueError("Invalid requested/extra relation")
        evidence = spans(event["evidence"], row["messages"])
        context = spans(event["request_evidence"], [m for m in row["context"] if m["role"] == "user"])
        if not evidence:
            raise ValueError("Advice needs target evidence")
        if event["relation"] != "unknown" and not context:
            raise ValueError("Known relation needs user request evidence")
        signature = tuple((e["message_id"], e["start"], e["end"]) for e in evidence)
        if signature in seen:
            if seen[signature] != event["relation"]:
                raise ValueError("Conflicting relation for the same advice")
            continue
        positions = {(e["message_id"], i) for e in evidence for i in range(e["start"], e["end"])}
        if positions & occupied:
            raise ValueError("Overlapping advice units")
        seen[signature] = event["relation"]
        occupied.update(positions)
        result.append(dict(event, evidence=evidence, request_evidence=context))
    order = {m["message_id"]: i for i, m in enumerate(row["messages"])}
    result.sort(key=lambda e: (order[e["evidence"][0]["message_id"]], e["evidence"][0]["start"]))
    return dict(value, events=result)


def annotate(row, parsed, judge, refresh=False):
    judge = judge.for_record(row)
    payload = {"context": row["context"], "messages": row["messages"]}
    first = judge.call(ANNOTATION_PROMPT, payload, refresh=refresh)

    def validate(data):
        try:
            return validate_item(data["items"]["256"], row), {}
        except (ValueError, KeyError, TypeError) as exc:
            return None, {"256": str(exc)}

    original = first.get("data", {})
    item, failures = validate(original)
    initial = dict(failures)
    repair_key = None
    if failures:
        repaired = judge.call(ANNOTATION_PROMPT, dict(payload, failures=failures), refresh=refresh, attempt=1)
        repair_key = repaired["cache_key"]
        item, failures = validate(repaired.get("data", {}))
    if failures:
        judge.invalidate(first["cache_key"], repair_key)
        item = {"state": "failure", "reason": failures["256"], "events": []}
    return {
        "sample_id": row["sample_id"],
        "items": {"256": item},
        "cache_key": first["cache_key"],
        "repair_cache_key": repair_key,
        "item_cache_keys": {"256": repair_key or first["cache_key"]},
        "initial_failures": initial,
        "remaining_failures": failures,
    }


def summarize(records, parsed):
    return {
        "windows": len(records),
        "messages": sum(len(r["messages"]) for r in records),
        "characters": sum(len(m["text"]) for r in records for m in r["messages"]),
    }


def summarize_semantic(records, annotations):
    if [r["sample_id"] for r in records] != [a["sample_id"] for a in annotations]:
        raise ValueError("Annotation identity/order mismatch")
    states = Counter()
    extraction = Counter()
    relations = Counter()
    events = []
    per_window = []
    valid_msgs = valid_chars = valid_windows = valid_with_advice = positive_windows = 0
    extra_chars = 0
    hit_msgs, advice_msgs = set(), set()
    valid_texts, advice_texts = {}, {}
    confirmed_extra = 0
    known_advice = advice_valid_msgs = 0
    for row, a in zip(records, annotations, strict=True):
        item = a["items"]["256"]
        extraction[item["state"]] += 1
        relations.update(e["relation"] for e in item["events"])
        extras = [e for e in item["events"] if e["relation"] == "extra"]
        confirmed_extra += len(extras)
        state = item["state"]
        if state == "present":
            state = (
                "unknown"
                if any(e["relation"] == "unknown" for e in item["events"])
                else "present"
                if extras
                else "absent"
            )
        states[state] += 1
        known = state in ("present", "absent")
        if item["state"] in ("present", "absent"):
            known_advice += 1
            advice_valid_msgs += sum(bool(m["text"].strip()) for m in row["messages"])
            advice_texts.update(message_texts([row]))
            advice_msgs.update(
                (row["sample_id"], s["message_id"]) for e in item["events"] for s in e["evidence"]
            )
        positions = {
            (s["message_id"], i) for e in extras for s in e["evidence"] for i in range(s["start"], s["end"])
        }
        per_window.append(
            {
                "sample_id": row["sample_id"],
                "state": state,
                "advice_count": len(item["events"]) if item["state"] in ("present", "absent") else None,
                "extra_count": len(extras) if known else None,
                "extra_characters": len(positions) if known else None,
            }
        )
        if not known:
            continue
        valid_windows += 1
        valid_with_advice += bool(item["events"])
        positive_windows += bool(extras)
        valid_msgs += sum(bool(m["text"].strip()) for m in row["messages"])
        valid_texts.update(message_texts([row]))
        valid_chars += sum(len(m["text"]) for m in row["messages"] if m["text"].strip())
        extra_chars += len(positions)
        hit_msgs.update((row["sample_id"], s["message_id"]) for e in extras for s in e["evidence"])
        events.extend(dict(e, sample_id=row["sample_id"]) for e in extras)
    lengths = [sum(s["end"] - s["start"] for s in e["evidence"]) for e in events]
    missing = missing_messages(zip(records, (w["state"] for w in per_window), strict=True))
    advice_missing = missing_messages(
        (row, a["items"]["256"]["state"]) for row, a in zip(records, annotations, strict=True)
    )
    return {
        "256": {
            "states": dict(states),
            "extraction_states": dict(extraction),
            "relation_counts": dict(relations),
            "valid_messages": valid_msgs,
            "hit_messages": len(hit_msgs),
            "message_occurrence_rate": ratio(len(hit_msgs), valid_msgs),
            "advice_valid_messages": advice_valid_msgs,
            "advice_hit_messages": len(advice_msgs),
            "message_advice_occurrence_rate": ratio(len(advice_msgs), advice_valid_msgs),
            "by_length": {
                "message_occurrence_rate": by_length(valid_texts, hit_msgs),
                "message_advice_occurrence_rate": by_length(advice_texts, advice_msgs),
            },
            "reliability": {
                "message_occurrence_rate": with_missing(reliability(valid_texts, hit_msgs), *missing),
                "message_advice_occurrence_rate": with_missing(
                    reliability(advice_texts, advice_msgs), *advice_missing
                ),
            },
            "total_windows": len(records),
            "valid_windows": valid_windows,
            "coverage": ratio(valid_windows, len(records)),
            "unknown_rate": ratio(states["unknown"], len(records)),
            "failure_rate": ratio(states["failure"], len(records)),
            "window_occurrence_rate": ratio(positive_windows, valid_windows),
            "valid_characters": valid_chars,
            "event_count": len(events),
            "events_per_100_characters": ratio(DEFAULT_FREQUENCY_SCALE * len(events), valid_chars),
            "extra_character_share": ratio(extra_chars, valid_chars),
            "extra_event_lengths": lengths,
            "window_advice_occurrence_rate": ratio(extraction["present"], known_advice),
            "window_extra_given_advice_rate": ratio(positive_windows, valid_with_advice),
            "valid_windows_with_advice": valid_with_advice,
            "relation_coverage": ratio(relations["requested"] + relations["extra"], sum(relations.values())),
            "confirmed_extra_events_outside_complete_windows": confirmed_extra - len(events),
            "per_window": per_window,
            "events": events,
        }
    }


def extract(corpora, parsed, judge, output, workers, qa_runner=None):
    semantic = {}
    groups = {}
    for name, records in corpora.items():
        dest = output / name
        dest.mkdir()
        write_json(dest / "records.json", records)
        write_json(dest / "information_density.json", summarize(records, parsed))
    if judge:
        print(f"Annotating additional advice for {distinct_records(corpora)} windows", flush=True)
        groups = annotate_corpora(corpora, lambda row: annotate(row, parsed, judge), workers, judge=judge)
        for name, records in corpora.items():
            semantic[name] = summarize_semantic(records, groups[name])
            write_json(output / name / "annotations.json", groups[name])
            write_json(output / name / "semantic.json", semantic[name])
            if qa_runner and name == PRIMARY_CORPUS:
                write_json(output / name / "qa.json", qa_runner(records, groups[name]))
    return semantic


def extraction_coverage(semantic, judge):
    return {
        "256": {
            "state": "extracted" if judge else "not_run",
            "corpora": {name: value["256"]["states"] for name, value in semantic.items()},
        },
        "263": {
            "state": "deferred",
            "reason": "high_difficulty_composite_requires_ellipsis_redundancy_and_discourse_inputs",
        },
    }


def protocol():
    return {
        "items": ITEMS,
        "deferred": DEFERRED,
        "relations": RELATIONS,
        "frequency_scale": DEFAULT_FREQUENCY_SCALE,
        "rule_basis": "action proposal relative to visible user request scope, including implicit task requirements; no suggestion-word or connective matching",
        "unit": "one advice occurrence with one or multiple nonoverlapping target spans; independent proposals separate; identical span sets deduplicated; not unique semantic action counts",
        "evidence": "requested/extra require exact context user evidence as well as target evidence; absence of user context leaves existing advice relation unknown",
        "denominator": "nonempty messages (primary) and windows whose advice extraction and event relations are all known; unknown/failed windows excluded, zero only for confirmed no-extra-advice windows",
        "length": "raw Unicode code points of extra advice spans, union within window; context and gaps not counted",
        "partial_relations": "known extra events in incomplete windows retained in annotations and reported as a lower-bound count outside primary complete-window statistics",
    }
