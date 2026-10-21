"""Pragmatic tone and stance: contextual functions and explicit denominators."""

from weclone.prompts.distill_style import (
    PRAGMATIC_STYLE_ANNOTATION_PROMPT as ANNOTATION_PROMPT,
)

from collections import Counter

from .data import PRIMARY_CORPUS, write_json
from .extraction import annotate_corpora, distinct_records, locate
from .lexical_choice import word_token
from .orthographic_style import capitalization, punctuation_events
from .paralinguistic_style import repetitions
from .stats import length_strata, missing_entries, nonspace_length, ratio, window_reliability, with_missing

PROTOCOL_VERSION = "pragmatic-style-v2"
SCOPE = "Pragmatic Tone & Stance: 11 merged measures from original 117-150"
DEFAULT_FREQUENCY_SCALE = 100
DEFAULT_POSITION_BINS = 3
SOURCE_IDS = {
    "117-119": list(range(117, 120)),
    "120-127": list(range(120, 128)),
    "128-131": list(range(128, 132)),
    "132-134": list(range(132, 135)),
    "135": [135],
    "136-137": [136, 137],
    "138": [138],
    "139": [139],
    "140-144": list(range(140, 145)),
    "145": [145],
    "146-150": list(range(146, 151)),
}
ITEMS = {
    "117-119": "命题确定性",
    "120-127": "立场限定与缓和",
    "128-131": "礼貌行为",
    "132-134": "敬意与自谦",
    "135": "客套程度",
    "136-137": "行动约束",
    "138": "坚持与直接性",
    "139": "拒绝缓冲",
    "140-144": "元话语",
    "145": "认同让步与反驳组织",
    "146-150": "强调",
}
CONDITIONAL = {"136-137", "138", "139", "145"}
ENUMS = {
    "certainty": ("strong", "medium", "weak", "unexpressed", "unknown"),
    "strategies": ("commitment_reduction", "perspective_information", "degree_scope"),
    "mitigation": ("yes", "no", "unknown"),
    "functions": ("request_care", "gratitude", "apology", "other"),
    "relations": ("respect", "self_humbling"),
    "level": ("light", "marked"),
    "act": ("command", "request", "suggestion", "permission"),
    "choice": ("compulsory", "optional", "unexpressed", "unknown"),
    "insistence": ("fixed", "adjustable", "unexpressed", "unknown"),
    "directness": ("explicit", "implicit", "unknown"),
    "buffering": ("yes", "no", "unknown"),
    "metadiscourse": (
        "judgment_process",
        "expression_repair",
        "organization_preview",
        "attitude_shift",
        "other",
    ),
    "alignment": ("yes", "no", "unknown"),
    "methods": ("repetition_restatement", "graphic", "quantitative_degree", "other"),
}
LABEL_FIELDS = {
    "117-119": ("certainty",),
    "120-127": ("strategies", "mitigation"),
    "128-131": ("functions",),
    "132-134": ("relations",),
    "135": ("level",),
    "136-137": ("act", "choice"),
    "138": ("insistence", "directness"),
    "139": ("buffering",),
    "140-144": ("metadiscourse",),
    "145": ("alignment",),
    "146-150": ("methods",),
}
MULTI_FIELDS = {"strategies", "functions", "relations", "metadiscourse", "methods"}
REFERENCES = {
    "proposition_ids": "117-119",
    "action_ids": "136-137",
    "politeness_ids": "128-131",
    "respect_ids": "132-134",
    "hedge_ids": "120-127",
}
REF_FIELDS = {
    "120-127": ("proposition_ids",),
    "128-131": ("action_ids",),
    "135": ("politeness_ids", "respect_ids"),
    "138": ("action_ids", "hedge_ids"),
}


def text_field(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected nonempty description")
    return value


def unique_strings(values, allowed=None):
    if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
        raise ValueError("Expected string list")
    if len(set(values)) != len(values) or (
        allowed is not None and (not values or not set(values) <= set(allowed))
    ):
        raise ValueError("Duplicate or invalid labels")
    return values


def spans(values, messages):
    if not isinstance(values, list):
        raise ValueError("Expected span list")
    result = [locate(v, messages) for v in values]
    keys = [(v["message_id"], v["start"], v["end"]) for v in result]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate evidence")
    return result


def validate_item(item, values, row):
    if not isinstance(values, list):
        raise ValueError("Item must cover every target message")
    expected = [m["message_id"] for m in row["messages"]]
    by_message = {v["message_id"]: v for v in values}
    if len(values) != len(by_message) or set(by_message) != set(expected):
        raise ValueError("Missing or duplicate message")
    ids, result = set(), []
    for msg in row["messages"]:
        value = by_message[msg["message_id"]]
        state, events = value["state"], value["events"]
        allowed = {"present", "unknown", "na"} if item in CONDITIONAL else {"present", "absent", "unknown"}
        if not msg["text"].strip():
            allowed = {"na"}
        if state not in allowed or not isinstance(events, list) or (state == "present") != bool(events):
            raise ValueError("State/event or applicability mismatch")
        checked, seen_spans, seen_targets = [], set(), set()
        for raw in events:
            e = dict(raw)
            eid = text_field(e["id"])
            if eid in ids:
                raise ValueError("Duplicate event id")
            ids.add(eid)
            e["evidence"] = locate(e["evidence"], [msg])
            key = (e["evidence"]["start"], e["evidence"]["end"])
            if key in seen_spans:
                raise ValueError("Same expression must be one multi-attribute event")
            seen_spans.add(key)
            text_field(e["target"])
            e["scope"] = spans(e["scope"], row["context"] + row["messages"])
            for field in LABEL_FIELDS[item]:
                if field in MULTI_FIELDS:
                    unique_strings(e[field], ENUMS[field])
                elif e[field] not in ENUMS[field]:
                    raise ValueError("Invalid " + field)
            for field in REF_FIELDS.get(item, ()):
                unique_strings(e[field])
            if item == "117-119":
                if e["certainty"] == "unexpressed":
                    raise ValueError("Unexpressed certainty is not an extracted event")
                e["certainty_evidence"] = spans(e["certainty_evidence"], [msg])
                if (e["certainty"] in {"strong", "medium", "weak"}) != bool(e["certainty_evidence"]):
                    raise ValueError("Explicit certainty requires separate supporting evidence")
            if item == "136-137":
                if not isinstance(e["condition"], str):
                    raise ValueError("Condition must be text")
                e["condition_evidence"] = spans(e["condition_evidence"], row["context"] + row["messages"])
                if bool(e["condition"].strip()) != bool(e["condition_evidence"]):
                    raise ValueError("Condition requires evidence")
            if item in {"139", "145"}:
                field, decision = ("buffers", "buffering") if item == "139" else ("supports", "alignment")
                if not isinstance(e[field], list) or (e[decision] == "yes") != bool(e[field]):
                    raise ValueError("Relation decision/evidence mismatch")
                nested, nested_seen = [], set()
                for child in e[field]:
                    c = dict(child)
                    c["evidence"] = locate(c["evidence"], row["messages"])
                    ck = tuple(c["evidence"][x] for x in ("message_id", "start", "end"))
                    if ck in nested_seen:
                        raise ValueError("Duplicate linked fragment")
                    nested_seen.add(ck)
                    label = "kind" if item == "139" else "role"
                    choices = (
                        {"apology", "reason", "alternative", "other"}
                        if item == "139"
                        else {"agreement", "concession"}
                    )
                    if c[label] not in choices:
                        raise ValueError("Invalid relation kind")
                    if item == "139":
                        unique_strings(c["politeness_ids"])
                        unique_strings(c["hedge_ids"])
                    nested.append(c)
                e[field] = nested
            if item == "146-150":
                target = (
                    tuple(sorted((s["message_id"], s["start"], s["end"]) for s in e["scope"])) or e["target"]
                )
                if target in seen_targets:
                    raise ValueError("Multiple emphasis methods on same content must merge")
                seen_targets.add(target)
            checked.append(e)
        if item == "135" and len(checked) > 1:
            raise ValueError("One phatic degree per message")
        result.append(dict(value, events=checked))
    return result


def validate_response(response, row):
    items, failures = {}, {}
    raw = response.get("items", {}) if isinstance(response, dict) else {}
    for key in ITEMS:
        try:
            items[key] = validate_item(key, raw[key], row)
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            failures[key] = str(exc)
    index = {k: {e["id"]: e for m in value for e in m["events"]} for k, value in items.items()}
    if "138" in items:
        applicable_messages = {
            m["message_id"]
            for key in ("136-137", "139", "145")
            for m in items.get(key, [])
            if m["state"] == "present"
        }
        if any(m["message_id"] in applicable_messages and m["state"] != "present" for m in items["138"]):
            failures["138"] = (
                "Known action/refusal/rebuttal requires applicable stance annotation; use unknown axes if needed"
            )
    for key, messages in items.items():
        try:
            for m in messages:
                for e in m["events"]:
                    nodes = [e] + e.get("buffers", [])
                    for node in nodes:
                        for field, target in REFERENCES.items():
                            for eid in node.get(field, []):
                                if eid not in index.get(target, {}):
                                    raise ValueError(f"Unresolved {field}: {eid}")
                    if key == "128-131" and "request_care" in e["functions"]:
                        if not e["action_ids"] or any(
                            index["136-137"][eid]["act"] != "request" for eid in e["action_ids"]
                        ):
                            raise ValueError("Request care needs actual request references")
        except (KeyError, ValueError, TypeError) as exc:
            failures[key] = str(exc)
    while True:
        dependent = {}
        for key, messages in items.items():
            if key in failures:
                continue
            for m in messages:
                for e in m["events"]:
                    for node in [e] + e.get("buffers", []):
                        for field, target in REFERENCES.items():
                            if node.get(field) and target in failures:
                                dependent[key] = "Referenced item failed: " + target
        if not dependent:
            break
        failures.update(dependent)
    for key in failures:
        items.pop(key, None)
    return items, failures


def form_observations(row, parsed):
    result = []
    for msg in row["messages"]:
        p = parsed[msg["text"]]
        result.append(
            {
                "message_id": msg["message_id"],
                "characters": len(msg["text"]),
                "words": sum(word_token(w, tag) for w, tag in zip(p["cws"], p["pos"], strict=True)),
                "repetitions": repetitions(msg["text"]),
                "punctuation": punctuation_events(msg["text"]),
                "capitalization": capitalization(msg["text"]),
            }
        )
    return result


def annotate(row, parsed, judge, refresh=False):
    judge = judge.for_record(row)
    payload = {
        "context": row["context"],
        "messages": row["messages"],
        "forms": form_observations(row, parsed),
    }
    first = judge.call(ANNOTATION_PROMPT, payload, refresh=refresh)
    original = first.get("data", {})
    items, failures = validate_response(original, row)
    initial = dict(failures)
    provenance = dict.fromkeys(items, first["cache_key"])
    repair_key = None
    if failures:
        repaired = judge.call(ANNOTATION_PROMPT, dict(payload, failures=failures), refresh=refresh, attempt=1)
        repair_key = repaired["cache_key"]
        repair_data = repaired.get("data", {})
        replacement = repair_data.get("items", {}) if isinstance(repair_data, dict) else {}
        merged = {"items": dict(items)}
        if isinstance(replacement, dict):
            merged["items"].update({key: replacement[key] for key in failures if key in replacement})
        items, failures = validate_response(merged, row)
        for key in items:
            provenance.setdefault(key, repair_key)
    if failures:
        judge.invalidate(first["cache_key"], repair_key)
    for key in failures:
        items[key] = [
            {"message_id": m["message_id"], "state": "failure", "events": [], "reason": failures[key]}
            for m in row["messages"]
        ]
    return {
        "sample_id": row["sample_id"],
        "items": items,
        "cache_key": first["cache_key"],
        "repair_cache_key": repair_key,
        "item_cache_keys": provenance,
        "initial_failures": initial,
        "remaining_failures": failures,
    }


def relative_position(evidence, text):
    left, right = len(text) - len(text.lstrip()), len(text.rstrip())
    midpoint = (evidence["start"] + evidence["end"]) / 2
    value = (midpoint - left) / (right - left)
    return {
        "relative_midpoint": value,
        "bin": ("front", "middle", "back")[
            min(DEFAULT_POSITION_BINS - 1, int(value * DEFAULT_POSITION_BINS))
        ],
    }


def order_of(first, second, row):
    order = {m["message_id"]: i for i, m in enumerate(row["messages"])}
    a = order[first["message_id"]]
    b = order[second["message_id"]]
    if a < b or (a == b and first["end"] <= second["start"]):
        return "before"
    if a > b or (a == b and first["start"] >= second["end"]):
        return "after"
    return "overlap"


def labels_summary(entries, getter, absent_label=None):
    counts, message_counts = Counter(), Counter()
    known_messages = unknown_labels = 0
    labeled = []
    for entry in entries:
        if entry["state"] not in {"present", "absent"}:
            continue
        labels = [label for e in entry["events"] for label in getter(e)]
        if entry["state"] == "absent" and absent_label is not None:
            labels = [absent_label]
        counts.update(labels)
        if "unknown" not in labels:
            known_messages += 1
            message_counts.update(set(labels))
            labeled.append((entry["sample_id"], entry["nonspace_characters"], set(labels)))
        else:
            unknown_labels += 1
    unknown, failure = missing_entries(entries)
    known = sum(v for k, v in counts.items() if k != "unknown")
    return {
        "counts": counts,
        "known_labels": known,
        "unknown_labels": counts["unknown"],
        "unknown_fraction": ratio(counts["unknown"], sum(counts.values())),
        "proportions": {k: ratio(v, known) for k, v in counts.items() if k != "unknown"},
        "message_denominator": known_messages,
        "message_counts": message_counts,
        "message_rates": {k: ratio(v, known_messages) for k, v in message_counts.items()},
        "by_length": {
            "message_rates": {
                k: length_strata((n, k in labels) for _, n, labels in labeled) for k in message_counts
            }
        },
        "reliability": {
            "message_rates": {
                k: with_missing(
                    window_reliability((w, k in labels) for w, _, labels in labeled),
                    unknown + unknown_labels,
                    failure,
                )
                for k in message_counts
            }
        },
    }


def summarize(records, parsed):
    observations = [
        dict(value, sample_id=row["sample_id"]) for row in records for value in form_observations(row, parsed)
    ]
    return {
        "n_windows": len(records),
        "n_messages": len(observations),
        "characters": sum(o["characters"] for o in observations),
        "words": sum(o["words"] for o in observations),
        "surface_only": True,
        "messages": observations,
        "forms": {
            key: Counter(e.get("label", e.get("unit")) for o in observations for e in o[key])
            for key in ("repetitions", "punctuation", "capitalization")
        },
    }


def annotated_entries(records, annotations, raw, item):
    sizes = {(o["sample_id"], o["message_id"]): o for o in raw["messages"]}
    result = []
    for row, a in zip(records, annotations, strict=True):
        if a["sample_id"] != row["sample_id"]:
            raise ValueError("Annotations and records must align")
        texts = {m["message_id"]: m["text"] for m in row["messages"]}
        for entry in a["items"][item]:
            size = sizes[(row["sample_id"], entry["message_id"])]
            result.append(
                dict(
                    entry,
                    row=row,
                    sample_id=row["sample_id"],
                    words=size["words"],
                    characters=size["characters"],
                    nonspace_characters=nonspace_length(texts[entry["message_id"]]),
                )
            )
    return result


def conditional_requests(annotations):
    counts = Counter()
    detection_unknown = 0
    for a in annotations:
        detection_unknown += sum(m["state"] in {"unknown", "failure"} for m in a["items"]["136-137"])
        polite = a["items"]["128-131"]
        complete = all(m["state"] in {"present", "absent", "na"} for m in polite)
        cared = {
            eid
            for m in polite
            for e in m["events"]
            if "request_care" in e["functions"]
            for eid in e["action_ids"]
        }
        for m in a["items"]["136-137"]:
            for e in m["events"]:
                if e["act"] == "request":
                    counts["unknown" if not complete else "yes" if e["id"] in cared else "no"] += 1
    known = counts["yes"] + counts["no"]
    return {
        "counts": counts,
        "request_candidates": sum(counts.values()),
        "request_denominator": known,
        "request_politeness_rate": ratio(counts["yes"], known),
        "unknown_fraction": ratio(counts["unknown"], sum(counts.values())),
        "action_detection_unknown_messages": detection_unknown,
    }


def refusal_structures(entries):
    states, structures, kinds = Counter(), Counter(), Counter()
    records = []
    for entry in entries:
        for event in entry["events"]:
            states[event["buffering"]] += 1
            if event["buffering"] == "unknown":
                continue
            linked = [
                {"kind": b["kind"], "order": order_of(b["evidence"], event["evidence"], entry["row"])}
                for b in event["buffers"]
            ]
            kinds.update(b["kind"] for b in linked)
            orders = sorted({b["order"] for b in linked})
            structure = "+".join(orders) or "unbuffered"
            structures[structure] += 1
            records.append(
                {
                    "sample_id": entry["sample_id"],
                    "event_id": event["id"],
                    "structure": structure,
                    "buffers": linked,
                }
            )
    denominator = states["yes"] + states["no"]
    return {
        "counts": states,
        "refusal_candidates": sum(states.values()),
        "refusal_denominator": denominator,
        "buffer_rate": ratio(states["yes"], denominator),
        "unknown_fraction": ratio(states["unknown"], sum(states.values())),
        "structure_counts": structures,
        "buffer_kind_counts": kinds,
        "records": records,
    }


def rebuttal_structures(records, annotations):
    event_orders, window_counts = Counter(), Counter()
    details = []
    candidates = detection_unknown = 0
    for row, a in zip(records, annotations, strict=True):
        messages = a["items"]["145"]
        events = [e for m in messages for e in m["events"]]
        complete = all(m["state"] in {"present", "na"} for m in messages)
        detection_unknown += not complete
        if not events:
            continue
        candidates += 1
        all_orders = []
        for e in events:
            if e["alignment"] == "unknown":
                complete = False
                continue
            linked = [
                {"role": c["role"], "order": order_of(c["evidence"], e["evidence"], row)}
                for c in e["supports"]
            ]
            orders = sorted({c["order"] for c in linked})
            event_orders["+".join(orders) or "rebuttal_only"] += 1
            all_orders.extend(orders)
            details.append({"sample_id": row["sample_id"], "event_id": e["id"], "supports": linked})
        window_counts[
            "unknown" if not complete else "prior_support" if "before" in all_orders else "no_prior_support"
        ] += 1
    known = window_counts["prior_support"] + window_counts["no_prior_support"]
    return {
        "window_counts": window_counts,
        "rebuttal_window_candidates": candidates,
        "window_denominator": known,
        "window_prior_support_rate": ratio(window_counts["prior_support"], known),
        "unknown_fraction": ratio(window_counts["unknown"], candidates),
        "detection_unknown_windows": detection_unknown,
        "event_order_counts": event_orders,
        "records": details,
    }


def summarize_semantic(records, annotations, raw):
    result = {}
    for item in ITEMS:
        entries = annotated_entries(records, annotations, raw, item)
        states = Counter(e["state"] for e in entries)
        valid = [e for e in entries if e["state"] in {"present", "absent"}]
        all_events = [e for m in valid for e in m["events"]]
        words = sum(m["words"] for m in valid)
        positions, position_records = Counter(), []
        for m in valid:
            text = next(t["text"] for t in m["row"]["messages"] if t["message_id"] == m["message_id"])
            for e in m["events"]:
                pos = relative_position(e["evidence"], text)
                positions[pos["bin"]] += 1
                position_records.append(
                    dict(pos, sample_id=m["sample_id"], message_id=m["message_id"], event_id=e["id"])
                )
        metric = {
            "source_ids": SOURCE_IDS[item],
            "protocol": PROTOCOL_VERSION,
            "n_messages": len(entries),
            "states": states,
            "valid_messages": len(valid),
            "coverage": ratio(states["present"] + states["absent"] + states["na"], len(entries)),
            "unknown_rate": ratio(states["unknown"], len(entries)),
            "failure_rate": ratio(states["failure"], len(entries)),
            "event_count": len(all_events),
            "word_denominator": words,
            "per_100_words": DEFAULT_FREQUENCY_SCALE * len(all_events) / words if words else None,
            "forms": Counter(e["evidence"]["quote"] for e in all_events),
            "positions": positions,
            "position_records": position_records,
            "initial_failure_windows": sum(item in a["initial_failures"] for a in annotations),
            "remaining_failure_windows": sum(item in a["remaining_failures"] for a in annotations),
        }
        if item in CONDITIONAL:
            metric["applicability_denominator"] = states["present"] + states["na"]
            metric["applicability_rate"] = ratio(states["present"], states["present"] + states["na"])
            observed = [e for e in entries if e["state"] in {"present", "na"}]
            metric["by_length"] = {
                "applicability_rate": length_strata(
                    (e["nonspace_characters"], e["state"] == "present") for e in observed
                )
            }
            metric["reliability"] = {
                "applicability_rate": with_missing(
                    window_reliability((e["sample_id"], e["state"] == "present") for e in observed),
                    *missing_entries(entries),
                )
            }
        else:
            metric["message_rate"] = ratio(states["present"], len(valid))
            metric["by_length"] = {
                "message_rate": length_strata(
                    (e["nonspace_characters"], e["state"] == "present") for e in valid
                )
            }
            metric["reliability"] = {
                "message_rate": with_missing(
                    window_reliability((e["sample_id"], e["state"] == "present") for e in valid),
                    *missing_entries(entries),
                )
            }
        metric["attributes"] = {
            field: labels_summary(
                entries,
                lambda e, f=field: e[f] if f in MULTI_FIELDS else [e[f]],
                "none" if field == "level" else None,
            )
            for field in LABEL_FIELDS[item]
        }
        if item == "120-127":
            metric["strategy_cooccurrence"] = labels_summary(
                entries, lambda e: ["+".join(sorted(e["strategies"]))]
            )
            metric["mitigation_by_strategy"] = {
                label: Counter(e["mitigation"] for e in all_events if label in e["strategies"])
                for label in ENUMS["strategies"]
            }
        if item == "128-131":
            metric["requests"] = conditional_requests(annotations)
        if item == "132-134":
            metric["relation_densities"] = {
                label: DEFAULT_FREQUENCY_SCALE * metric["attributes"]["relations"]["counts"][label] / words
                if words
                else None
                for label in ENUMS["relations"]
            }
        if item == "136-137":
            metric["choice_by_act"] = {
                label: labels_summary(
                    [
                        dict(m, events=[e for e in m["events"] if e["act"] == label])
                        for m in valid
                        if any(e["act"] == label for e in m["events"])
                    ],
                    lambda e: [e["choice"]],
                )
                for label in ENUMS["act"]
            }
            metric["condition_by_choice"] = Counter(
                ("conditional" if e["condition"] else "unconditional") + ":" + e["choice"] for e in all_events
            )
        if item == "138":
            metric["joint_axes"] = labels_summary(
                entries,
                lambda e: [
                    "unknown"
                    if "unknown" in (e["insistence"], e["directness"])
                    else e["insistence"] + ":" + e["directness"]
                ],
            )
        if item == "139":
            metric["refusals"] = refusal_structures(entries)
        if item == "140-144":
            metric["function_positions"] = Counter(
                label
                + ":"
                + relative_position(
                    e["evidence"],
                    next(t["text"] for t in m["row"]["messages"] if t["message_id"] == m["message_id"]),
                )["bin"]
                for m in valid
                for e in m["events"]
                for label in e["metadiscourse"]
            )
        if item == "145":
            metric["rebuttals"] = rebuttal_structures(records, annotations)
        if item == "146-150":
            metric["method_cooccurrence"] = labels_summary(
                entries, lambda e: ["+".join(sorted(e["methods"]))]
            )
            observations = {(o["sample_id"], o["message_id"]): o for o in raw["messages"]}
            metric["form_attributes"] = [
                {
                    "sample_id": m["sample_id"],
                    "event_id": e["id"],
                    "methods": e["methods"],
                    **{
                        key: [
                            v
                            for v in observations[(m["sample_id"], m["message_id"])][key]
                            if v["start"] < e["evidence"]["end"] and v["end"] > e["evidence"]["start"]
                        ]
                        for key in ("repetitions", "punctuation", "capitalization")
                    },
                }
                for m in valid
                for e in m["events"]
            ]
        result[item] = metric
    return result


def extract(corpora, parsed, judge, output, workers, qa_runner=None):
    raw, semantic, annotations = {}, {}, {}
    for name, records in corpora.items():
        dest = output / name
        dest.mkdir()
        raw[name] = summarize(records, parsed)
        write_json(dest / "pragmatic_style.json", raw[name])
        write_json(dest / "records.json", records)
    if judge:
        print(f"Annotating pragmatic functions for {distinct_records(corpora)} windows", flush=True)
        annotations = annotate_corpora(
            corpora, lambda row: annotate(row, parsed, judge), workers, judge=judge
        )
        for name, records in corpora.items():
            semantic[name] = summarize_semantic(records, annotations[name], raw[name])
            write_json(output / name / "annotations.json", annotations[name])
            write_json(output / name / "semantic.json", semantic[name])
            if qa_runner and name == PRIMARY_CORPUS:
                write_json(output / name / "qa.json", qa_runner(records, annotations[name]))
    return semantic


def extraction_coverage(semantic, judge):
    return {
        item: {
            "state": "extracted" if judge else "partial" if item == "146-150" else "not_run",
            "source_ids": SOURCE_IDS[item],
            "protocol": PROTOCOL_VERSION,
            "corpora": {name: value[item]["states"] for name, value in semantic.items()},
        }
        for item in ITEMS
    }


def protocol():
    return {
        "source_ids": SOURCE_IDS,
        "label_definitions": ENUMS,
        "message_denominator": "present+absent per function; empty NA, unknown and failure excluded and reported; conditional objects report applicability separately",
        "attribute_message_denominator": "present+absent messages with no unknown label for that attribute; conditional items use applicable messages only",
        "conditional_denominators": {
            "certainty": "explicit certainty expressions only; not a proportion of all propositions",
            "request_politeness": "actual requests with complete window-level politeness annotation",
            "refusal_buffering": "actual refusals with known buffer relation",
            "prior_support": "windows with rebuttals and complete known alignment",
        },
        "word_denominator": "LTP CWS/POS filtered by shared lexical_choice.word_token; same determinate-message subset as numerator",
        "frequency_scale": DEFAULT_FREQUENCY_SCALE,
        "position": "midpoint of exact evidence in trimmed message, equal thirds; raw offsets retained",
        "cross_message_relations": "only within an observed window; refusal/support order from original message order and offsets, never across sampled windows",
        "surface_reuse": [
            "orthographic_style.punctuation_events (081)",
            "orthographic_style.capitalization (093)",
            "paralinguistic_style.repetitions (106-107,109)",
        ],
        "surface_policy": "all original forms retained, no semantic prefilter; --no-llm provides no pragmatic function judgments",
        "categories": "fixed function definitions; no trigger-word lists; uncovered functions retain the other label",
        "shared_evidence": "typed event references within one annotation; no summed overall style score",
    }
