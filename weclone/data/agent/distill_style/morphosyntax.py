"""Morphosyntax: POS density, positions, nominalization and aspect/time usage."""

from weclone.prompts.distill_style import (
    MORPHOSYNTAX_ANNOTATION_PROMPT as ANNOTATION_PROMPT,
)

import re
from collections import Counter

from .data import PRIMARY_CORPUS, write_json
from .extraction import annotate_corpora, distinct_records, locate
from .lexical_choice import CONTENT_POS, FUNCTION_POS, lexical_token, word_token
from .stats import by_length, message_texts, missing_messages, ratio, reliability, with_missing

PROTOCOL_VERSION = "morphosyntax-v2"
SCOPE = "Morphosyntax 025-036"
DEFAULT_POSITION_BINS = 5
ITEMS = {
    "026": ("nominalized_event",),
    "036": (
        "perfective",
        "experiential",
        "progressive",
        "durative",
        "habitual",
        "prospective",
        "change_of_state",
        "past_time",
        "present_time",
        "future_time",
    ),
}
DENSITY_POS = {"027": {"a"}, "028": {"d"}, "029": {"v"}, "030": {"r"}}
NOUN_POS = {"n", "nd", "nh", "ni", "nl", "ns", "nt", "nz"}
POS_GROUPS = {
    "a": "adjective",
    "d": "adverb",
    "v": "verb",
    "r": "pronoun",
    "p": "preposition",
    "c": "conjunction",
    "u": "particle",
    "e": "interjection",
    "o": "onomatopoeia",
    "m": "numeral",
    "q": "classifier",
    "wp": "punctuation",
    "ws": "foreign_word",
}
SENTENCE_END = re.compile(r"[。！？!?；;\n]")


def sentence_units(row, parsed):
    """Operational sentence segments with original offsets; no inferred semantic clauses."""
    for message in row["messages"]:
        text, current, previous_end, index = message["text"], [], 0, 0
        p = parsed[text]
        for token, pos, start, end in zip(p["cws"], p["pos"], p["starts"], p["ends"], strict=True):
            if "\n" in text[previous_end:start] and current:
                yield message["message_id"], index, current
                current, index = [], index + 1
            current.append({"text": token, "pos": pos, "start": start, "end": end})
            previous_end = end
            if pos == "wp" and (SENTENCE_END.search(token) or token == "."):
                yield message["message_id"], index, current
                current, index = [], index + 1
        if current:
            yield message["message_id"], index, current


def probabilities(counts):
    n = sum(counts.values())
    return {key: value / n for key, value in counts.items()} if n else {}


def add_positions(counter, labels, bins=DEFAULT_POSITION_BINS):
    for i, label in enumerate(labels):
        bucket = min(bins - 1, ((2 * i + 1) * bins) // (2 * len(labels)))
        counter.setdefault(label, Counter())[str(bucket)] += 1


def summarize(records, parsed):
    raw_pos, word_pos, first, last = (Counter() for _ in range(4))
    positions = {}
    observations = []
    n_messages = empty_messages = 0
    for row in records:
        for message in row["messages"]:
            tags = parsed[message["text"]]["pos"]
            n_messages += 1
            empty_messages += not tags
            raw_pos.update(tags)
        for message_id, sentence_index, tokens in sentence_units(row, parsed):
            words = [t for t in tokens if lexical_token(t["text"], t["pos"])]
            tags = [t["pos"] for t in words]
            counts = Counter(tags)
            n = len(tags)
            functions = sum(counts[t] for t in FUNCTION_POS)
            content = sum(counts[t] for t in CONTENT_POS)
            front = [tag for i, tag in enumerate(tags) if 5 * (2 * i + 1) <= 3 * n]
            back = [tag for i, tag in enumerate(tags) if 5 * (2 * i + 1) >= 7 * n]
            word_pos.update(tags)
            add_positions(positions, tags)
            if tags:
                first[tags[0]] += 1
                last[tags[-1]] += 1
            observations.append(
                {
                    "sample_id": row["sample_id"],
                    "message_id": message_id,
                    "sentence_index": sentence_index,
                    "start": tokens[0]["start"],
                    "end": tokens[-1]["end"],
                    "word_tokens": n,
                    "function_count": functions,
                    "content_count": content,
                    "other_count": n - functions - content,
                    "densities": {
                        **{
                            item: ratio(sum(counts[t] for t in tags_set), n)
                            for item, tags_set in DENSITY_POS.items()
                        },
                        "031": ratio(content, functions),
                        "032": ratio(content, n),
                    },
                    "front_tokens": len(front),
                    "back_tokens": len(back),
                    "front_function_count": sum(t in FUNCTION_POS for t in front),
                    "back_function_count": sum(t in FUNCTION_POS for t in back),
                    "front_density": ratio(sum(t in FUNCTION_POS for t in front), len(front)),
                    "back_density": ratio(sum(t in FUNCTION_POS for t in back), len(back)),
                }
            )
    total = sum(word_pos.values())
    functional = sum(word_pos[t] for t in FUNCTION_POS)
    content = sum(word_pos[t] for t in CONTENT_POS)
    groups = Counter()
    for tag, count in word_pos.items():
        groups["noun" if tag in NOUN_POS else POS_GROUPS.get(tag, "other")] += count
    metrics = {
        "025,027-030": {
            "counts": word_pos,
            "rates": probabilities(word_pos),
            "total_tokens": total,
            "raw_pos_counts_including_punctuation": raw_pos,
            "denominator": "non-punctuation word tokens",
            "densities": {},
            "group_counts": groups,
            "group_rates": probabilities(groups),
            "word_pos_counts": word_pos,
            "word_pos_rates": probabilities(word_pos),
        }
    }
    for item, tags in DENSITY_POS.items():
        count = sum(word_pos[t] for t in tags)
        metrics["025,027-030"]["densities"][item] = {
            "count": count,
            "word_tokens": total,
            "density": ratio(count, total),
        }
    metrics["031"] = {
        "counts": {"content": content, "function": functional, "other": total - content - functional},
        "word_tokens": total,
        "content_rate": ratio(content, total),
        "function_rate": ratio(functional, total),
        "content_to_function_ratio": ratio(content, functional),
    }
    metrics["032"] = {"count": content, "word_tokens": total, "density": ratio(content, total)}
    metrics["033"] = {
        "bins": DEFAULT_POSITION_BINS,
        "counts": positions,
        "probabilities": {tag: probabilities(c) for tag, c in positions.items()},
    }
    metrics["034"] = {
        "first": first,
        "last": last,
        "n_valid_sentences": sum(first.values()),
        "first_rates": probabilities(first),
        "last_rates": probabilities(last),
    }
    metrics["035"] = {}
    for side in ("front", "back"):
        count = sum(o[f"{side}_function_count"] for o in observations)
        denominator = sum(o[f"{side}_tokens"] for o in observations)
        metrics["035"][side] = {
            "count": count,
            "tokens": denominator,
            "density": ratio(count, denominator),
            "n_valid_sentences": sum(o[f"{side}_tokens"] > 0 for o in observations),
        }
    return {
        "n_windows": len(records),
        "n_messages": n_messages,
        "empty_messages": empty_messages,
        "n_sentences": len(observations),
        "n_word_sentences": sum(o["word_tokens"] > 0 for o in observations),
        "metrics": metrics,
        "sentences": observations,
    }


def validate_evidence(event, row):
    result = dict(event)
    result["evidence"] = locate(event["evidence"], row["messages"])
    result["basis"] = locate(event["basis"], row["messages"])
    e, basis = result["evidence"], result["basis"]
    if e["message_id"] != basis["message_id"] or not basis["start"] <= e["start"] < e["end"] <= basis["end"]:
        raise ValueError("Grammatical basis must contain target evidence in the same message")
    return result


def validate_item(item, value, row):
    state, events, reason = value["state"], value["events"], value["reason"]
    if state not in ("present", "absent", "unknown", "na") or not isinstance(events, list):
        raise ValueError("Invalid state, events or reason")
    if (state == "present") != bool(events):
        raise ValueError("Inconsistent state/events/reason")
    validated, seen = [], set()
    for event in events:
        if event["label"] not in ITEMS[item]:
            raise ValueError("Invalid morphology label")
        event = validate_evidence(event, row)
        e = event["evidence"]
        key = (event["label"], e["message_id"], e["start"], e["end"])
        if key in seen:
            raise ValueError("Duplicate grammatical event")
        seen.add(key)
        validated.append(event)
    result = {"state": state, "events": validated, "reason": reason}
    if item == "036":
        other = value["other_uses"]
        if not isinstance(other, list) or (state in ("unknown", "na") and other):
            raise ValueError("Invalid other_uses")
        result["other_uses"] = [validate_evidence(event, row) for event in other]
        keys = [
            (e["evidence"]["message_id"], e["evidence"]["start"], e["evidence"]["end"])
            for e in result["other_uses"]
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate non-temporal use")
    return result


def validate_response(response, row):
    data = response.get("data", {})
    direct = isinstance(data, dict) and set(data) == set(ITEMS)
    values = data if direct else data.get("items", {}) if isinstance(data, dict) else {}
    result = {}
    for item in ITEMS:
        try:
            if response["state"] != "ok":
                raise ValueError(response.get("error", "Call failed"))
            result[item] = validate_item(item, values[item], row)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            result[item] = {"state": "failure", "events": [], "reason": str(exc)}
    return result, direct


def annotate(row, parsed, judge, refresh=False):
    judge = judge.for_record(row)
    payload = {"context": row["context"], "messages": row["messages"]}
    response = judge.call(ANNOTATION_PROMPT, payload, refresh=refresh)
    result, direct = validate_response(response, row)
    failures = {item: value["reason"] for item, value in result.items() if value["state"] == "failure"}
    repair = None
    item_keys = dict.fromkeys(ITEMS, response["cache_key"])
    if failures:
        repair = judge.call(
            ANNOTATION_PROMPT,
            payload,
            refresh=refresh,
            attempt=1,
        )
        repaired, _ = validate_response(repair, row)
        for item in failures:
            result[item] = repaired[item]
            item_keys[item] = repair["cache_key"]
    if any(value["state"] == "failure" for value in result.values()):
        judge.invalidate(response["cache_key"], repair["cache_key"] if repair else None)
    return {
        "sample_id": row["sample_id"],
        "items": result,
        "cache_key": response["cache_key"],
        "normalized_direct_item_map": direct,
        "initial_failures": failures,
        "repair_cache_key": repair["cache_key"] if repair else None,
        "item_cache_keys": item_keys,
    }


def summarize_semantic(records, annotations, parsed):
    by_id = {r["sample_id"]: r for r in records}
    result = {}
    for item in ITEMS:
        states, labels, expressions, other_uses = (Counter() for _ in range(4))
        words = messages = 0
        hits, texts = set(), {}
        for annotation in annotations:
            value = annotation["items"][item]
            states[value["state"]] += 1
            if value["state"] not in ("present", "absent"):
                continue
            row = by_id[annotation["sample_id"]]
            messages += sum(bool(m["text"].strip()) for m in row["messages"])
            texts.update(message_texts([row]))
            hits.update((row["sample_id"], e["evidence"]["message_id"]) for e in value["events"])
            words += sum(
                sum(
                    word_token(t, p)
                    for t, p in zip(parsed[m["text"]]["cws"], parsed[m["text"]]["pos"], strict=True)
                )
                for m in row["messages"]
            )
            labels.update(e["label"] for e in value["events"])
            expressions.update(e["evidence"]["quote"] for e in value["events"])
            other_uses.update(e["evidence"]["quote"] for e in value.get("other_uses", []))
        n, valid = len(annotations), states["present"] + states["absent"]
        unknown, failure = missing_messages(
            (by_id[a["sample_id"]], a["items"][item]["state"]) for a in annotations
        )
        result[item] = {
            "states": {s: states[s] for s in ("present", "absent", "unknown", "na", "failure")},
            "valid_messages": messages,
            "hit_messages": len(hits),
            "message_occurrence_rate": ratio(len(hits), messages),
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
            "labels": labels,
            "expressions": expressions,
            "eligible_word_tokens": words,
            "events_per_100_words": 100 * sum(labels.values()) / words if words else None,
            "initial_failure_count": sum(item in a.get("initial_failures", {}) for a in annotations),
            "repaired_count": sum(
                item in a.get("initial_failures", {}) and a["items"][item]["state"] != "failure"
                for a in annotations
            ),
        }
        if item == "036":
            result[item]["other_uses"] = other_uses
            result[item]["aspect_labels"] = {
                label: count for label, count in labels.items() if not label.endswith("_time")
            }
            result[item]["time_labels"] = {
                label: count for label, count in labels.items() if label.endswith("_time")
            }
    return result


def extract(corpora, parsed, judge, output, workers, qa_runner=None):
    annotations, semantic, summaries = {}, {}, {}
    if judge:
        print(f"Annotating morphology for {distinct_records(corpora)} windows", flush=True)
        annotations = annotate_corpora(
            corpora, lambda row: annotate(row, parsed, judge), workers, judge=judge
        )
    for name, records in corpora.items():
        dest = output / name
        dest.mkdir()
        summaries[name] = summarize(records, parsed)
        write_json(dest / "morphosyntax.json", summaries[name])
        write_json(dest / "records.json", records)
        if judge:
            semantic[name] = summarize_semantic(records, annotations[name], parsed)
            write_json(dest / "annotations.json", annotations[name])
            write_json(dest / "semantic.json", semantic[name])
            if qa_runner and name == PRIMARY_CORPUS:
                write_json(dest / "qa.json", qa_runner(records, annotations[name]))
    return semantic


def extraction_coverage(semantic, judge):
    coverage = {f"{i:03}": {"state": "extracted"} for i in range(25, 37)}
    for item in ITEMS:
        coverage[item] = {
            "state": "extracted" if judge else "not_run",
            "corpora": {name: value[item]["states"] for name, value in semantic.items()},
        }
    for key in ("025", *DENSITY_POS):
        coverage.pop(key, None)
    coverage["025,027-030"] = {"state": "extracted"}
    return coverage


def protocol():
    return {
        "pos_denominator": "Non-punctuation word tokens; raw POS counts retained only as diagnostics",
        "density_denominator": "word_token: non-wp with a Unicode letter or number, vocal strings (lexical_choice.vocal_token) excluded",
        "function_pos": sorted(FUNCTION_POS),
        "content_pos": sorted(CONTENT_POS),
        "sentence_boundary": "wp containing 。！？!?；; or exact period; raw newline; message end",
        "position": "(i+0.5)/L over non-punctuation words in each operational sentence",
        "position_bins": DEFAULT_POSITION_BINS,
        "front": "position <= 0.3",
        "back": "position >= 0.7",
        "single_word_policy": "retained at position 0.5; front/back denominators zero -> null",
        "semantic_labels": ITEMS,
    }
