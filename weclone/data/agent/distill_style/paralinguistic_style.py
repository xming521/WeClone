"""Paralinguistic style: merged textual attributes and sequence-level emoji metrics."""

from weclone.prompts.distill_style import (
    PARALINGUISTIC_STYLE_ANNOTATION_PROMPT as ANNOTATION_PROMPT,
)

import hashlib
import re
import unicodedata
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import regex

from .data import PRIMARY_CORPUS, bubble_spans, write_json
from .extraction import annotate_corpora, distinct_records, locate
from .lexical_choice import vocal_token
from .morphosyntax import probabilities
from .stats import by_length, missing_messages, ratio, reliability, with_missing

PROTOCOL_VERSION = "paralinguistic-style-v1"
SCOPE = "Paralinguistic Style: five merged text measures plus Emoji 111-116"
DEFAULT_FREQUENCY_SCALE = 100
DEFAULT_EMOJI_DATA = Path(__file__).parent / "resources/emoji-test-17.0.txt"
MEASURES = ("097-098", "099-105", "106-107,109", "108", "110", "111", "112", "113", "114", "115", "116")
SOURCE_IDS = {
    "097-098": [97, 98],
    "099-105": list(range(99, 106)),
    "106-107,109": [106, 107, 109],
    **{str(i): [i] for i in [108, 110, *range(111, 117)]},
}
# Closed grammatical classes, checked on chat: 哦/嘞/哈/捏 end clauses as particles, LTP tags laughter e/o.
PARTICLES = frozenset(
    "吗 呢 吧 啊 嘛 么 呀 哇 哪 呐 啦 呗 哟 呦 喽 咯 嚜 罢了 而已 来着 哦 噢 喔 嘞 哈 捏".split()
)
INTERJECTIONS = frozenset(
    "啊 嗯 哦 噢 喔 哎 唉 欸 诶 嗐 咦 哟 呦 哇 喂 嗨 呃 额 啧 嘶 哼 呸 哎呀 哎哟 哎呦 唉呀 嗯哼 嗯呐".split()
)
AMBIGUOUS_PARTICLES = {"了", "的"}
NON_MODAL_AUXILIARIES = {"过", "着", "地", "得"}
SOUND = (
    "laughter",
    "crying_voice",
    "hesitation",
    "exclamation",
    "human_vocalization",
    "nonhuman_sound",
    "conventional_reaction",
    "none",
    "other",
    "unknown",
)
FUNCTIONS = (
    "affiliation",
    "acknowledgment",
    "softening",
    "amusement",
    "distress",
    "hesitation",
    "surprise",
    "teasing_irony",
    "emphasis",
    "rhythm",
    "depiction",
    "other",
    "unknown",
)
REPEAT_USES = ("word_formation", "grammatical", "sound_simulation", "emphasis", "rhythm", "other", "unknown")
# 108 counts the textual events that imitate a sound, from the same annotation as 099-105.
SIMULATED_SOUNDS = (
    "laughter",
    "crying_voice",
    "hesitation",
    "exclamation",
    "human_vocalization",
    "nonhuman_sound",
)
ITEMS = {
    "099-105": SOUND,
    "108": SIMULATED_SOUNDS,
    "106-107,109": REPEAT_USES,
    "110": ("elongation",),
    "112": FUNCTIONS,
}


@lru_cache(maxsize=1)
def emoji_table():
    table, group, subgroup = {}, "", ""
    for line in DEFAULT_EMOJI_DATA.read_text().splitlines():
        if line.startswith("# group:"):
            group = line.split(":", 1)[1].strip()
        elif line.startswith("# subgroup:"):
            subgroup = line.split(":", 1)[1].strip()
        elif line and not line.startswith("#"):
            codes, rest = line.split(";", 1)
            status, description = rest.split("#", 1)
            sequence = "".join(chr(int(c, 16)) for c in codes.split())
            table[sequence] = {
                "qualification": status.strip(),
                "category": group,
                "subcategory": subgroup,
                "name": re.sub(r"^\S+\s+E[0-9.]+\s+", "", description.strip()),
            }
    return table


def span_event(text, start, end, **attributes):
    return {"start": start, "end": end, "quote": text[start:end], **attributes}


def writing(text):
    types = set()
    for c in text:
        if c.isspace():
            continue
        if regex.fullmatch(r"\p{Han}", c):
            types.add("han")
        elif regex.fullmatch(r"\p{Latin}", c):
            types.add("latin")
        elif c.isdecimal():
            types.add("digits")
        elif unicodedata.category(c).startswith(("P", "S")):
            types.add("symbols")
        else:
            types.add("other")
    return next(iter(types)) if len(types) == 1 else "mixed" if types else "whitespace"


def repetitions(text):
    """Maximal integral tandem spans, shortest unit; retain longest nonoverlapping spans."""
    candidates = {}
    n = len(text)
    for period in range(1, n // 2 + 1):
        lcp = 0
        for i in range(n - period - 1, -1, -1):
            lcp = lcp + 1 if text[i] == text[i + period] else 0
            if lcp < period:
                continue
            if i and text[i - 1] == text[i + period - 1]:
                continue
            count = lcp // period + 1
            end = i + count * period
            key = (i, end)
            if key not in candidates or period < len(candidates[key]["unit"]):
                candidates[key] = span_event(
                    text,
                    i,
                    end,
                    unit=text[i : i + period],
                    repetitions=count,
                    characters=end - i,
                    writing=writing(text[i:end]),
                )
    accepted = []
    for candidate in sorted(
        candidates.values(), key=lambda e: (-(e["end"] - e["start"]), e["start"], len(e["unit"]))
    ):
        if not any(candidate["start"] < e["end"] and e["start"] < candidate["end"] for e in accepted):
            accepted.append(candidate)
    return sorted(accepted, key=lambda e: e["start"])


def unit_form(word):
    for length in range(1, len(word) + 1):
        if len(word) % length == 0 and word == word[:length] * (len(word) // length):
            return word[:length]
    return word


def clause_final(i, p, text):
    end = p["ends"][i]
    for j in range(i + 1, len(p["cws"])):
        if "\n" in text[end : p["starts"][j]] or "\r" in text[end : p["starts"][j]]:
            return True
        if p["pos"][j] == "wp" and re.search(r"[,，、。！？!?；;.]", p["cws"][j]):
            return True
        if all(unicodedata.category(c).startswith(("P", "S", "M", "C")) for c in p["cws"][j]):
            end = p["ends"][j]
            continue
        if p["pos"][j] != "wp" and not (
            p["pos"][j] == "u" and unit_form(p["cws"][j]) in PARTICLES | AMBIGUOUS_PARTICLES
        ):
            return False
        end = p["ends"][j]
    return True


def grammar_events(msg, p):
    """Modal particles and interjections from closed classes; vocal strings are left to 099-105."""
    result = []
    for i, (word, tag, start, end) in enumerate(zip(p["cws"], p["pos"], p["starts"], p["ends"], strict=True)):
        if vocal_token(word):
            continue
        label = "interjection" if tag not in {"u", "v"} and unit_form(word) in INTERJECTIONS else None
        if (
            tag == "u"
            and p["dep"]["label"][i] == "RAD"
            and 0 < p["dep"]["head"][i] <= len(p["pos"])
            and p["dep"]["head"][i] != i + 1
            and clause_final(i, p, msg["text"])
        ):
            form = unit_form(word)
            if form in PARTICLES:
                label = "modal_particle"
            elif form not in NON_MODAL_AUXILIARIES:
                label = "unknown_particle"
        if label:
            result.append(
                span_event(
                    msg["text"],
                    start,
                    end,
                    grammar=label,
                    pos=tag,
                    token_index=i,
                    dep_label=p["dep"]["label"][i],
                    head=p["dep"]["head"][i],
                )
            )
    return result


def emoji_events(msg):
    text, table = msg["text"], emoji_table()
    unicode_events, unknown, platform = [], [], []
    for m in regex.finditer(r"\X", text):
        if m.group() in table:
            unicode_events.append(span_event(text, m.start(), m.end(), kind="unicode", **table[m.group()]))
        elif "\ufe0e" not in m.group() and regex.search(
            r"\p{Extended_Pictographic}|\p{Regional_Indicator}|\u20e3", m.group()
        ):
            unknown.append(span_event(text, m.start(), m.end(), kind="unrecognized_sequence"))
    for value in msg.get("platform_emojis", []):
        start, end, identity = value["start"], value["end"], value["emoji_id"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(text)
            or not isinstance(identity, str)
            or not identity
        ):
            raise ValueError("Invalid platform emoji metadata")
        platform.append(span_event(text, start, end, kind="platform", emoji_id=identity))
    if msg.get("message_type") == "emoji" and not platform:
        identity = msg.get("emoji_id")
        if isinstance(identity, str) and identity:
            platform.append(span_event(text, 0, len(text), kind="platform", emoji_id=identity))
    for i, e in enumerate(platform):
        if any(e["start"] < f["end"] and f["start"] < e["end"] for f in platform[:i]):
            raise ValueError("Overlapping platform emoji metadata")
    unicode_events = [
        e
        for e in unicode_events
        if not any(e["start"] < f["end"] and f["start"] < e["end"] for f in platform)
    ]
    labels = [
        span_event(text, m.start(), m.end(), kind="unverified_text_label")
        for m in re.finditer(r"\[[^\[\]\r\n]+\]", text)
        if not any(m.start() < f["end"] and f["start"] < m.end() for f in platform)
    ]
    return sorted(unicode_events + platform, key=lambda e: e["start"]), unknown, labels


def anchors(row, parsed):
    grammar, repeats, emojis, unknown, labels = [], [], [], [], []
    for msg in row["messages"]:
        grammar.extend(
            dict(e, message_id=msg["message_id"]) for e in grammar_events(msg, parsed[msg["text"]])
        )
        repeats.extend(
            dict(
                e,
                message_id=msg["message_id"],
                anchor_id=f"repeat:{msg['message_id']}:{e['start']}:{e['end']}",
            )
            for e in repetitions(msg["text"])
        )
        known, bad, unverified = emoji_events(msg)
        emojis.extend(
            dict(
                e,
                message_id=msg["message_id"],
                anchor_id=f"emoji:{msg['message_id']}:{e['start']}:{e['end']}",
            )
            for e in known
        )
        unknown.extend(dict(e, message_id=msg["message_id"]) for e in bad)
        labels.extend(dict(e, message_id=msg["message_id"]) for e in unverified)
    return {
        "grammar": grammar,
        "repeats": repeats,
        "emoji": emojis,
        "unrecognized_emoji": unknown,
        "unverified_text_labels": labels,
    }


def position(e, text):
    start, end = len(text) - len(text.lstrip()), len(text.rstrip())
    if e["start"] <= start and e["end"] >= end:
        return "whole_message"
    if e["start"] <= start:
        return "start"
    if e["end"] >= end:
        return "end"
    return "middle"


def aggregate(records, groups):
    texts = {(r["sample_id"], m["message_id"]): m["text"] for r in records for m in r["messages"]}
    eligible = {k for k, t in texts.items() if t.strip()}
    hit = {(r["sample_id"], e["message_id"]) for r, es in zip(records, groups, strict=True) for e in es}
    chars = sum(len(t) for t in texts.values())
    forms = Counter(e["quote"] for es in groups for e in es)
    positions = Counter(
        position(e, texts[(r["sample_id"], e["message_id"])])
        for r, es in zip(records, groups, strict=True)
        for e in es
    )
    return {
        "count": sum(map(len, groups)),
        "forms": forms,
        "positions": positions,
        "n_messages": len(texts),
        "nonempty_messages": len(eligible),
        "matched_messages": len(hit),
        "message_rate": ratio(len(hit), len(texts)),
        "by_length": {"message_rate": by_length(texts, hit)},
        "reliability": {"message_rate": reliability(texts, hit)},
        "characters": chars,
        "per_100_characters": DEFAULT_FREQUENCY_SCALE * sum(map(len, groups)) / chars if chars else None,
        "window_event_counts": [len(es) for es in groups],
    }


def emoji_intervals(row, events):
    messages = row["messages"]
    order = {m["message_id"]: i for i, m in enumerate(messages)}
    offsets, offset = {}, 0
    spans, first = {}, 0
    for m in messages:
        offsets[m["message_id"]] = offset
        offset += len(m["text"])
        spans[m["message_id"]] = (first, bubble_spans(m["text"]))
        first += len(spans[m["message_id"]][1])

    def bubble(e):
        base, pieces = spans[e["message_id"]]
        return base + next(i for i, (s, t) in enumerate(pieces) if s <= e["start"] < t)

    # A dialogue window lists all target turns of one conversation in order, so gaps across turns are real.
    sequential = row["boundary"] == "message" or "dialogue" in row
    result = []
    for a, b in zip(events, events[1:]):
        ai, bi = order[a["message_id"]], order[b["message_id"]]
        if ai != bi and not sequential:
            continue
        seconds = None
        if "timestamp" in messages[ai] and "timestamp" in messages[bi]:
            try:
                ta, tb = [
                    datetime.fromisoformat(messages[i]["timestamp"].replace("Z", "+00:00")) for i in (ai, bi)
                ]
                if ta.tzinfo and tb.tzinfo and tb >= ta:
                    seconds = (tb - ta).total_seconds()
            except (ValueError, TypeError, AttributeError):
                pass
        result.append(
            {
                "from": a["anchor_id"],
                "to": b["anchor_id"],
                "character_gap": offsets[b["message_id"]] + b["start"] - offsets[a["message_id"]] - a["end"],
                "message_index_distance": bi - ai,
                "intervening_messages": max(0, bi - ai - 1),
                "bubble_index_distance": bubble(b) - bubble(a),
                "intervening_bubbles": max(0, bubble(b) - bubble(a) - 1),
                "message_timestamp_delta_seconds": seconds,
            }
        )
    return result


def summarize(records, parsed):
    values = [anchors(r, parsed) for r in records]
    grammar = [
        [e for e in a["grammar"] if e["grammar"] in {"modal_particle", "interjection"}] for a in values
    ]
    repeats = [a["repeats"] for a in values]
    emojis = [a["emoji"] for a in values]
    metrics = {
        "097-098": aggregate(records, grammar),
        "106-107,109": aggregate(records, repeats),
        "111": aggregate(records, emojis),
    }
    metrics["097-098"]["grammar"] = Counter(e["grammar"] for es in grammar for e in es)
    metrics["097-098"]["by_class"] = {
        kind: aggregate(records, [[e for e in es if e["grammar"] == kind] for es in grammar])
        for kind in ("modal_particle", "interjection")
    }
    metrics["097-098"]["unknown_particles"] = sum(
        e["grammar"] == "unknown_particle" for a in values for e in a["grammar"]
    )
    particle_candidates = (
        metrics["097-098"]["grammar"]["modal_particle"] + metrics["097-098"]["unknown_particles"]
    )
    metrics["097-098"]["particle_candidate_count"] = particle_candidates
    metrics["097-098"]["particle_resolution_rate"] = ratio(
        metrics["097-098"]["grammar"]["modal_particle"], particle_candidates
    )
    repeat_metric = metrics["106-107,109"]
    repeat_metric.update(
        units=Counter(e["unit"] for es in repeats for e in es),
        writing=Counter(e["writing"] for es in repeats for e in es),
        repetitions=[e["repetitions"] for es in repeats for e in es],
        lengths=[e["characters"] for es in repeats for e in es],
    )
    repeated_grammar = []
    for a, es in zip(values, repeats, strict=True):
        selected = []
        for e in es:
            covered = {
                j
                for g in a["grammar"]
                if g["grammar"] in {"modal_particle", "interjection"} and g["message_id"] == e["message_id"]
                for j in range(g["start"], g["end"])
            }
            if set(range(e["start"], e["end"])) <= covered:
                selected.append(e)
        repeated_grammar.append(selected)
    repeat_metric["particle_interjection_subset"] = aggregate(records, repeated_grammar)
    repeat_metric["particle_interjection_subset"]["repetitions"] = [
        e["repetitions"] for es in repeated_grammar for e in es
    ]
    metrics["111"].update(
        kinds=Counter(e["kind"] for es in emojis for e in es),
        platform_ids=Counter(e["emoji_id"] for es in emojis for e in es if e["kind"] == "platform"),
        unrecognized_sequences=sum(len(a["unrecognized_emoji"]) for a in values),
        unverified_text_labels=sum(len(a["unverified_text_labels"]) for a in values),
    )
    metrics["112"] = {
        "categories": Counter(e["category"] for es in emojis for e in es if e["kind"] == "unicode"),
        "subcategories": Counter(e["subcategory"] for es in emojis for e in es if e["kind"] == "unicode"),
        "qualification": Counter(e["qualification"] for es in emojis for e in es if e["kind"] == "unicode"),
    }
    metrics["113"] = {
        k: metrics["111"][k]
        for k in (
            "count",
            "n_messages",
            "matched_messages",
            "message_rate",
            "by_length",
            "reliability",
            "characters",
            "per_100_characters",
        )
    }
    intervals = [
        dict(g, sample_id=r["sample_id"])
        for r, es in zip(records, emojis, strict=True)
        for g in emoji_intervals(r, es)
    ]
    metrics["114"] = {
        "intervals": intervals,
        "pairs": len(intervals),
        "same_message_pairs": sum(g["message_index_distance"] == 0 for g in intervals),
        "bubble_distance_counts": Counter(str(g["bubble_index_distance"]) for g in intervals),
        "timestamp_pairs": sum(g["message_timestamp_delta_seconds"] is not None for g in intervals),
    }
    metrics["115"] = {
        "positions": metrics["111"]["positions"],
        "rates": probabilities(metrics["111"]["positions"]),
    }
    return {
        "n_windows": len(records),
        "metrics": metrics,
        "records": [
            dict(a, sample_id=r["sample_id"], event_registry=registry(r, a, {}))
            for r, a in zip(records, values, strict=True)
        ],
    }


def labels_valid(values, allowed):
    return (
        isinstance(values, list)
        and bool(values)
        and all(isinstance(x, str) and x in allowed for x in values)
        and len(values) == len(set(values))
        and ("unknown" not in values or len(values) == 1)
    )


def validate_part(key, data, row, raw):
    if key != "text":
        candidates = raw["repeats" if key == "repeat_usage" else "emoji"]
        expected = {e["anchor_id"]: e for e in candidates}
        values = data[key]
        if not isinstance(values, list):
            raise ValueError("Anchor annotations must be an array")
        field, allowed = ("uses", REPEAT_USES) if key == "repeat_usage" else ("functions", FUNCTIONS)
        seen, result = set(), []
        for value in values:
            identity = value["anchor_id"]
            if identity not in expected or identity in seen or (not labels_valid(value[field], allowed)):
                raise ValueError("Invalid anchor or labels")
            seen.add(identity)
            result.append(dict(expected[identity], **{field: value[field]}))
        if seen != set(expected):
            raise ValueError("Every input anchor must be labeled exactly once")
        return result
    value = data["text"]
    state, events, reason = value["state"], value["events"], value["reason"]
    if state not in {"present", "absent", "unknown", "na"} or not isinstance(events, list):
        raise ValueError("Invalid text annotation")
    if (state == "present") != bool(events):
        raise ValueError("Inconsistent text state")
    if state == "na" and any(m["text"].strip() for m in row["messages"]):
        raise ValueError("Nonempty text cannot be NA")
    result = []
    for e in events:
        evidence = locate(e["evidence"], row["messages"])
        if (
            e["paralinguistic"] not in {"yes", "no", "unknown"}
            or e["elongation"] not in {"yes", "no", "unknown"}
            or e["sound"] not in SOUND
            or (not labels_valid(e["functions"], FUNCTIONS))
            or (e["paralinguistic"] == e["elongation"] == "no")
            or (not evidence["quote"].strip())
        ):
            raise ValueError("Invalid textual attributes")
        if any(
            f["message_id"] == evidence["message_id"]
            and f["start"] < evidence["end"]
            and evidence["start"] < f["end"]
            for f in result
        ):
            raise ValueError("Overlapping textual events")
        if any(
            f["message_id"] == evidence["message_id"]
            and f["start"] < evidence["end"]
            and evidence["start"] < f["end"]
            for f in raw["emoji"]
        ):
            raise ValueError("Emoji belongs to the separate emoji axis")
        result.append(
            dict(
                evidence,
                **{k: e[k] for k in ("paralinguistic", "sound", "functions", "elongation")},
            )
        )
    return {"state": state, "events": result, "reason": reason}


def validate_response(response, row, raw):
    result, errors = {}, {}
    for key in ("text", "repeat_usage", "emoji_usage"):
        try:
            if response["state"] != "ok":
                raise ValueError(response.get("error", "Call failed"))
            result[key] = validate_part(key, response["data"], row, raw)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            errors[key] = str(exc)
    return result, errors


def registry(row, raw, annotation):
    """Deduplicate span observations while retaining each source's exact boundaries and attributes."""
    output = []
    repeat_usage = {e["anchor_id"]: e for e in annotation.get("repeat_usage", [])}
    for msg in row["messages"]:
        events = []
        sources = ("grammar", "repeats")
        for source in sources:
            for e in raw[source]:
                if e["message_id"] == msg["message_id"]:
                    observation = dict(e, source=source)
                    if source == "repeats":
                        usage = repeat_usage.get(e["anchor_id"], {})
                        observation.update(uses=usage.get("uses", ["unknown"]))
                    events.append(observation)
        events.extend(
            dict(e, source="llm")
            for e in annotation.get("text", {}).get("events", [])
            if e["message_id"] == msg["message_id"]
        )
        components = []
        for e in sorted(events, key=lambda e: (e["start"], e["end"])):
            if components and e["start"] < components[-1]["end"]:
                components[-1]["end"] = max(components[-1]["end"], e["end"])
                components[-1]["observations"].append(e)
            else:
                components.append({"start": e["start"], "end": e["end"], "observations": [e]})
        for c in components:
            result = span_event(
                msg["text"],
                c["start"],
                c["end"],
                message_id=msg["message_id"],
                observations=c["observations"],
            )
            result.update(
                event_id=f"{msg['message_id']}:{c['start']}:{c['end']}",
                writing=writing(result["quote"]),
                characters=c["end"] - c["start"],
                position=position(result, msg["text"]),
                elongation=sorted({e["elongation"] for e in c["observations"] if e["source"] == "llm"})
                or ["unknown"],
                grammar=sorted({e["grammar"] for e in c["observations"] if e["source"] == "grammar"})
                or ["unknown"],
                sound=sorted({e["sound"] for e in c["observations"] if e["source"] == "llm"}) or ["unknown"],
                functions=sorted(
                    {f for e in c["observations"] if e["source"] == "llm" for f in e["functions"]}
                )
                or ["unknown"],
            )
            output.append(result)
    return output


def annotate(row, parsed, judge, refresh=False):
    judge = judge.for_record(row)
    raw = anchors(row, parsed)
    payload = {
        "context": row["context"],
        "messages": row["messages"],
        "anchors": {k: raw[k] for k in ("grammar", "repeats", "emoji")},
    }
    response = judge.call(ANNOTATION_PROMPT, payload, refresh=refresh)
    data, errors = validate_response(response, row, raw)
    initial = dict(errors)
    keys = dict.fromkeys(("text", "repeat_usage", "emoji_usage"), response["cache_key"])
    repair = None
    if errors:
        repair = judge.call(
            ANNOTATION_PROMPT,
            payload,
            refresh=refresh,
            attempt=1,
        )
        fixed, failures = validate_response(repair, row, raw)
        for key in list(errors):
            keys[key] = repair["cache_key"]
            if key in fixed:
                data[key] = fixed[key]
                del errors[key]
            else:
                errors[key] = failures[key]
    if errors:
        judge.invalidate(response["cache_key"], repair["cache_key"] if repair else None)
    items = {}
    for item, key in (
        ("099-105", "text"),
        ("108", "text"),
        ("110", "text"),
        ("106-107,109", "repeat_usage"),
        ("112", "emoji_usage"),
    ):
        if key in errors:
            items[item] = {"state": "failure", "events": [], "reason": errors[key]}
        elif key == "text":
            value = data[key]
            attr = "elongation" if item == "110" else "paralinguistic"
            events = [e for e in value["events"] if e[attr] == "yes"]
            unknown = sum(e[attr] == "unknown" for e in value["events"])
            if item == "108":
                unknown += sum(e["sound"] == "unknown" for e in events)
                events = [e for e in events if e["sound"] in SIMULATED_SOUNDS]
            items[item] = {
                "state": "present"
                if events
                else "unknown"
                if unknown
                else "absent"
                if value["state"] == "present"
                else value["state"],
                "events": events,
                "unknown_events": unknown,
                "reason": value["reason"],
            }
        else:
            items[item] = {
                "state": "present" if data[key] else "na",
                "events": data[key],
                "reason": "" if data[key] else "no candidates",
            }
    result = {
        "sample_id": row["sample_id"],
        "items": items,
        "text": data.get("text", {}),
        "repeat_usage": data.get("repeat_usage", []),
        "emoji_usage": data.get("emoji_usage", []),
        "cache_key": response["cache_key"],
        "repair_cache_key": repair["cache_key"] if repair else None,
        "initial_failures": initial,
        "remaining_failures": errors,
        "part_cache_keys": keys,
    }
    result["event_registry"] = registry(row, raw, result)
    return result


def summarize_semantic(records, annotations, raw):
    result = {}
    for item in ITEMS:
        values = [a["items"][item] for a in annotations]
        states = Counter(v["state"] for v in values)
        valid_ids = [i for i, v in enumerate(values) if v["state"] in {"present", "absent"}]
        subset = [records[i] for i in valid_ids]
        groups = [values[i]["events"] for i in valid_ids]
        summary = aggregate(subset, groups)
        # aggregate() keeps blank messages in the denominator, so missing windows count them too.
        with_missing(
            summary["reliability"]["message_rate"],
            *missing_messages(((records[i], v["state"]) for i, v in enumerate(values)), blank=True),
        )
        summary.update(
            states=states,
            n_windows=len(values),
            valid_windows=len(valid_ids),
            coverage=ratio(len(valid_ids), len(values)),
            window_occurrence_rate=ratio(states["present"], len(valid_ids)),
            unknown_events=sum(v.get("unknown_events", 0) for v in values),
        )
        all_events = [e for v in values for e in v["events"]]
        if item == "099-105":
            summary["sound"] = Counter(e["sound"] for e in all_events)
            summary["functions"] = Counter(f for e in all_events for f in e["functions"])
            summary["function_by_writing"] = Counter(
                f + "|" + writing(e["quote"]) for e in all_events for f in e["functions"]
            )
        elif item == "108":
            summary["sound"] = Counter(e["sound"] for e in all_events)
            summary["writing"] = Counter(writing(e["quote"]) for e in all_events)
        elif item == "106-107,109":
            summary["uses"] = Counter(f for e in all_events for f in e["uses"])
            total = sum(len(r["repeats"]) for r in raw["records"])
            known = sum("unknown" not in e["uses"] for e in all_events)
            summary.update(
                candidate_repeats=total,
                known_repeats=known,
                annotation_coverage=ratio(known, total),
            )
        elif item == "110":
            summary["lengths"] = [len(e["quote"]) for e in all_events]
            summary["symbols"] = Counter(
                c for e in all_events for c in e["quote"] if unicodedata.category(c).startswith(("P", "S"))
            )
            summary["carriers"] = Counter(
                "".join(
                    c
                    for c in e["quote"]
                    if not unicodedata.category(c).startswith(("P", "S")) and not c.isspace()
                )
                for e in all_events
            )
            summary["repetitions"] = [r["repetitions"] for e in all_events for r in repetitions(e["quote"])]
        elif item == "112":
            summary["functions"] = Counter(f for e in all_events for f in e["functions"])
            total = sum(len(r["emoji"]) for r in raw["records"])
            known = sum("unknown" not in e["functions"] for e in all_events)
            summary.update(
                candidate_emoji=total,
                known_emoji=known,
                annotation_coverage=ratio(known, total),
            )
        result[item] = summary
    result["unique_text_events"] = {
        "count": sum(len(a["event_registry"]) for a in annotations),
        "rule": "overlapping text observations merged; adjacent spans separate; attributes not added as event totals",
    }
    return result


def extract(corpora, parsed, judge, output, workers, qa_runner=None):
    summaries, semantic, annotations = {}, {}, {}
    for name, records in corpora.items():
        dest = output / name
        dest.mkdir()
        summaries[name] = summarize(records, parsed)
        write_json(dest / "paralinguistic_style.json", summaries[name])
        write_json(dest / "records.json", records)
    if judge:
        print(f"Annotating paralinguistic attributes for {distinct_records(corpora)} windows", flush=True)
        annotations = annotate_corpora(
            corpora, lambda row: annotate(row, parsed, judge), workers, judge=judge
        )
        for name, records in corpora.items():
            semantic[name] = summarize_semantic(records, annotations[name], summaries[name])
            write_json(output / name / "annotations.json", annotations[name])
            write_json(output / name / "semantic.json", semantic[name])
            if qa_runner and name == PRIMARY_CORPUS:
                write_json(output / name / "qa.json", qa_runner(records, annotations[name]))
    return semantic


def extraction_coverage(semantic, judge):
    return {
        item: {
            "state": "not_run"
            if item == "116"
            else "extracted"
            if judge or item not in ITEMS
            else "partial"
            if item in {"106-107,109", "112"}
            else "not_run",
            "source_ids": SOURCE_IDS[item],
            "corpora": {name: values[item]["states"] for name, values in semantic.items()}
            if item in ITEMS
            else {},
        }
        for item in MEASURES
    }


def protocol():
    return {
        "source_ids": SOURCE_IDS,
        "frequency_scale": DEFAULT_FREQUENCY_SCALE,
        "characters": "raw Unicode codepoints including whitespace",
        "message_rate": "all observed target text messages; empty platform-emoji messages retained",
        "modal_particle_forms": sorted(PARTICLES),
        "ambiguous_auxiliaries": sorted(AMBIGUOUS_PARTICLES),
        "interjection_forms": sorted(INTERJECTIONS),
        "grammar": "closed classes on LTP tokens, vocal strings (lexical_choice.vocal_token) skipped; interjection = repeat unit in the interjection class, POS not u/v; u requires RAD + finite head + clause-final position + frozen grammatical class; other terminal u unknown, structural/aspect uses not assumed modal",
        "onomatopoeia_108": "text events of the 099-105 annotation with paralinguistic=yes and a simulated sound; POS o is not used",
        "repetition": "any codepoint unit, k>=2; shortest unit per maximal integral span; longest-first nonoverlap, start breaks ties; no cross-message repetition",
        "emoji_data": {
            "version": "17.0",
            "path": str(DEFAULT_EMOJI_DATA),
            "sha256": hashlib.sha256(DEFAULT_EMOJI_DATA.read_bytes()).hexdigest(),
            "source": "https://www.unicode.org/Public/17.0.0/emoji/emoji-test.txt",
        },
        "emoji_matching": "exact extended grapheme against official table; qualification retained; unmatched sequences and bracketed text labels separate",
        "emoji_intervals": "consecutive Unicode emoji events of one window; cross-turn gaps only when the window lists all target turns of the conversation in order (dialogue) or boundary=message; distances in target turns and in bubbles (sent messages); character gaps count target text only; aware ISO timestamps only",
        "character_emoticons": "LLM extraction removed",
        "sound_labels": SOUND,
        "function_labels": FUNCTIONS,
        "repetition_use_labels": REPEAT_USES,
        "other_categories": "open description retained; not mapped to existing function by spelling",
    }
