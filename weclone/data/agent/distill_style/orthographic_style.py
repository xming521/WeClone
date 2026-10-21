"""Orthographic/typographic style: original characters and boundaries, all counted locally."""

import re
import unicodedata
from collections import Counter
from datetime import date

import regex

from .data import bubble_spans, write_json
from .morphosyntax import probabilities
from .stats import length_strata, ratio, window_reliability

PROTOCOL_VERSION = "orthographic-style-v1"
SCOPE = "Orthographic / Typographic Style 078-095"
DEFAULT_FREQUENCY_SCALE = 1000
PUNCTUATION_CLASSES = {
    "comma": ",，",
    "period": ".．。｡",
    "question": "?？",
    "exclamation": "!！",
    "wave": "~～〜",
    "colon": ":：",
    "semicolon": ";；",
    "enumeration_comma": "、､",
}
ITEM_CLASSES = {
    "078": {"comma"},
    "079": {"period"},
    "080": {"question"},
    "081": {"exclamation"},
    "082": {"ellipsis"},
    "083": {"wave"},
    "084": {"colon", "semicolon"},
}
ELLIPSIS = re.compile(r"(?:…+|[.．]{3,})+")
SPACE = regex.compile(r"[\p{Zs}\t]+")
NEWLINE = re.compile(r"\r\n|[\r\n\u0085\u2028\u2029]")
LATIN_WORD = regex.compile(r"[\p{Latin}\p{M}]+")
GRAPHEME = regex.compile(r"\X")
DATE_PATTERN = re.compile(
    r"(?<!\d)(\d{4})([-/.])(\d{1,2})\2(\d{1,2})(?!\d)|(?<!\d)(\d{4})年(\d{1,2})月(\d{1,2})日?(?!\d)"
)
TIME_PATTERN = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\d:])")


def event(text, start, end, label):
    return {"start": start, "end": end, "quote": text[start:end], "label": label}


def is_punctuation(char):
    return unicodedata.category(char).startswith("P") or char in PUNCTUATION_CLASSES["wave"]


def punctuation_events(text):
    ellipses = {m.start(): m.end() for m in ELLIPSIS.finditer(text)}
    result, i = [], 0
    while i < len(text):
        if i in ellipses:
            end = ellipses[i]
            result.append(event(text, i, end, "ellipsis"))
            i = end
            continue
        c = text[i]
        if is_punctuation(c):
            label = next((k for k, chars in PUNCTUATION_CLASSES.items() if c in chars), None)
            if c in ".．" and 0 < i < len(text) - 1 and text[i - 1].isdecimal() and text[i + 1].isdecimal():
                label = "numeric_dot"
            result.append(event(text, i, i + 1, label or "unicode_" + unicodedata.category(c)))
        i += 1
    return result


def punctuation_runs(text):
    result, begin = [], None
    for i in range(len(text) + 1):
        if i < len(text) and is_punctuation(text[i]):
            if begin is None:
                begin = i
        elif begin is not None:
            if i - begin >= 2:
                result.append(event(text, begin, i, "punctuation_run"))
            begin = None
    return result


def emoji_end(text):
    clusters = list(GRAPHEME.finditer(text.rstrip()))
    if not clusters:
        return False
    s = clusters[-1].group()
    if "\ufe0e" in s:
        return False
    return bool(
        regex.search(r"\p{Emoji_Presentation}", s)
        or regex.fullmatch(r"[0-9#*]\ufe0f?\u20e3", s)
        or ("\ufe0f" in s and regex.search(r"\p{Emoji}", s))
    )


def ending(text, punctuation):
    stop = len(text.rstrip())
    if not stop:
        return {"label": "empty", "quote": ""}
    if emoji_end(text):
        return {"label": "unicode_emoji", "quote": list(GRAPHEME.finditer(text[:stop]))[-1].group()}
    terminal = next((e for e in reversed(punctuation) if e["end"] == stop), None)
    return (
        {"label": terminal["label"], "quote": terminal["quote"]}
        if terminal
        else {"label": "no_punctuation", "quote": text[stop - 1 : stop]}
    )


def boundary_candidates(row):
    result = {"087": [], "088": []}
    for msg in row["messages"]:
        for item, pattern in (("087", SPACE), ("088", NEWLINE)):
            for match in pattern.finditer(msg["text"]):
                result[item].append(
                    dict(
                        event(msg["text"], match.start(), match.end(), "boundary"),
                        message_id=msg["message_id"],
                        boundary_id=f"{item}:{msg['message_id']}:{match.start()}",
                    )
                )
    return result


def number_observations(text, parsed):
    p = parsed[text]
    result = []
    for i, (token, pos, start, end) in enumerate(
        zip(p["cws"], p["pos"], p["starts"], p["ends"], strict=True)
    ):
        if pos not in {"m", "nt"}:
            continue
        decimal = any(c.isdecimal() for c in token)
        han = bool(regex.search(r"\p{Han}", token))
        writing = (
            "mixed"
            if han and decimal
            else "arabic"
            if decimal
            else "han_numeral"
            if han and pos == "m"
            else "non_numeric_time"
            if pos == "nt"
            else "other"
        )
        quantity = (
            pos == "m"
            and i + 1 < len(p["pos"])
            and p["pos"][i + 1] == "q"
            and not text[end : p["starts"][i + 1]].strip()
        )
        result.append(
            dict(
                event(text, start, end, "time" if pos == "nt" else "quantity" if quantity else "number"),
                writing=writing,
                pos=pos,
                classifier=p["cws"][i + 1] if quantity else None,
            )
        )
    normalized = text.translate(str.maketrans({chr(i): chr(i - 0xFEE0) for i in range(0xFF01, 0xFF5F)}))
    dates, times = [], []
    for m in DATE_PATTERN.finditer(normalized):
        y, month, day = (m[1], m[3], m[4]) if m[1] else (m[5], m[6], m[7])
        try:
            date(int(y), int(month), int(day))
        except ValueError:
            continue
        dates.append(event(text, m.start(), m.end(), "ymd_" + m[2] if m[1] else "ymd_han_units"))
    for m in TIME_PATTERN.finditer(normalized):
        if int(m[1]) < 24 and int(m[2]) < 60 and (m[3] is None or int(m[3]) < 60):
            times.append(event(text, m.start(), m.end(), "hms" if m[3] else "hm"))
    return {"tokens": result, "dates": dates, "times": times}


def capitalization(text):
    result = []
    for m in LATIN_WORD.finditer(text):
        s = m.group()
        cased = [c for c in s if c.lower() != c.upper()]
        if not cased:
            continue
        label = (
            "uppercase"
            if all(c.isupper() for c in cased)
            else "lowercase"
            if all(c.islower() for c in cased)
            else "initial_capital"
            if cased[0].isupper() and all(c.islower() for c in cased[1:])
            else "mixed"
        )
        result.append(event(text, m.start(), m.end(), label))
    return result


def bubble_observations(text, punct):
    """Each sent message of a chat turn: its punctuation and how it ends."""
    spans = bubble_spans(text)
    return [
        {
            "start": start,
            "end": end,
            "nonspace_characters": sum(not c.isspace() for c in text[start:end]),
            "punctuated": any(start <= e["start"] and e["end"] <= end for e in punct),
            "ending": ending(text[:end], [e for e in punct if e["end"] <= end]),
            "final": i == len(spans) - 1,
        }
        for i, (start, end) in enumerate(spans)
    ]


def message_observation(row, msg, parsed):
    text = msg["text"]
    punct = punctuation_events(text)
    boundaries = boundary_candidates(dict(row, messages=[msg]))
    return {
        "sample_id": row["sample_id"],
        "message_id": msg["message_id"],
        "boundary": row["boundary"],
        "characters": len(text),
        "nonspace_characters": sum(not c.isspace() for c in text),
        "punctuation_characters": sum(is_punctuation(c) for c in text),
        "unicode_punctuation_characters": sum(unicodedata.category(c).startswith("P") for c in text),
        "punctuation": punct,
        "runs": punctuation_runs(text),
        "ending": ending(text, punct),
        "bubbles": bubble_observations(text, punct),
        "boundaries": boundaries,
        "numbers": number_observations(text, parsed),
        "capitalization": capitalization(text),
        "symbols": [
            event(text, i, i + 1, unicodedata.category(c))
            for i, c in enumerate(text)
            if unicodedata.category(c).startswith("S")
        ],
    }


def event_summary(observations, groups):
    nonempty = [o["nonspace_characters"] > 0 for o in observations]
    valid = sum(nonempty)
    matched = sum(bool(g) and keep for g, keep in zip(groups, nonempty, strict=True))
    n = sum(map(len, groups))
    chars = sum(o["nonspace_characters"] for o in observations)
    counts = Counter(e["label"] for group in groups for e in group)
    return {
        "count": n,
        "counts": counts,
        "rates": probabilities(counts),
        "forms": Counter(e["quote"] for group in groups for e in group),
        "n_messages": len(observations),
        "valid_messages": valid,
        "matched_messages": matched,
        "message_rate": ratio(matched, valid),
        "by_length": {
            "message_rate": length_strata(
                (o["nonspace_characters"], bool(g))
                for o, g in zip(observations, groups, strict=True)
                if o["nonspace_characters"]
            )
        },
        "reliability": {
            "message_rate": window_reliability(
                (o["sample_id"], bool(g))
                for o, g in zip(observations, groups, strict=True)
                if o["nonspace_characters"]
            )
        },
        "nonspace_characters": chars,
        "per_1000_characters": DEFAULT_FREQUENCY_SCALE * n / chars if chars else None,
        "message_counts": [len(g) if keep else None for g, keep in zip(groups, nonempty, strict=True)],
    }


def summarize(records, parsed):
    observations = [message_observation(r, m, parsed) for r in records for m in r["messages"]]
    metrics = {
        item: event_summary(
            observations, [[e for e in o["punctuation"] if e["label"] in labels] for o in observations]
        )
        for item, labels in ITEM_CLASSES.items()
    }
    metrics["085"] = event_summary(observations, [o["runs"] for o in observations])
    metrics["085"]["lengths"] = [len(e["quote"]) for o in observations for e in o["runs"]]
    valid = [o for o in observations if o["nonspace_characters"]]
    counts = Counter("no_punctuation" if not o["punctuation"] else "has_punctuation" for o in valid)
    metrics["086"] = {
        "counts": counts,
        "valid_messages": len(valid),
        "na_messages": len(observations) - len(valid),
        "no_punctuation_rate": ratio(counts["no_punctuation"], len(valid)),
        "by_length": {
            "no_punctuation_rate": length_strata(
                (o["nonspace_characters"], not o["punctuation"]) for o in valid
            )
        },
        "reliability": {
            "no_punctuation_rate": window_reliability((o["sample_id"], not o["punctuation"]) for o in valid)
        },
    }
    bubbles = [(o["sample_id"], b) for o in observations for b in o["bubbles"]]
    bare = [not b["punctuated"] for _, b in bubbles]
    metrics["086"].update(
        valid_bubbles=len(bubbles),
        no_punctuation_bubbles=sum(bare),
        bubble_no_punctuation_rate=ratio(sum(bare), len(bubbles)),
    )
    metrics["086"]["by_length"]["bubble_no_punctuation_rate"] = length_strata(
        (b["nonspace_characters"], not b["punctuated"]) for _, b in bubbles
    )
    metrics["086"]["reliability"]["bubble_no_punctuation_rate"] = window_reliability(
        (sid, not b["punctuated"]) for sid, b in bubbles
    )
    metrics["087"] = event_summary(observations, [o["boundaries"]["087"] for o in observations])
    metrics["087"]["boundary_count_including_whitespace_only_messages"] = sum(
        len(o["boundaries"]["087"]) for o in observations
    )
    # Chat input joins consecutive sends with a newline, so a newline is a message boundary;
    # 088 asks whether that boundary stands in for a comma or period.
    joins = [(sid, b) for sid, b in bubbles if not b["final"]]
    unpunctuated = [b["ending"]["label"] == "no_punctuation" for _, b in joins]
    metrics["088"] = {
        "newline_count": sum(len(o["boundaries"]["088"]) for o in observations),
        "boundaries": len(joins),
        "multi_bubble_messages": sum(len(o["bubbles"]) > 1 for o in observations),
        "preceding_endings": Counter(b["ending"]["label"] for _, b in joins),
        "unpunctuated_boundaries": sum(unpunctuated),
        "unpunctuated_boundary_rate": ratio(sum(unpunctuated), len(joins)),
        "by_length": {
            "unpunctuated_boundary_rate": length_strata(
                (b["nonspace_characters"], b["ending"]["label"] == "no_punctuation") for _, b in joins
            )
        },
        "reliability": {
            "unpunctuated_boundary_rate": window_reliability(
                (sid, b["ending"]["label"] == "no_punctuation") for sid, b in joins
            )
        },
    }
    chars = sum(o["nonspace_characters"] for o in observations)
    punctuation_chars = sum(o["punctuation_characters"] for o in observations)
    metrics["089"] = {
        "punctuation_characters": punctuation_chars,
        "nonspace_characters": chars,
        "density": ratio(punctuation_chars, chars),
        "event_count": sum(len(o["punctuation"]) for o in observations),
        "unicode_punctuation_characters": sum(o["unicode_punctuation_characters"] for o in observations),
        "message_densities": [
            ratio(o["punctuation_characters"], o["nonspace_characters"]) for o in observations
        ],
    }
    metrics["090"] = event_summary(observations, [o["punctuation"] for o in observations])
    metrics["091"] = {
        "counts": Counter(b["ending"]["label"] for _, b in bubbles),
        "forms": Counter(b["ending"]["quote"] for _, b in bubbles),
        "bubbles": len(bubbles),
        "nonfinal_counts": Counter(b["ending"]["label"] for _, b in joins),
        "turn_final_counts": Counter(o["ending"]["label"] for o in observations),
        "character_emoticon_resolution": "not extracted; surface endings only",
    }
    metrics["092"] = {
        axis: event_summary(observations, [o["numbers"][axis] for o in observations])
        for axis in ("tokens", "dates", "times")
    }
    metrics["092"]["writing"] = Counter(e["writing"] for o in observations for e in o["numbers"]["tokens"])
    metrics["093"] = event_summary(observations, [o["capitalization"] for o in observations])
    metrics["094"] = event_summary(observations, [o["symbols"] for o in observations])
    return {
        "n_windows": len(records),
        "n_messages": len(observations),
        "messages": observations,
        "boundary_kinds": Counter(r["boundary"] for r in records),
        "observed_cross_message_boundaries": sum(
            max(0, len(r["messages"]) - 1) for r in records if r["boundary"] == "message"
        ),
        "metrics": metrics,
    }


def extract(corpora, parsed, judge, output, workers, qa_runner=None):
    for name, records in corpora.items():
        dest = output / name
        dest.mkdir()
        write_json(dest / "orthographic_style.json", summarize(records, parsed))
        write_json(dest / "records.json", records)
    return {}


def extraction_coverage(semantic, judge):
    coverage = {f"{i:03}": {"state": "extracted"} for i in range(78, 96)}
    for item in ("087", "095"):
        coverage[item] = {
            "state": "partial" if item == "087" else "not_run",
            "reason": "Surface counts only; contextual function and emoticon LLM extraction removed",
        }
    coverage["088"] = {
        "state": "extracted",
        "reason": "Chat newlines are message boundaries; surface rate of boundaries after a bubble without terminal punctuation",
    }
    coverage["091"]["character_emoticon_ending"] = "not_run"
    return coverage


def protocol():
    return {
        "punctuation": "Unicode P* plus explicit tilde variants; original code points; ellipsis maximal event; numeric interior dot separated",
        "punctuation_classes": PUNCTUATION_CLASSES,
        "unicode_version": unicodedata.unidata_version,
        "regex_version": regex.__version__,
        "density": "punctuation codepoints / nonspace Unicode codepoints; events separately",
        "frequency_scale": DEFAULT_FREQUENCY_SCALE,
        "message_rate": "nonempty observed target message units; whitespace-only NA",
        "spaces": "maximal Unicode Zs/tab runs",
        "newlines": "CRLF one boundary; CR/LF/NEL/LS/PS individually; never cross messages",
        "bubbles": "nonblank CR/LF-separated pieces of a turn; chat input joins consecutive sends with a newline, so one bubble is one sent message",
        "no_punctuation": "086 no_punctuation_rate per turn; bubble_no_punctuation_rate per bubble",
        "boundary_without_punctuation": "088: every boundary between two bubbles of a turn, labelled by the ending of the bubble before it; unpunctuated_boundary_rate = share ending without punctuation, stratified by that bubble's length; real in-message newlines are indistinguishable from send boundaries in merged chat input",
        "endings": "091 counts every bubble's ending, nonfinal_counts bubbles followed by another bubble, turn_final_counts the last bubble of each turn (blank turns as empty); last event at rstrip end; Emoji presentation grapheme candidate first; no contextual emoticon classification; not RGI validation",
        "numbers": "LTP m/nt plus adjacent q; original digit/Han writing; valid Gregorian YMD and 24h clock regex separately; no inferred price/ID use",
        "capitalization": "maximal Latin+combining-mark runs; one uppercase letter is uppercase",
        "symbols": "all Unicode S* codepoints, unfiltered; can overlap wave punctuation and emoji components",
    }
