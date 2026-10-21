"""Rhetorical devices: shared contextual extraction and evidence-based counts."""

from weclone.prompts.distill_style import (
    RHETORICAL_DEVICES_ANNOTATION_PROMPT as ANNOTATION_PROMPT,
)

from collections import Counter

from .data import PRIMARY_CORPUS, write_json
from .extraction import annotate_corpora, distinct_records, locate
from .stats import by_length, message_texts, missing_messages, ratio, reliability, with_missing

PROTOCOL_VERSION = "rhetorical-devices-v2"
DEFAULT_FREQUENCY_SCALE = 100
ITEMS = {
    "166": "比喻",
    "167": "拟人",
    "168": "夸张",
    "169": "反讽表达（总体用法）",
    "170": "排比",
    "metonymy": "借代",
    "antiphrasis": "反语",
    "171": "对比",
    "172": "反差",
    "174": "自嘲",
    "175": "调侃",
    "178-179": "网络梗使用与密度",
}
MEASURES = dict(ITEMS)
DEFERRED = ("173", "176", "177", "181")
SCOPE = "Shared rhetorical, communicative-tone and meme extraction"
DEFINITIONS = {
    "166": "比喻：借不同事物或领域间的相似性描写目标，含明喻、暗喻；普通同类比较、字面相似或解释词义不算。",
    "167": "拟人：将人的心理、意愿、社会行为赋予非人对象；真实人的行为、动物实际能做的动作、普通技术功能描述不算。",
    "168": "夸张：有语境依据的明显放大或缩小，用来加强表达效果；普通程度强调、可能真实的大数值、无法核实的事实不自动算。",
    "169": "反讽表达（总体用法）：结合可见语境，表达通过评价反向、回声引用、假装附和或预期落差，呈现质疑、嘲弄或戏谑式疏离。普通否定、事实差异或意外本身不算。包含反语，也可没有字面评价反向；不推断真实心理或人格。",
    "170": "排比：至少三个结构相近、语义相关、共同承担表达推进或强调功能的分句/短语；普通名词清单和偶然句法相似不算。可跨同一回复的消息。",
    "metonymy": "借代：基于邻接或关联关系，用相关事物称代目标对象，如部分与整体、载体与内容；需说明替代双方及关系。区别基于相似性的比喻与普通字面指称。",
    "antiphrasis": "反语：字面评价与可见语境支持的传达评价方向相反；说明反向依据。可以表达批评，也可以亲昵打趣。它是169的实现方式，同一事件同时标169与本子类，共享证据，不作为独立总分项。",
    "171": "对比：有意并置两个或多个对象、状态或立场的差异，以突出特点或评价；普通逐项报告不同数值不自动算。",
    "172": "反差：用预期与现实、文本中明确呈现的前后表达之间显著落差组织表达效果；普通差异、单纯转折不自动算。可与对比、反讽共现；只识别文本中有证据的表达落差，不比较表层风格与未知的真实性格、真实想法。",
    "174": "自嘲：说话者以玩笑、讽刺等方式贬抑或打趣自己及明确包括自己的群体；客观承认失误、普通自谦或认真自责不自动算。",
    "175": "调侃：以可辨认的玩笑式评论打趣人或事，含自我调侃；区分认真批评、攻击和普通提问。关系亲疏、真实善意无法确定时不推断。",
    "178-179": "网络梗：当前表达可识别为复用或化用共享的网络表达模板、典故或文化符号；一般网络口语、普通词义、仅提及梗这个词不算。无需命中预设词典；认不准是否用梗则unknown，确认用梗但不知道名称或来源可留null。",
}


def validate_item(key, value, row):
    if not isinstance(value, dict):
        raise ValueError("Missing item object")
    state, reason, events = value.get("state"), value.get("reason"), value.get("events")
    if state not in ("present", "absent", "unknown", "na"):
        raise ValueError("Invalid state")
    if not isinstance(events, list):
        raise ValueError("Expected reason and events")
    empty = not any(m["text"].strip() for m in row["messages"])
    if (state == "na") != empty or (state == "present") != bool(events):
        raise ValueError("State conflicts with text/events")
    order = {m["message_id"]: i for i, m in enumerate(row["messages"])}
    normalized, seen, occupied = [], set(), set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Expected event detail")
        evidence = event.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("Expected evidence array")
        located = [locate(e, row["messages"]) for e in evidence]
        if any(not e["quote"].strip() for e in located):
            raise ValueError("Whitespace is not rhetorical evidence")
        located.sort(key=lambda e: (order[e["message_id"]], e["start"], e["end"]))
        spans, used = [], set()
        for e in located:
            positions = {(e["message_id"], i) for i in range(e["start"], e["end"])}
            if positions & used:
                raise ValueError("Overlapping evidence inside event")
            used.update(positions)
            spans.append({k: e[k] for k in ("message_id", "quote", "occurrence", "start", "end")})
        identity = tuple((e["message_id"], e["start"], e["end"]) for e in spans)
        result = {"evidence": spans}
        if identity in seen:
            continue
        if used & occupied:
            raise ValueError("Overlapping events: combine the same rhetorical occurrence")
        seen.add(identity)
        occupied.update(used)
        normalized.append(result)
    normalized.sort(key=lambda e: (order[e["evidence"][0]["message_id"]], e["evidence"][0]["start"]))
    return {"state": state, "reason": reason, "events": normalized}


def validate_response(value, row):
    raw = value.get("items", {}) if isinstance(value, dict) else {}
    valid, failures = {}, {}
    for key in ITEMS:
        try:
            valid[key] = validate_item(key, raw.get(key) if isinstance(raw, dict) else None, row)
        except (ValueError, KeyError, TypeError) as exc:
            failures[key] = str(exc)
    if "antiphrasis" in valid and "169" in valid and valid["antiphrasis"]["state"] == "present":

        def positions(item):
            return {
                (s["message_id"], i)
                for e in item["events"]
                for s in e["evidence"]
                for i in range(s["start"], s["end"])
            }

        if valid["169"]["state"] != "present" or not positions(valid["antiphrasis"]) <= positions(
            valid["169"]
        ):
            failures.update(
                dict.fromkeys(("169", "antiphrasis"), "Antiphrasis must be included in irony evidence")
            )
    for key in failures:
        valid.pop(key, None)
    return valid, failures


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


def summarize(records, parsed):
    return {
        "windows": len(records),
        "messages": sum(len(r["messages"]) for r in records),
        "characters": sum(len(m["text"]) for r in records for m in r["messages"]),
        "nonempty_messages": sum(bool(m["text"].strip()) for r in records for m in r["messages"]),
    }


def summarize_semantic(records, annotations):
    if [r["sample_id"] for r in records] != [a["sample_id"] for a in annotations]:
        raise ValueError("Annotation/record alignment mismatch")
    result = {}
    for key in ITEMS:
        states, occurrences, events = Counter(), Counter(), []
        message_n = char_n = hit_messages = evidence_characters = 0
        texts, hits = {}, set()
        for row, annotation in zip(records, annotations, strict=True):
            item = annotation["items"][key]
            states[item["state"]] += 1
            if item["state"] not in ("present", "absent"):
                continue
            nonempty = [m for m in row["messages"] if m["text"].strip()]
            message_n += len(nonempty)
            char_n += sum(len(m["text"]) for m in nonempty)
            occurrences[str(len(item["events"]))] += 1
            hit_messages += len({s["message_id"] for e in item["events"] for s in e["evidence"]})
            texts.update(message_texts([row]))
            hits.update((row["sample_id"], s["message_id"]) for e in item["events"] for s in e["evidence"])
            evidence_characters += len(
                {
                    (s["message_id"], i)
                    for e in item["events"]
                    for s in e["evidence"]
                    for i in range(s["start"], s["end"])
                }
            )
            events.extend(dict(e, sample_id=row["sample_id"]) for e in item["events"])
        n = states["present"] + states["absent"]
        unknown, failure = missing_messages(
            (row, a["items"][key]["state"]) for row, a in zip(records, annotations, strict=True)
        )
        result[key] = {
            "states": dict(states),
            "valid_messages": message_n,
            "hit_messages": hit_messages,
            "message_occurrence_rate": ratio(hit_messages, message_n),
            "by_length": {"message_occurrence_rate": by_length(texts, hits)},
            "reliability": {
                "message_occurrence_rate": with_missing(reliability(texts, hits), unknown, failure)
            },
            "valid_windows": n,
            "total_windows": len(records),
            "coverage": ratio(n, len(records)),
            "unknown_rate": ratio(states["unknown"], len(records)),
            "failure_rate": ratio(states["failure"], len(records)),
            "window_occurrence_rate": ratio(states["present"], n),
            "valid_characters": char_n,
            "event_count": len(events),
            "events_per_100_messages": ratio(DEFAULT_FREQUENCY_SCALE * len(events), message_n),
            "events_per_100_characters": ratio(DEFAULT_FREQUENCY_SCALE * len(events), char_n),
            "evidence_character_share": ratio(evidence_characters, char_n),
            "window_event_count_distribution": dict(occurrences),
            "events": events,
        }
    return result


def extract(corpora, parsed, judge, output, workers, qa_runner=None):
    semantic, annotations = {}, {}
    for name, records in corpora.items():
        dest = output / name
        dest.mkdir()
        write_json(dest / "records.json", records)
        write_json(dest / "rhetorical_devices.json", summarize(records, parsed))
    if judge:
        print(f"Annotating rhetorical devices for {distinct_records(corpora)} windows", flush=True)
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
            key: {
                "name": title,
                "state": "extracted" if judge else "not_run",
                "corpora": {name: values[key]["states"] for name, values in semantic.items()},
            }
            for key, title in MEASURES.items()
        },
        **{key: {"state": "deferred", "reason": "high_difficulty"} for key in DEFERRED},
    }


def protocol():
    return {
        "items": MEASURES,
        "definitions": DEFINITIONS,
        "deferred": DEFERRED,
        "unnumbered_items": {"metonymy": "借代", "antiphrasis": "反语"},
        "method": "One contextual LLM extraction, at most one structural repair; no trigger dictionaries or LTP-based semantic prefilter",
        "rule_basis": "Definitions distinguish similarity, association, reversal, arrangement and social expression; three units is the operational minimum for parallelism; examples never define candidate eligibility",
        "unit": "One event may span several target messages within a window; context evidence is explanatory only and never counted",
        "deduplication": "Identical ordered span sets counted once per item; overlapping distinct events rejected for repair; repeated expressions at different positions remain separate",
        "denominator": "Only complete present/absent windows; nonempty target messages (primary) and their raw Unicode character lengths (including spaces/punctuation/emoji code points); all unknown/NA/failure excluded and reported",
        "message_occurrence": "Messages touched by event evidence / nonempty messages in determinate windows; one cross-message event can touch several messages",
        "frequency_scale": DEFAULT_FREQUENCY_SCALE,
        "meme_density": "178-179 reports occurrence and density of the same events; no generated name/source descriptions or era estimates",
        "subtypes": {"antiphrasis": {"parent": "169", "independent_score": False}},
        "multilabel": "Items may overlap; antiphrasis implies irony with included evidence; no summed overall score",
    }
