"""Compact LLM transport; canonical annotations retain the existing validators."""

from weclone.prompts.distill_style import (
    COMMON_PROMPT as COMMON_PROMPT,
    format_prompt as format_prompt,
)

from importlib import import_module

WIRE_VERSION = "style-state-table-v4"
MODULES = {
    "lexical-v5": "lexical_choice",
    "morphosyntax-v2": "morphosyntax",
    "syntactic-style-v4": "syntactic_style",
    "paralinguistic-style-v1": "paralinguistic_style",
    "pragmatic-style-v2": "pragmatic_style",
    "affective-style-v3": "affective_style",
    "rhetorical-devices-v2": "rhetorical_devices",
    "register-style-v2": "register_style",
    "information-density-v1": "information_density",
    "interactional-style-v2": "interactional_style",
}
DEGREE_CODES = {
    "0": ("no", "na"),
    "1": ("yes", "no"),
    "2": ("yes", "yes"),
    "u": ("unknown", "unknown"),
    "?": ("yes", "unknown"),
}
STATES = {"p": "present", "a": "absent", "u": "unknown", "n": "na", "k": "known"}


def schema_for(protocol, payload):
    if protocol not in MODULES or not isinstance(payload, dict) or "messages" not in payload:
        return None
    name = MODULES[protocol]
    module = import_module(f"{__package__}.{name}")
    root = name in {"syntactic_style", "paralinguistic_style"}
    extra = ()
    if name == "syntactic_style":
        keys, extra = ("self_qa",), ("units",)
    elif name == "paralinguistic_style":
        keys, extra = ("text",), ("repeat_usage", "emoji_usage")
    else:
        keys = tuple(module.ITEMS)
    if name == "lexical_choice" and not module.correction_applicable(
        payload["boundary"], payload["messages"]
    ):
        keys = tuple(k for k in keys if k != "015")
    failures = payload.get("failures")
    if isinstance(failures, dict) and set(failures) <= set(keys + extra):
        keys = tuple(k for k in keys if k in failures)
        extra = tuple(k for k in extra if k in failures)
    return {
        "module": name,
        "keys": keys,
        "extra": extra,
        "root": root,
        "per_message": name in {"pragmatic_style", "affective_style"},
        "message_ids": [m["message_id"] for m in payload["messages"]],
        "degree_anchors": payload.get("degree_anchors", []),
        "messages": payload["messages"],
    }


def pack_evidence(value):
    if isinstance(value, list):
        return [pack_evidence(v) for v in value]
    if isinstance(value, dict):
        if {"message_id", "quote", "occurrence"} <= set(value):
            return [value["message_id"], value["quote"], value["occurrence"]]
        return {
            k: pack_evidence(v)
            for k, v in value.items()
            if k not in {"reason", "detail", "function", "proposal", "meaning"}
            and not (k == "basis" and isinstance(v, str))
        }
    return value


def pack_output(data, schema):
    """Encode canonical cached/repair data without generating any missing states."""
    source = data if schema["root"] else data.get("items", {})
    codes = {v: k for k, v in STATES.items()}
    states, details = {}, {}

    def detail_of(key, value):
        if schema["module"] == "affective_style" and key == "163":
            by_anchor = {e["anchor_id"]: e for e in value["events"]}
            anchors = [a for a in schema["degree_anchors"] if a["message_id"] == value["message_id"]]
            if value["state"] == "present":
                if set(by_anchor) != {a["anchor_id"] for a in anchors} or len(by_anchor) != len(
                    value["events"]
                ):
                    raise ValueError("Incomplete degree candidate coverage")
                codes = {v: k for k, v in DEGREE_CODES.items()}
                return {
                    "decisions": "".join(
                        codes[(by_anchor[a["anchor_id"]]["degree"], by_anchor[a["anchor_id"]]["emotion"])]
                        for a in anchors
                    )
                }
            return {}
        detail = pack_evidence({k: v for k, v in value.items() if k not in {"state", "message_id"}})
        for field in ("events", "pairs", "questions", "other_uses"):
            if detail.get(field) == []:
                detail.pop(field)
        if schema["module"] == "affective_style" and value["state"] != "present":
            detail.pop("intensity", None)
            detail.pop("tone", None)
        if schema["module"] == "register_style" and key == "233" and value["state"] != "present":
            detail.pop("label", None)
        for event in detail.get("events", []):
            if schema["module"] == "pragmatic_style":
                for field in list(event):
                    if (
                        event[field] == []
                        or (field == "condition" and event[field] == "")
                        or (field in {"choice", "insistence"} and event[field] == "unexpressed")
                    ):
                        event.pop(field)
            if schema["module"] == "lexical_choice":
                if event.get("attributes") == {}:
                    event.pop("attributes")
                if event.get("source") is None:
                    event.pop("source", None)
            if schema["module"] == "rhetorical_devices":
                event.pop("meme_name", None)
                event.pop("meme_source", None)
        return detail

    for key in schema["keys"]:
        if key not in source:
            continue
        value = source[key]
        if schema["per_message"]:
            by_id = {v["message_id"]: v for v in value}
            if len(by_id) != len(value) or set(by_id) != set(schema["message_ids"]):
                raise ValueError("Incomplete canonical message coverage")
            states[key] = "".join(codes[by_id[mid]["state"]] for mid in schema["message_ids"])
            parts = {mid: d for mid in schema["message_ids"] if (d := detail_of(key, by_id[mid]))}
            if parts:
                details[key] = parts[schema["message_ids"][0]] if len(schema["message_ids"]) == 1 else parts
        else:
            states[key] = codes[value["state"]]
            if detail := detail_of(key, value):
                details[key] = detail["questions"] if key == "questions" else detail
    for key in schema["extra"]:
        if key in source:
            details[key] = pack_evidence(source[key])
    result = {"states": "".join(states[k] for k in schema["keys"] if k in states)}
    if schema["root"]:
        result.update(details)
    elif details:
        result["items"] = details
    return result


def pack_input(payload, schema=None):
    result = dict(payload)
    if schema and schema["module"] == "lexical_choice":
        result.pop("boundary", None)
    for key in ("context", "messages", "dialogue"):
        if key in result:
            result[key] = [[m["message_id"], m["role"], m["text"]] for m in result[key]]
    if "degree_anchors" in result:
        result["degree_anchors"] = [
            {
                "anchor_id": a["anchor_id"],
                "message_id": a["message_id"],
                "source": a["source"],
                "modifier": a["quote"],
                "target": a["target"]["quote"],
            }
            for a in result["degree_anchors"]
        ]
    for key in ("original_output", "previous_output"):
        if key in result:
            result[key] = (
                pack_output(result[key], schema)
                if schema and isinstance(result[key], dict)
                else pack_evidence(result[key])
            )
    return result


def expand_evidence(value):
    if isinstance(value, list):
        if (
            len(value) == 3
            and isinstance(value[0], str)
            and isinstance(value[1], str)
            and type(value[2]) is int
        ):
            return {"message_id": value[0], "quote": value[1], "occurrence": value[2]}
        return [expand_evidence(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_evidence(v) for k, v in value.items()}
    return value


def expand_item(key, code, detail, module):
    if code not in STATES or (code == "k" and key != "questions"):
        raise ValueError("Invalid compact state")
    if key == "questions" and isinstance(detail, list):
        detail = {"questions": detail}
    if not isinstance(detail, dict) or "state" in detail or "message_id" in detail:
        raise ValueError("Invalid compact detail")
    field = "pairs" if key == "self_qa" else "questions" if key == "questions" else "events"
    value = {"state": STATES[code], field: [], "reason": "", **expand_evidence(detail)}
    if module == "morphosyntax" and key == "036":
        value.setdefault("other_uses", [])
    if module == "register_style" and key == "233" and code in "un":
        value.setdefault("label", None)
    if module == "register_style":
        for event in value.get("events", []):
            evidence = event.get("evidence") if isinstance(event, dict) else None
            if isinstance(evidence, list) and len(evidence) == 1:
                event["evidence"] = evidence[0]
    if module == "affective_style":
        if key == "152" and code in "aun":
            value.setdefault("intensity", 0 if code == "a" else None)
        if key == "161" and code in "aun":
            value.setdefault("tone", "unmarked" if code == "a" else "unknown")
    if module == "pragmatic_style":
        from .pragmatic_style import REF_FIELDS

        for event in value.get("events", []):
            event.setdefault("scope", [])
            for field in REF_FIELDS.get(key, ()):
                event.setdefault(field, [])
            if key == "117-119":
                event.setdefault("certainty_evidence", [])
            if key == "136-137":
                event.setdefault("choice", "unexpressed")
                event.setdefault("condition", "")
                event.setdefault("condition_evidence", [])
            if key == "138":
                event.setdefault("insistence", "unexpressed")
            for field in ("buffers", "supports"):
                if key == {"buffers": "139", "supports": "145"}[field]:
                    event.setdefault(field, [])
            for child in event.get("buffers", []):
                child.setdefault("politeness_ids", [])
                child.setdefault("hedge_ids", [])
    if module == "lexical_choice":
        for event in value.get("events", []):
            if isinstance(event, dict):
                event.setdefault("attributes", {})
                event.setdefault("source", None)
    return value


def expand_output(data, schema):
    width = len(schema["message_ids"]) if schema["per_message"] else 1
    if (
        isinstance(data, dict)
        and isinstance(data.get("states"), list)
        and len(data["states"]) == len(schema["keys"])
        and all(isinstance(c, str) and len(c) == width for c in data["states"])
    ):
        # One state string per item is the same table, already split.
        data = dict(data, states="".join(data["states"]))
    if isinstance(data, dict) and isinstance(data.get("states"), str):
        codes = data["states"]
        if len(codes) != len(schema["keys"]) * width:
            raise ValueError("Incomplete compact state coverage")
        data = dict(
            data, states={k: codes[i * width : (i + 1) * width] for i, k in enumerate(schema["keys"])}
        )
    if not isinstance(data, dict) or not isinstance(data.get("states"), dict):
        raise ValueError("Missing complete state table")
    if set(data["states"]) != set(schema["keys"]):
        raise ValueError("Missing or unexpected state IDs")
    details = {k: v for k, v in data.items() if k != "states"} if schema["root"] else data.get("items", {})
    if not schema["root"] and set(data) - {"states", "items"}:
        raise ValueError("Unexpected compact output field")
    if not isinstance(details, dict) or set(details) - set(schema["keys"] + schema["extra"]):
        raise ValueError("Unexpected detail IDs")
    result = {}
    for key in schema["keys"]:
        codes = data["states"][key]
        count = len(schema["message_ids"]) if schema["per_message"] else 1
        if not isinstance(codes, str) or len(codes) != count:
            raise ValueError("Incomplete message state coverage")
        detail = details.get(key, {})
        if schema["per_message"]:
            if count == 1 and isinstance(detail, dict) and not (set(detail) & set(schema["message_ids"])):
                fields = {"events"}
                if schema["module"] == "affective_style":
                    fields |= {"152": {"intensity"}, "161": {"tone"}, "163": {"decisions"}}.get(key, set())
                if set(detail) - fields:
                    raise ValueError("Unexpected single-message detail field")
                detail = {schema["message_ids"][0]: detail}
            if not isinstance(detail, dict) or set(detail) - set(schema["message_ids"]):
                raise ValueError("Unexpected detail message IDs")
            if schema["module"] == "affective_style" and key == "163":
                expanded = {}
                for mid, code in zip(schema["message_ids"], codes, strict=True):
                    part = detail.get(mid, {})
                    if code != "p":
                        if part:
                            raise ValueError("Unexpected degree decisions for non-present state")
                        expanded[mid] = {}
                        continue
                    anchors = [a for a in schema["degree_anchors"] if a["message_id"] == mid]
                    decisions = part.get("decisions") if isinstance(part, dict) else None
                    if (
                        not isinstance(decisions, str)
                        or set(part) != {"decisions"}
                        or len(decisions) != len(anchors)
                    ):
                        raise ValueError("Incomplete degree decisions")
                    events = []
                    text = next(m["text"] for m in schema["messages"] if m["message_id"] == mid)
                    for anchor, decision in zip(anchors, decisions, strict=True):
                        if decision not in DEGREE_CODES:
                            raise ValueError("Unknown degree decision")
                        degree, emotion = DEGREE_CODES[decision]
                        target = anchor["target"]
                        quote = text[target["start"] : target["end"]]
                        occurrence = sum(text.startswith(quote, i) for i in range(target["start"]))
                        events.append(
                            {
                                "anchor_id": anchor["anchor_id"],
                                "degree": degree,
                                "emotion": emotion,
                                "evidence": [mid, quote, occurrence],
                            }
                        )
                    expanded[mid] = {"events": events}
                detail = expanded
            result[key] = [
                dict(expand_item(key, code, detail.get(mid, {}), schema["module"]), message_id=mid)
                for mid, code in zip(schema["message_ids"], codes, strict=True)
            ]
        else:
            result[key] = expand_item(key, codes, detail, schema["module"])
    for key in schema["extra"]:
        if key not in details:
            raise ValueError("Missing non-state result")
        result[key] = expand_evidence(details[key])
    if schema["module"] == "syntactic_style":
        for unit in result.get("units", []):
            if isinstance(unit, dict):
                unit.setdefault("reason", "")
    return result if schema["root"] else {"items": result}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result
