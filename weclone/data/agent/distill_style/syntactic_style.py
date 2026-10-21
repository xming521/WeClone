"""Syntactic style: sentence units, observable structures and contextual annotation."""

from weclone.prompts.distill_style import (
    SYNTACTIC_STYLE_ANNOTATION_PROMPT as ANNOTATION_PROMPT,
)

import math
from collections import Counter, defaultdict
from statistics import mean, pvariance

from weclone.utils import secure_storage

from .data import PRIMARY_CORPUS, digest, write_json
from .extraction import annotate_corpora, distinct_records, locate, merge_parses
from .lexical_choice import vocal_token, word_token
from .morphosyntax import (
    NOUN_POS,
    add_positions,
    probabilities,
    sentence_units,
)
from .stats import by_length, message_texts, ratio, reliability, with_missing

PROTOCOL_VERSION = "syntactic-style-v4"
SCOPE = "Syntactic Style 037-076; low/medium difficulty and existing 070/075"
HIGH_DIFFICULTY = {"049", "050", "055"}
# Checked on chat text: LTP never yields a condition role for 042 and counts replies such as "好的" as
# predicates for 047, which 043 already measures.
REMOVED = {
    "042": "removed: no condition role or clause boundary in chat parses, low value for chat",
    "047": "removed: replies parsed as predicates; completeness and responses are measured by 043",
}
DEFAULT_SHORT_WORDS = 5
DEFAULT_NGRAM_ORDER = 3
OBJECT_RELATIONS = {"VOB", "IOB", "FOB"}
EVENT_RELATIONS = {"eCOO", "ePREC", "eSUCC"}
DISTANCE_RELATIONS = {"063": "SBV", "064": "VOB", "065": "ADV", "066": "CMP"}
ITEMS = {
    "043": ("complete", "elliptical", "fragment", "unknown"),
    "051": ("declarative",),
    "052,211": ("interrogative",),
    "053": ("exclamative",),
    "054": ("imperative",),
    "056": ("self_question_answer",),
    "057": ("dislocated",),
    "058-059,062": ("fronted",),
    "060": ("elliptical", "complete", "response_particle", "fragment", "unknown"),
}
# 057 and 058-059,062 read the unit word order; LTP TPC/FOB fire on 0.5%/1.4% of chat segments.
ORDER_ITEMS = {"057", "058-059,062"}


class SyntaxParser:
    """Parse original text with all syntax tasks in an independent cache."""

    def __init__(self, parser):
        self.parser = parser
        self.identity = digest(
            {"ltp": parser.identity, "tasks": ["pos", "dep", "sdp"], "tokenization": "style-cws-v1"}
        )
        self.cache = parser.cache.parent / "ltp_syntax"
        self.cache.mkdir(parents=True, exist_ok=True)

    def parse(self, texts):
        base = self.parser.parse(texts)
        result, missing = {}, []
        for text, value in base.items():
            path = self.cache / (digest({"parser": self.identity, "text": text}) + ".json")
            if secure_storage.file_exists(path):
                result[text] = secure_storage.read_json(path)
            elif not value["cws"]:
                result[text] = dict(value, dep={"head": [], "label": []}, sdp={"head": [], "label": []})
            elif value.get("chunks"):
                chunks = [(c["start"], text[c["start"] : c["end"]]) for c in value["chunks"]]
                values = self.parse([part for _, part in chunks])
                result[text] = merge_parses([(offset, part, values[part]) for offset, part in chunks])
                write_json(path, result[text])
            else:
                missing.append((text, path))
        for start in range(0, len(missing), self.parser.batch_size):
            batch = missing[start : start + self.parser.batch_size]
            output = self.parser.pipeline([t for t, _ in batch], tasks=["cws", "pos", "dep", "sdp"])
            for i, (text, path) in enumerate(batch):
                starts, ends, words, cursor = [], [], [], 0
                for word in output.cws[i]:
                    begin = text.find(word, cursor)
                    if begin < 0 or not word.strip():
                        raise ValueError("Syntax token cannot align with original text")
                    cursor = begin + len(word)
                    starts.append(begin + len(word) - len(word.lstrip()))
                    ends.append(cursor - len(word) + len(word.rstrip()))
                    words.append(word.strip())
                covered = {j for begin, end in zip(starts, ends) for j in range(begin, end)}
                if any(not c.isspace() and j not in covered for j, c in enumerate(text)):
                    raise ValueError("Syntax parser dropped original characters")
                value = {
                    "cws": words,
                    "starts": starts,
                    "ends": ends,
                    "pos": output.pos[i],
                    "dep": output.dep[i],
                    "sdp": output.sdp[i],
                }
                n = len(value["cws"])
                if len(value["pos"]) != n or any(
                    len(value[k][field]) != n for k in ("dep", "sdp") for field in ("head", "label")
                ):
                    raise ValueError("Syntax parser truncated the fixed token sequence")
                write_json(path, value)
                result[text] = value
        return result


def prepare(records, parser):
    texts = [m["text"] for r in records for m in r["messages"]]
    parsed = parser.parse(texts)
    segments = [s["text"] for r in records for s in segments_for(r, parsed)]
    parsed.update(parser.parse(segments))
    return parsed


def segments_for(row, parsed):
    messages = {m["message_id"]: m["text"] for m in row["messages"]}
    for message_id, index, tokens in sentence_units(row, parsed):
        start, end = tokens[0]["start"], tokens[-1]["end"]
        yield {
            "message_id": message_id,
            "sentence_index": index,
            "start": start,
            "end": end,
            "text": messages[message_id][start:end],
        }


def tree_structure(heads):
    n = len(heads)
    if not n:
        raise ValueError("empty_tree")
    if any(type(h) is not int or h < 0 or h > n or h == i + 1 for i, h in enumerate(heads)):
        raise ValueError("invalid_head")
    if heads.count(0) != 1:
        raise ValueError("non_single_root")
    children = [[] for _ in heads]
    for i, h in enumerate(heads):
        if h:
            children[h - 1].append(i)
    depths = {}
    for i in range(n):
        path, seen, node = [], set(), i
        while node not in depths:
            if node in seen:
                raise ValueError("dependency_cycle")
            seen.add(node)
            path.append(node)
            if heads[node] == 0:
                depths[node] = 0
                break
            node = heads[node] - 1
        for node in reversed(path):
            if heads[node]:
                depths[node] = depths[heads[node] - 1] + 1
    return children, max(depths.values())


def span(p, indices):
    ids = sorted(set(indices))
    return {
        "token_indices": ids,
        "start": min(p["starts"][i] for i in ids),
        "end": max(p["ends"][i] for i in ids),
    }


def component(p, children, root):
    ids, pending = [], [root]
    while pending:
        node = pending.pop()
        ids.append(node)
        pending.extend(i for i in children[node] if p["dep"]["label"][i] not in {"COO", "IS", "WP"})
    return dict(span(p, ids), head=root)


def structure(p):
    """Conservative clause partition: own-subject event coordination vs shared-subject predicates."""
    heads, labels, pos = p["dep"]["head"], p["dep"]["label"], p["pos"]
    children, depth = tree_structure(heads)
    # Laughter and "emmm" are parsed as subjects or fronted objects; they are not constituents.
    vocal = {i for i, word in enumerate(p["cws"]) if vocal_token(word)}

    def subjects(v):
        return [i for i in children[v] if labels[i] == "SBV" and i not in vocal]

    # Predicate candidates require a structural attachment; attributive verbs alone do not qualify.
    predicates = {
        i
        for i, tag in enumerate(pos)
        if (
            (
                tag in {"v", "a"}
                and (
                    heads[i] == 0
                    or labels[i] in {"COO", "IS", "VOB", "IOB", "FOB", "CMP", "DBL"}
                    or (labels[i] == "ADV" and (subjects(i) or p["sdp"]["label"][i].startswith("d")))
                )
            )
            or (subjects(i) and tag in NOUN_POS | {"m", "q", "r"})
        )
    }
    root = heads.index(0)
    main = {root} if root in predicates else set()
    embedded, unresolved = set(), set()
    groups = {root: [root]} if main else {}
    event_links = []
    pending = predicates - main
    while pending:
        progressed = False
        for v in sorted(pending):
            h = heads[v] - 1
            if h in pending:
                continue
            pending.remove(v)
            progressed = True
            sdph, sdpl = p["sdp"]["head"][v] - 1, p["sdp"]["label"][v]
            owner = next((k for k, values in groups.items() if h in values), None)
            if labels[v] in {"COO", "IS"} and owner is not None:
                event_ok = sdph == h and sdpl in EVENT_RELATIONS
                event_links.append(
                    {"predicate": v, "head": h, "dep": labels[v], "sdp": sdpl, "aligned_event": event_ok}
                )
                if labels[v] == "IS" or (subjects(v) and event_ok):
                    groups[v] = [v]
                    main.add(v)
                elif not subjects(v) and subjects(owner) and event_ok:
                    groups[owner].append(v)
                else:
                    unresolved.add(v)
            elif labels[v] in {"VOB", "IOB", "FOB", "CMP", "ADV"} and (
                sdpl.startswith(("d", "r")) or labels[v] == "CMP"
            ):
                embedded.add(v)
            else:
                unresolved.add(v)
        if not progressed:
            unresolved.update(pending)
            break
    # DBL/serial and unresolved event links do not establish independent clause boundaries.
    if any(label == "DBL" for label in labels):
        unresolved.add(next(i for i, label in enumerate(labels) if label == "DBL"))
    sentence_type = (
        "unknown" if unresolved else "complex" if len(groups) > 1 else "simple" if groups else "fragment"
    )
    clauses = []
    for v, members in sorted(groups.items()):
        ids = [i for i in range(len(pos)) if i == v]
        for i in range(len(pos)):
            node = i
            while heads[node] and node not in groups:
                node = heads[node] - 1
            if node == v:
                ids.append(i)
        clauses.append(
            dict(
                span(p, ids),
                predicates=members,
                subject_state="present" if any(subjects(k) for k in members) else "not_detected",
            )
        )
    observations = []
    for v in sorted(predicates):
        sv = subjects(v)
        ov = [i for i in children[v] if labels[i] in OBJECT_RELATIONS and i not in vocal]
        orders = ["".join(k for _, k in sorted(((s, "S"), (v, "V"), (o, "O")))) for s in sv for o in ov]
        observations.append(
            {
                "predicate": v,
                "pos": pos[v],
                "subjects": [component(p, children, s) for s in sv],
                "objects": [dict(component(p, children, o), relation=labels[o]) for o in ov],
                "combination": ("S" if sv else "") + "V" + ("O" if ov else ""),
                "orders": orders,
                "pre_adverbs": sum(labels[i] == "ADV" and i < v for i in children[v]),
                "coordinated": labels[v] == "COO",
                "embedded": v in embedded,
            }
        )
    edges = [
        {
            "dependent": i,
            "head": h - 1,
            "label": labels[i],
            "signed_distance": i - (h - 1),
            "absolute_distance": abs(i - (h - 1)),
        }
        for i, h in enumerate(heads)
        if h
    ]
    distances = [e["absolute_distance"] for e in edges]
    nonleaf = [len(c) for c in children if c]
    return {
        "state": "ok",
        "sentence_type": sentence_type,
        "unresolved_predicates": sorted(unresolved),
        "embedded_predicates": sorted(embedded),
        "clauses": clauses,
        "predicates": observations,
        "event_links": event_links,
        "edges": edges,
        "tree_depth": depth,
        "nonleaf_branching": mean(nonleaf) if nonleaf else None,
        "all_node_branching": sum(map(len, children)) / len(children),
        "mean_dependency_distance": mean(distances) if distances else None,
        "max_dependency_distance": max(distances) if distances else None,
    }


def connections(sentence_observations, parsed):
    """Adjacent confirmed clauses; keep uncertain boundaries as unknown opportunities."""
    units = []
    for obs in sentence_observations:
        s = obs["structure"]
        if s["state"] != "ok" or s["sentence_type"] == "unknown":
            units.append((obs, None))
        else:
            units.extend((obs, c) for c in s["clauses"])
    result = []
    if len(units) == 1 and units[0][1] is None:
        return [{"state": "unknown", "forms": [], "reason": "unresolved_clause_boundary"}]
    for (left, a), (right, b) in zip(units, units[1:]):
        if a is None or b is None:
            result.append({"state": "unknown", "forms": []})
            continue
        forms, uncertain = [], False
        for obs, clause in ((left, a), (right, b)):
            p = parsed[obs["text"]]
            for i in clause["token_indices"]:
                if p["pos"][i] != "c":
                    continue
                dh, sh = p["dep"]["head"][i] - 1, p["sdp"]["head"][i] - 1
                if (
                    obs is right
                    and clause is b
                    and i < min(clause["predicates"])
                    and dh in clause["predicates"]
                    and sh in clause["predicates"]
                    and p["sdp"]["label"][i] == "mRELA"
                ):
                    evidence = {
                        "message_id": obs["message_id"],
                        "start": obs["start"] + p["starts"][i],
                        "end": obs["start"] + p["ends"][i],
                        "quote": p["cws"][i],
                    }
                    if evidence not in forms:
                        forms.append(evidence)
                elif dh in clause["predicates"] or sh in clause["predicates"]:
                    uncertain = True
                # A conjunction attached to a noun is an intra-clause connector.
        result.append(
            {
                "state": "unknown" if uncertain else "present" if forms else "absent",
                "forms": forms,
                "left": {"message_id": left["message_id"], "start": left["start"] + a["start"]},
                "right": {"message_id": right["message_id"], "start": right["start"] + b["start"]},
            }
        )
    return result


def pos_group(tag):
    groups = {
        "NOUN": "N NR NS NT NZ NN",
        "VERB": "V VV VA VC VE VN",
        "ADJ": "A AD AN AG",
        "ADV": "D DG",
        "PREP": "P PBA PBEI",
        "CONJ": "C CC CS",
        "PARTICLE": "U UDE1 UDE2 UDE3 UDENG UDSC UG UJ UL USUO UV UZ",
        "NUM": "M MQ CD OD",
        "PRON": "R PN DT",
        "PUNCT": "WP PU W",
    }
    return next((group for group, tags in groups.items() if tag.upper() in tags.split()), "OTHER")


def dep_group(label):
    groups = {
        "CORE_HEAD": {"ROOT", "HED"},
        "ARG_CORE": {"SBV", "VOB", "IOB", "FOB", "DBL"},
        "ATTR": {"ATT"},
        "ADVERBIAL": {"ADV", "CMP"},
        "PREP_OBJ": {"POB"},
        "AUX_ATTACH": {"LAD", "RAD"},
        "COORD": {"COO"},
        "INDEP": {"IS"},
    }
    return next((group for group, labels in groups.items() if label.upper() in labels), "OTHER")


def skeleton(p):
    result = []
    for i, (pos, label, head) in enumerate(zip(p["pos"], p["dep"]["label"], p["dep"]["head"], strict=True)):
        distance = head - i - 1
        bucket = (
            "ROOT"
            if head == 0
            else ("R" if distance > 0 else "L") + (str(abs(distance)) if abs(distance) < 3 else "3+")
        )
        result.append(f"{pos_group(pos)}:{dep_group(label)}:{bucket}")
    return " ".join(result)


def sequence(p, kind):
    if kind == "pos":
        return p["pos"]
    if kind in ("dep", "sdp"):
        return p[kind]["label"]
    return [
        f"{pos}:{label}:{'ROOT' if not h else ('R' if h > i + 1 else 'L')}"
        for i, (pos, label, h) in enumerate(zip(p["pos"], p["dep"]["label"], p["dep"]["head"], strict=True))
    ]


def ngram_counts(seq, n):
    return Counter(" ".join(seq[i : i + n]) for i in range(len(seq) - n + 1))


def numeric(values):
    valid = [v for v in values if v is not None]
    return {
        "values": values,
        "n": len(values),
        "valid_n": len(valid),
        "mean": mean(valid) if valid else None,
        "variance": pvariance(valid) if valid else None,
    }


def categorical(values, valid_labels=None):
    counts = Counter(values)
    valid = (
        sum(counts[k] for k in valid_labels)
        if valid_labels is not None
        else sum(v for k, v in counts.items() if k not in {"unknown", "na", "failure"})
    )
    return {
        "counts": counts,
        "rates": probabilities(counts),
        "n": len(values),
        "valid_n": valid,
        "coverage": ratio(valid, len(values)),
    }


def template_stats(counts):
    n = sum(counts.values())
    entropy = -sum(c / n * math.log(c / n) for c in counts.values()) if n else None
    return {
        "counts": counts,
        "n": n,
        "duplicate_rate": 1 - len(counts) / n if n > 1 else 0 if n else None,
        "entropy_nats": entropy,
        "normalized_entropy": entropy / math.log(len(counts))
        if len(counts) > 1
        else 0
        if n > 1
        else 1
        if n
        else None,
        "top_10": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]),
    }


def summarize(records, parsed):
    observations, message_lengths, window_lengths, links = [], [], [], []
    positions = {}
    empty = 0
    for row in records:
        window_chars = window_words = 0
        for msg in row["messages"]:
            p = parsed[msg["text"]]
            words = sum(word_token(t, tag) for t, tag in zip(p["cws"], p["pos"], strict=True))
            message_lengths.append(
                {
                    "sample_id": row["sample_id"],
                    "message_id": msg["message_id"],
                    "characters": len(msg["text"]),
                    "words": words,
                }
            )
            window_chars += len(msg["text"])
            window_words += words
            empty += not p["cws"]
        window_lengths.append(
            {"sample_id": row["sample_id"], "characters": window_chars, "words": window_words}
        )
        per_window = []
        for seg in segments_for(row, parsed):
            p = parsed[seg["text"]]
            try:
                struct = structure(p)
            except ValueError as exc:
                struct = {"state": "failure", "reason": str(exc)}
            obs = dict(
                seg,
                sample_id=row["sample_id"],
                characters=len(seg["text"]),
                nonspace_characters=sum(not c.isspace() for c in seg["text"]),
                cws_tokens=len(p["cws"]),
                words=sum(word_token(t, tag) for t, tag in zip(p["cws"], p["pos"], strict=True)),
                structure=struct,
            )
            observations.append(obs)
            per_window.append(obs)
            add_positions(positions, p["dep"]["label"])
        pairs = connections(per_window, parsed)
        links.append({"sample_id": row["sample_id"], "pairs": pairs, "state": "extracted" if pairs else "na"})
    structs = [o["structure"] for o in observations if o["structure"]["state"] == "ok"]
    preds = [v for s in structs for v in s["predicates"]]
    clauses = [c for s in structs if s["sentence_type"] != "unknown" for c in s["clauses"]]
    edges = [e for s in structs for e in s["edges"]]
    pairs = [p for row in links for p in row["pairs"]]
    lengths = {
        key: numeric([o[key] for o in observations])
        for key in ("characters", "nonspace_characters", "words", "cws_tokens")
    }
    metrics = {
        "037": lengths,
        "038": {k: {"mean": v["mean"], "n": v["n"]} for k, v in lengths.items()},
        "039": {k: {"population_variance": v["variance"], "n": v["n"]} for k, v in lengths.items()},
        "040": {
            k: {"counts": Counter(map(str, v["values"])), "values": v["values"]} for k, v in lengths.items()
        },
        "041": categorical(
            [o["structure"].get("sentence_type", "failure") for o in observations],
            {"simple", "complex", "fragment"},
        ),
        "044": categorical([v["combination"] for v in preds]),
        "045": categorical([c["subject_state"] for c in clauses]),
        "046": dict(
            categorical(["present" if v["objects"] else "not_detected" for v in preds]),
            objects_per_predicate=numeric([len(v["objects"]) for v in preds]),
            coordinated_predicates=sum(v["coordinated"] for v in preds),
        ),
        "048": dict(
            categorical([p["state"] for p in pairs]),
            forms=Counter(e["quote"] for p in pairs if p["state"] == "present" for e in p["forms"]),
            na_windows=sum(r["state"] == "na" for r in links),
        ),
        "061": dict(
            categorical([order for v in preds for order in v["orders"]]),
            component_combinations=Counter(v["combination"] for v in preds),
            complete_predicates=sum(bool(v["orders"]) for v in preds),
            total_predicates=len(preds),
        ),
        "067": numeric([v["pre_adverbs"] for v in preds]),
        "068-069": categorical([label for o in observations for label in parsed[o["text"]]["dep"]["label"]]),
        "070": {"counts": positions},
        "071": numeric([o["structure"].get("tree_depth") for o in observations]),
        "072": {
            "nonleaf": numeric([o["structure"].get("nonleaf_branching") for o in observations]),
            "all_nodes": numeric([o["structure"].get("all_node_branching") for o in observations]),
        },
        "073": numeric([o["structure"].get("mean_dependency_distance") for o in observations]),
        "074": numeric([o["structure"].get("max_dependency_distance") for o in observations]),
        "075": template_stats(
            Counter(skeleton(parsed[o["text"]]) for o in observations if o["structure"]["state"] == "ok")
        ),
    }
    metrics["041"]["simple_to_complex_ratio"] = ratio(
        metrics["041"]["counts"]["simple"], metrics["041"]["counts"]["complex"]
    )
    metrics["045"].update(
        unknown_segments=sum(s["sentence_type"] == "unknown" for s in structs),
        na_segments=sum(s["sentence_type"] == "fragment" for s in structs),
        failed_segments=len(observations) - len(structs),
        boundary_coverage=ratio(
            sum(s["sentence_type"] in {"simple", "complex"} for s in structs), len(observations)
        ),
        explicit_subject_rate=ratio(sum(c["subject_state"] == "present" for c in clauses), len(clauses)),
    )
    metrics["044"]["complete_svo_rate"] = ratio(
        sum(bool(v["subjects"]) and bool(v["objects"]) for v in preds), len(preds)
    )
    metrics["046"]["explicit_object_rate"] = ratio(sum(bool(v["objects"]) for v in preds), len(preds))
    metrics["048"]["explicit_connection_rate"] = ratio(
        sum(p["state"] == "present" for p in pairs), sum(p["state"] in {"present", "absent"} for p in pairs)
    )
    for item, relation in DISTANCE_RELATIONS.items():
        metrics[item] = {
            axis: numeric([e[axis + "_distance"] for e in edges if e["label"] == relation])
            for axis in ("signed", "absolute")
        }
    metrics["076"] = {}
    for kind in ("pos", "dep", "sdp"):
        metrics["076"][kind] = {}
        for n in range(1, DEFAULT_NGRAM_ORDER + 1):
            counts = Counter()
            for obs in observations:
                counts.update(ngram_counts(sequence(parsed[obs["text"]], kind), n))
            metrics["076"][kind][str(n)] = {"counts": counts, "n": sum(counts.values())}
    return {
        "n_windows": len(records),
        "n_messages": len(message_lengths),
        "empty_messages": empty,
        "n_sentences": len(observations),
        "parse_coverage": ratio(len(structs), len(observations)),
        "parse_failures": Counter(
            o["structure"].get("reason") for o in observations if o["structure"]["state"] != "ok"
        ),
        "metrics": metrics,
        "sentences": observations,
        "messages": message_lengths,
        "windows": window_lengths,
        "connections": links,
    }


def validate_units(data, row):
    units = data["units"]
    if not isinstance(units, list):
        raise ValueError("units must be an array")
    messages = {m["message_id"]: m["text"] for m in row["messages"]}
    order = {m["message_id"]: i for i, m in enumerate(row["messages"])}
    covered, validated = defaultdict(set), []
    previous = (-1, -1)
    for unit in units:
        e = locate(unit["evidence"], row["messages"])
        position = (order[e["message_id"]], e["start"])
        if position <= previous:
            raise ValueError("units must be in original order")
        previous = position
        if unit["completeness"] not in {
            "complete",
            "elliptical",
            "fragment",
            "response_particle",
            "unknown",
        } or unit["form"] not in {
            "declarative",
            "interrogative",
            "exclamative",
            "imperative",
            "other",
            "unknown",
        }:
            raise ValueError("Invalid completeness or form")
        indices = set(range(e["start"], e["end"]))
        if covered[e["message_id"]] & indices:
            raise ValueError("Overlapping semantic units")
        covered[e["message_id"]].update(indices)
        result = dict(unit, evidence=e)
        if unit["completeness"] in {"complete", "elliptical"}:
            word_order = unit.get("order", "canonical")
            if word_order not in {"canonical", "fronted", "dislocated", "unknown"}:
                raise ValueError("Invalid unit order")
        elif unit.get("order", "na") in {"fronted", "dislocated"}:
            raise ValueError("Only complete or elliptical units have a word order")
        else:
            word_order = "unknown" if unit["completeness"] == "unknown" else "na"
        if word_order in {"fronted", "dislocated"}:
            moved = locate(unit["moved"], row["messages"])
            if moved["message_id"] != e["message_id"] or not (
                e["start"] <= moved["start"] < moved["end"] <= e["end"]
                and (moved["start"], moved["end"]) != (e["start"], e["end"])
            ):
                raise ValueError("Moved constituent must lie inside its unit")
            result["moved"] = moved
        elif "moved" in unit:
            raise ValueError("Only fronted or dislocated units carry a moved constituent")
        result["order"] = word_order
        validated.append(result)
    for mid, text in messages.items():
        if any(not c.isspace() and i not in covered[mid] for i, c in enumerate(text)):
            raise ValueError("Semantic units do not cover every nonspace character")
    return validated


def validate_self_qa(data, row):
    value = data["self_qa"]
    state, pairs = value["state"], value["pairs"]
    if state not in {"present", "absent", "unknown", "na"} or not isinstance(pairs, list):
        raise ValueError("Invalid self_qa fields")
    if (state == "present") != bool(pairs):
        raise ValueError("Inconsistent self_qa state")
    if state == "na" and any(m["text"].strip() for m in row["messages"]):
        raise ValueError("Nonempty reply cannot be NA for self_qa")
    order = {m["message_id"]: i for i, m in enumerate(row["messages"])}
    validated, seen = [], set()
    for pair in pairs:
        q, a = [locate(pair[k], row["messages"]) for k in ("question", "answer")]
        if (order[q["message_id"]], q["end"]) > (order[a["message_id"]], a["start"]):
            raise ValueError("Self-answer must follow its question")
        key = (q["message_id"], q["start"], q["end"], a["message_id"], a["start"], a["end"])
        if key in seen:
            raise ValueError("Duplicate pair or invalid function")
        seen.add(key)
        validated.append(dict(pair, question=q, answer=a))
    return dict(value, pairs=validated)


def validate_response(response, row):
    result, errors = {}, {}
    for key, validator in (("units", validate_units), ("self_qa", validate_self_qa)):
        try:
            if response["state"] != "ok":
                raise ValueError(response.get("error", "Call failed"))
            result[key] = validator(response["data"], row)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            errors[key] = str(exc)
    return result, errors


def word_count(e, row, parsed):
    text = next(m["text"] for m in row["messages"] if m["message_id"] == e["message_id"])
    p = parsed[text]
    return sum(
        word_token(t, tag) and start < e["end"] and end > e["start"]
        for t, tag, start, end in zip(p["cws"], p["pos"], p["starts"], p["ends"], strict=True)
    )


def annotate(row, parsed, judge, refresh=False, short_words=DEFAULT_SHORT_WORDS):
    judge = judge.for_record(row)
    payload = {"context": row["context"], "messages": row["messages"]}
    response = judge.call(ANNOTATION_PROMPT, payload, refresh=refresh)
    validated, errors = validate_response(response, row)
    initial_errors = dict(errors)
    keys = dict.fromkeys(("units", "self_qa"), response["cache_key"])
    repair = None
    if errors:
        repair = judge.call(
            ANNOTATION_PROMPT,
            payload,
            refresh=refresh,
            attempt=1,
        )
        fixed, failed = validate_response(repair, row)
        for key in list(errors):
            keys[key] = repair["cache_key"]
            if key in fixed:
                validated[key] = fixed[key]
                del errors[key]
            else:
                errors[key] = failed[key]
    if errors:
        judge.invalidate(response["cache_key"], repair["cache_key"] if repair else None)
    items = {}
    for item in ITEMS:
        source = "self_qa" if item == "056" else "units"
        if source in errors:
            items[item] = {"state": "failure", "events": [], "reason": errors[source]}
            continue
        if item == "056":
            value = validated[source]
            items[item] = {
                "state": value["state"],
                "reason": value["reason"],
                "events": [
                    dict(p, label="self_question_answer", evidence=p["question"]) for p in value["pairs"]
                ],
            }
            continue
        eligible = validated["units"]
        if item == "060":
            eligible = [u for u in eligible if 0 < word_count(u["evidence"], row, parsed) <= short_words]
        if item in ORDER_ITEMS:
            eligible = [u for u in eligible if u["order"] != "na"]
        events, unknown = [], 0
        for unit in eligible:
            label = (
                unit["completeness"]
                if item in {"043", "060"}
                else unit["order"]
                if item in ORDER_ITEMS
                else unit["form"]
            )
            unknown += label == "unknown"
            if label == "unknown":
                continue
            if item == "043" and label == "response_particle":
                label = "fragment"
            if item in {"043", "060"} or label in ITEMS[item]:
                events.append(
                    {
                        "label": label,
                        "evidence": unit["evidence"],
                        "word_count": word_count(unit["evidence"], row, parsed),
                        **({"moved": unit["moved"]} if item in ORDER_ITEMS else {}),
                    }
                )
        items[item] = {
            "state": "present" if events else "unknown" if unknown else "absent" if eligible else "na",
            "events": events,
            "reason": "uncertain semantic units" if unknown else "no eligible units" if not eligible else "",
            "n_units": len(eligible),
            "unknown_units": unknown,
            "valid_units": len(eligible) - unknown,
        }
        if item in ORDER_ITEMS:
            # Word order applies only to messages with a complete or elliptical unit.
            applicable = {u["evidence"]["message_id"] for u in eligible}
            items[item]["message_ids"] = [
                m["message_id"] for m in row["messages"] if m["message_id"] in applicable
            ]
    return {
        "sample_id": row["sample_id"],
        "items": items,
        "units": validated.get("units", []),
        "message_ids": [m["message_id"] for m in row["messages"] if m["text"].strip()],
        "cache_key": response["cache_key"],
        "repair_cache_key": repair["cache_key"] if repair else None,
        "initial_failures": initial_errors,
        "remaining_failures": errors,
        "item_cache_keys": {item: keys["self_qa" if item == "056" else "units"] for item in ITEMS},
    }


def summarize_semantic(annotations, texts):
    """`texts` maps (sample_id, message_id) to the nonempty target messages, for the length strata."""
    result = {}
    for item in ITEMS:
        values = [a["items"][item] for a in annotations]
        counts = Counter(v["state"] for v in values)
        labels = Counter(e["label"] for v in values for e in v["events"])
        n_valid = counts["present"] + counts["absent"]
        valid_units = sum(v.get("valid_units", 0) for v in values)
        valid = [
            (a, v) for a, v in zip(annotations, values, strict=True) if v["state"] in ("present", "absent")
        ]
        messages = sum(len(v.get("message_ids", a["message_ids"])) for a, v in valid)
        hits = {(a["sample_id"], e["evidence"]["message_id"]) for a, v in valid for e in v["events"]}
        valid_texts = {
            (a["sample_id"], m): texts[(a["sample_id"], m)]
            for a, v in valid
            for m in v.get("message_ids", a["message_ids"])
        }
        missing = {"unknown": 0, "failure": 0}
        for a, v in zip(annotations, values, strict=True):
            if v["state"] in missing:
                missing[v["state"]] += len(v.get("message_ids", a["message_ids"]))
        result[item] = {
            "states": dict(counts),
            **(
                {}
                if item == "043"
                else {
                    "valid_messages": messages,
                    "hit_messages": len(hits),
                    "message_occurrence_rate": ratio(len(hits), messages),
                    "by_length": {"message_occurrence_rate": by_length(valid_texts, hits)},
                    "reliability": {
                        "message_occurrence_rate": with_missing(
                            reliability(valid_texts, hits), missing["unknown"], missing["failure"]
                        )
                    },
                }
            ),
            "n_windows": len(values),
            "valid_windows": n_valid,
            "coverage": ratio(n_valid, len(values)),
            "window_occurrence_rate": ratio(counts["present"], n_valid),
            "labels": labels,
            "unit_rates": {k: ratio(v, valid_units) for k, v in labels.items()} if item != "056" else {},
            "valid_units": valid_units,
            "unknown_units": sum(v.get("unknown_units", 0) for v in values),
            "initial_failure_count": sum(
                ("self_qa" if item == "056" else "units") in a["initial_failures"] for a in annotations
            ),
            "repaired_count": sum(
                ("self_qa" if item == "056" else "units") in a["initial_failures"] and v["state"] != "failure"
                for a, v in zip(annotations, values, strict=True)
            ),
        }
    for item in ORDER_ITEMS:
        applicable = sum(len(a["items"][item].get("message_ids", [])) for a in annotations)
        result[item]["applicable_messages"] = applicable
        result[item]["applicability_rate"] = ratio(
            applicable, sum(len(a["message_ids"]) for a in annotations if "message_ids" in a["items"][item])
        )
        moved = [(e["evidence"], e["moved"]) for a in annotations for e in a["items"][item]["events"]]
        # Center of the moved constituent within its unit, 0 = unit start and 1 = unit end.
        result[item]["moved_position"] = numeric(
            [(m["start"] + m["end"] - 2 * u["start"]) / (2 * (u["end"] - u["start"])) for u, m in moved]
        )
        result[item]["moved_characters"] = numeric([m["end"] - m["start"] for _, m in moved])
    message_states = Counter()
    question_texts, question_hits = {}, set()
    for annotation in annotations:
        by_message = {}
        for unit in annotation["units"]:
            by_message.setdefault(unit["evidence"]["message_id"], []).append(unit["form"])
        for mid in annotation.get("message_ids", by_message):
            if mid not in by_message:
                message_states["failure"] += 1
        for mid, forms in by_message.items():
            state = "unknown" if "unknown" in forms else "present" if "interrogative" in forms else "absent"
            message_states[state] += 1
            key = (annotation["sample_id"], mid)
            if state != "unknown":
                question_texts[key] = texts[key]
            if state == "present":
                question_hits.add(key)
    result["052,211"].update(
        message_form_states=dict(message_states),
        valid_messages=message_states["present"] + message_states["absent"],
        hit_messages=message_states["present"],
        message_occurrence_rate=ratio(
            message_states["present"], message_states["present"] + message_states["absent"]
        ),
        message_question_rate=ratio(
            message_states["present"], message_states["present"] + message_states["absent"]
        ),
        by_length={"message_occurrence_rate": by_length(question_texts, question_hits)},
        reliability={
            "message_occurrence_rate": with_missing(
                reliability(question_texts, question_hits),
                message_states["unknown"],
                message_states["failure"],
            )
        },
        message_form_coverage=ratio(
            message_states["present"] + message_states["absent"], sum(message_states.values())
        ),
        sentence_question_rate=result["052,211"]["unit_rates"].get("interrogative", 0.0)
        if result["052,211"]["valid_units"]
        else None,
    )
    result["semantic_lengths"] = {
        axis: numeric(
            [
                len(u["evidence"]["quote"])
                if axis == "characters"
                else sum(not c.isspace() for c in u["evidence"]["quote"])
                for a in annotations
                for u in a["units"]
            ]
        )
        for axis in ("characters", "nonspace_characters")
    }
    return result


def extract(
    corpora,
    parsed,
    judge,
    output,
    workers,
    qa_runner=None,
    short_words=DEFAULT_SHORT_WORDS,
):
    annotations, summaries, semantic = {}, {}, {}
    for name, records in corpora.items():
        dest = output / name
        dest.mkdir()
        summaries[name] = summarize(records, parsed)
        write_json(dest / "syntactic_style.json", summaries[name])
        write_json(dest / "records.json", records)
    if judge:
        print(f"Annotating syntax for {distinct_records(corpora)} windows", flush=True)
        annotations = annotate_corpora(
            corpora, lambda row: annotate(row, parsed, judge, short_words=short_words), workers, judge=judge
        )
        for name, records in corpora.items():
            semantic[name] = summarize_semantic(annotations[name], message_texts(records))
            write_json(output / name / "annotations.json", annotations[name])
            write_json(output / name / "semantic.json", semantic[name])
            if qa_runner and name == PRIMARY_CORPUS:
                write_json(output / name / "qa.json", qa_runner(records, annotations[name]))
    return semantic


def extraction_coverage(semantic, judge):
    coverage = {
        f"{i:03}": {"state": "not_implemented", "reason": "high difficulty"}
        if f"{i:03}" in HIGH_DIFFICULTY
        else {"state": "not_run", "reason": REMOVED[f"{i:03}"]}
        if f"{i:03}" in REMOVED
        else {"state": "extracted"}
        for i in range(37, 77)
    }
    for item in ITEMS:
        coverage[item] = {
            "state": "extracted" if judge else "not_run",
            "corpora": {name: value[item]["states"] for name, value in semantic.items()},
        }
    for key in ("052", "058", "059", "062", "068", "069"):
        coverage.pop(key, None)
    coverage["068-069"] = {"state": "extracted"}
    return coverage


def protocol(short_words=DEFAULT_SHORT_WORDS):
    return {
        "sentence_boundary": "Same operational segments as chapter 2; each substring reparsed independently",
        "characters": "Unicode code points of raw segment including punctuation and interior whitespace; nonspace count separately",
        "words": "non-wp tokens containing Unicode letter/number; cws count incl punctuation separately",
        "variance": "population variance, denominator N; singleton 0, empty null",
        "clause_rules": "dep COO/IS plus aligned SDP event relations and own SBV; coordinated predicates with shared explicit subject merged; DBL/ambiguous serial structure unknown",
        "subject_object": "explicit syntactic SBV/VOB/IOB/FOB; vocal tokens (lexical_choice.vocal_token) never subjects or objects; never infer ellipsis from missing edges",
        "word_order": "057 dislocated and 058-059,062 fronted from the semantic-unit annotation; omitted order = canonical for complete/elliptical units; moved constituent inside its unit; moved_position = center / unit length",
        "removed": REMOVED,
        "distance": "dependent index minus head index; absolute distance separately; artificial ROOT excluded; punctuation retained",
        "tree_depth": "lexical root depth 0; no virtual root; cycles/out-of-range/self heads invalid",
        "branching": "nonleaf lexical nodes primary, all lexical nodes separately; singleton nonleaf null",
        "skeleton": "POS group:DEP group:head direction and distance 1/2/3+ over operational segments, short input retained; groups keep the original extractor's mappings",
        "ngram_orders": list(range(1, DEFAULT_NGRAM_ORDER + 1)),
        "ngram_boundary": "operational sentence; raw POS/DEP/SDP labels",
        "short_sentence_max_words": short_words,
        "short_sentence_threshold_basis": "operational cutoff, not linguistic fact; all lengths retained",
        "semantic_units": "LLM spans with full nonspace original-text coverage; distinct from rule segments",
        "semantic_labels": ITEMS,
    }
