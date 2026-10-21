"""Project shared extraction artifacts into the current checklist categories."""

import os

from weclone.utils import secure_storage

from .data import PRIMARY_CORPUS, UNITS, write_json
from .stats import LENGTH_STRATA, RELIABILITY
from .taxonomy import CATALOG, CATEGORIES, PARTS, PROTOCOL_VERSION, extractor_for, metadata, parts
from .variation import group_variation, summary_counts


LEXICAL_FIELDS = {
    "001": ("words", "top_words"),
    "002": ("function_words", "function_rate", "word_tokens"),
    "017": ("personal_pronouns", "personal_pronoun_rate", "word_tokens"),
    "020": ("diversity",),
    "021": ("word_lengths", "mean_word_length", "word_tokens"),
    "022": ("top_mass", "word_tokens"),
    "023": ("zipf",),
    "024": ("word_ngrams", "han_character_ngrams"),
}
BURST_FIELDS = (
    "units",
    "windows",
    "nonempty_messages",
    "bubbles_per_message",
    "message_newline_burst_rate",
    "by_length",
    "reliability",
    "messages_per_window",
    "bubbles_per_window",
    "window_newline_burst_rate",
)
LONG_FIELDS = (
    "long_characters",
    "nonempty_messages",
    "length_summary",
    "message_long_rate",
    "window_long_rate",
    "message_length_counts",
    "message_length_shares",
    "reliability",
    "message_lengths",
    "bubble_lengths",
    "window_lengths",
)
MEASURED_STATES = {"extracted", "implemented", "complete"}
# Most to least complete; used only when the parts of a merged row disagree.
STATE_ORDER = ("partial", "unavailable", "not_run", "deferred", "not_implemented")


def read_json(path):
    return secure_storage.read_json(path) if secure_storage.file_exists(path) else {}


def select_fields(value, fields):
    """`fields` of `value`; its by_length and reliability keep only the rates that were selected."""
    selected = {key: value[key] for key in fields if key in value}
    for key in ("by_length", "reliability"):
        if isinstance(selected.get(key), dict):
            selected[key] = {rate: v for rate, v in selected[key].items() if rate in selected}
    return selected


def reliability_table(metrics):
    """Every `reliability` record of one corpus, keyed by item and the path of its rate."""
    items, classes = {}, dict.fromkeys(RELIABILITY["classes"], 0)

    def walk(item, node, path):
        found = node.get("reliability")
        for rate, value in (found if isinstance(found, dict) else {}).items():
            records = (
                {rate: value}
                if isinstance(value.get("class"), str)
                else {f"{rate}/{label}": record for label, record in value.items()}
            )
            for name, record in records.items():
                items.setdefault(item, {})["/".join((*path, name))] = record
                classes[record["class"]] += 1
        for key, child in node.items():
            if isinstance(child, dict) and key not in ("by_length", "reliability"):
                walk(item, child, (*path, str(key)))

    for item, entry in metrics.items():
        walk(item, entry.get("measurements", {}), ())
    return {"method": RELIABILITY, "classes": classes, "items": items}


def corpus_artifacts(directory, extractor):
    raw_name = "lexical" if extractor == "lexical_choice" else extractor
    raw = read_json(directory / (raw_name + ".json"))
    return {
        "raw": raw,
        "semantic": read_json(directory / "semantic.json"),
        "personal": read_json(directory / "personal.json"),
    }


def metric_values(key, artifacts):
    raw, semantic, personal = (artifacts[k] for k in ("raw", "semantic", "personal"))
    result = {}
    surface = raw.get("metrics", raw)
    if key in surface:
        result["surface"] = surface[key]
    if key in semantic:
        result["contextual"] = semantic[key]
    if key in personal:
        result["discovered_expressions"] = personal[key]
    if key in LEXICAL_FIELDS:
        result["surface"] = select_fields(raw, LEXICAL_FIELDS[key])
    if key == "burst_pattern":
        result["surface"] = select_fields(raw, BURST_FIELDS)
    if key == "146-150":
        result["surface"] = select_fields(raw, ("forms", "n_messages", "surface_only"))
    if key == "218":
        result["surface"] = select_fields(raw, LONG_FIELDS)
    return result


def merged_state(states):
    states = set(states)
    if len(states) == 1:
        return states.pop()
    if states <= MEASURED_STATES:
        return "extracted"
    if states & (MEASURED_STATES | {"partial"}):
        return "partial"
    return min(states, key=lambda s: STATE_ORDER.index(s) if s in STATE_ORDER else len(STATE_ORDER))


def part_status(key, manifests):
    module = extractor_for(key)
    status = manifests.get(module, {}).get("coverage", {}).get(key)
    return dict(
        extractor=module,
        **(
            status
            or {
                "state": "not_implemented",
                "reason": "No extractor implemented for this checklist item",
            }
        ),
    )


def part_values(key, module, artifacts):
    return metric_values(key, artifacts[module]) if module in artifacts else {}


def corpus_names_of(*manifest_groups):
    """Person first, then every group corpus seen by any extractor or surface pass."""
    names = dict.fromkeys(
        name for group in manifest_groups for m in group.values() for name in m.get("corpus_digests", {})
    )
    return sorted(names, key=lambda name: (name != PRIMARY_CORPUS, name))


def scope_of(manifests, name):
    return next(
        (m["corpus_scope"][name] for m in manifests.values() if name in m.get("corpus_scope", {})), None
    )


def write_report(
    output, categories, extractor_dirs, *, surface_dirs=None, sampling=None, window=None, grouping=None
):
    surface_dirs = surface_dirs or {}
    manifests = {module: read_json(path / "manifest.json") for module, path in extractor_dirs.items()}
    surface_manifests = {module: read_json(path / "manifest.json") for module, path in surface_dirs.items()}
    corpus_names = corpus_names_of(manifests, surface_manifests)
    semantic_run = any(m.get("judge_model") for m in manifests.values())
    corpus_metrics = {}
    coverage = {}
    for key, (_, category) in CATALOG.items():
        if category not in categories:
            continue
        statuses = {part: part_status(part, manifests) for part in parts(key)}
        if key not in PARTS:
            coverage[key] = dict(metadata(key), **statuses[key])
            continue
        modules = list(dict.fromkeys(s["extractor"] for s in statuses.values()))
        coverage[key] = dict(
            metadata(key),
            extractor=modules[0] if len(modules) == 1 else modules,
            state=merged_state(s["state"] for s in statuses.values()),
            parts=statuses,
        )
    for name in corpus_names:
        scope = {
            "units": "windows and nonempty target messages; rates are per message unless named window_*",
            "surface": scope_of(surface_manifests, name) or scope_of(manifests, name),
            "semantic": scope_of(manifests, name) if semantic_run else None,
        }
        artifacts = {module: corpus_artifacts(path / name, module) for module, path in extractor_dirs.items()}
        for module, path in surface_dirs.items():
            full = corpus_artifacts(path / name, module)
            artifacts[module]["raw"] = full["raw"]
            artifacts[module]["personal"] = full["personal"]
        metrics = {}
        for key, status in coverage.items():
            if key in PARTS:
                values = {
                    part: part_values(part, value["extractor"], artifacts)
                    for part, value in status["parts"].items()
                }
            else:
                values = part_values(key, status["extractor"], artifacts)
            metrics[key] = dict(status, measurements=values, measurement_scope=scope)
        directory = output / name
        directory.mkdir()
        write_json(directory / "metrics.json", metrics)
        corpus_metrics[name] = metrics
    reliability = (
        reliability_table(corpus_metrics[PRIMARY_CORPUS]) if PRIMARY_CORPUS in corpus_metrics else None
    )
    if reliability:
        write_json(output / "reliability.json", reliability)
    variation = None
    if len(corpus_metrics) > 1:
        variation = group_variation(corpus_metrics, grouping or {})
        write_json(output / "group_variation.json", variation)
    manifest = {
        "protocol": PROTOCOL_VERSION,
        "categories": {
            category: {
                "name": CATEGORIES[category],
                "items": [k for k, v in coverage.items() if v["category"] == category],
            }
            for category in categories
        },
        "coverage": coverage,
        "extractors": {
            module: str(path.relative_to(output) / "manifest.json") for module, path in extractor_dirs.items()
        },
        "corpora": corpus_names,
        "units": UNITS,
        "length_buckets": LENGTH_STRATA,
        "reliability": {
            "path": "reliability.json",
            "corpus": PRIMARY_CORPUS,
            "classes": reliability["classes"],
        }
        if reliability
        else None,
        "window": window,
        "grouping": grouping,
        "group_variation": {
            "path": "group_variation.json",
            "item_classes": summary_counts(variation),
        }
        if variation
        else None,
        "corpus_digests": next(iter(manifests.values()))["corpus_digests"] if manifests else {},
        "surface_corpus_digests": next(iter(surface_manifests.values()))["corpus_digests"]
        if surface_manifests
        else {},
        "sampling": {k: v for k, v in sampling.items() if not k.endswith("_ids")} if sampling else None,
        "sampling_path": "sampling.json" if sampling else None,
        "surface_extractors": {
            module: os.path.relpath(path / "manifest.json", output) for module, path in surface_dirs.items()
        },
        "engineering_smoke_test": any(m["engineering_smoke_test"] for m in manifests.values()),
        "limit": next(iter(manifests.values()))["limit"] if manifests else None,
        "report_policy": "One canonical entry per merged measure; report-level merged rows keep each extractor output under parts; antiphrasis is a subtype of 169; no summed score. Shared extractor artifacts retain evidence and diagnostics.",
    }
    write_json(output / "manifest.json", manifest)
    return manifest
