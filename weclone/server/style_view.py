"""Read-only view of a style extraction run for the web page."""

from pathlib import Path

from weclone.data.agent.distill_style.variation import RATE_KEYS, collect
from weclone.utils import secure_storage as secure

ROOT = Path(__file__).resolve().parents[2]
STYLE_ROOT = ROOT / "dataset/res_csv/agent/distill/style"
PRIMARY = "person"
# Counters keep only their most frequent entries; the full counts stay in the run directory.
TOP_ENTRIES = 12
SKIP_KEYS = {"by_topic", "by_length", "reliability", "sentences", "messages", "context_groups"}
STATE_KEYS = {"present", "absent", "unknown", "na", "failure"}
RELIABILITY_KEYS = (
    "interval",
    "class",
    "rare_outcome",
    "rare_upper",
    "effective_events",
    "bounds",
    "missing_share",
    "windows",
)
TEST_KEYS = (
    "test",
    "class",
    "rate_ratio",
    "q_value",
    "p_value",
    "chi2",
    "dof",
    "highest_group",
    "lowest_group",
    "rare_outcome",
    "testable_groups",
    "reason",
)


def complete(run: Path) -> bool:
    return secure.file_exists(run / "manifest.json") and secure.file_exists(run / "person" / "metrics.json")


def style_run_dir(path: Path | None = None) -> Path | None:
    """The given run, else the newest complete run that is neither a smoke test nor a --limit run."""
    if path is not None:
        return path if complete(path) else None
    runs = (
        sorted((run for run in STYLE_ROOT.iterdir() if run.is_dir()), reverse=True)
        if STYLE_ROOT.is_dir()
        else []
    )
    for run in runs:
        if complete(run):
            manifest = secure.read_json(run / "manifest.json")
            if not manifest.get("engineering_smoke_test") and manifest.get("limit") is None:
                return run
    return None


def rate_node(measurements: dict, path: str):
    """(node, key, label) holding the rate at a `variation.collect` path."""
    parts = path.split("/")
    trail, key, label = (parts[:-1], parts[-1], None) if parts[-1] in RATE_KEYS else (parts[:-2], *parts[-2:])
    node = measurements
    for part in trail:
        node = node[part]
    return node, key, label


def rate_cell(measurements: dict, path: str, hits: int, messages: int, strata) -> dict:
    cell = {"hits": hits, "messages": messages}
    if strata:
        cell["by_length"] = {bucket: [k, n] for bucket, (k, n) in strata.items()}
    node, key, label = rate_node(measurements, path)
    record = (node.get("reliability") or {}).get(key)
    if label is not None and isinstance(record, dict):
        record = record.get(label)
    if isinstance(record, dict):
        cell.update({name: record[name] for name in RELIABILITY_KEYS if name in record})
    return cell


def counters(value, path=(), depth=0):
    """Yield (path, counter) for every dict of integer counts below `value`."""
    if not isinstance(value, dict) or depth > 6:
        return
    if len(value) >= 2 and all(type(count) is int for count in value.values()):
        if not set(value) <= STATE_KEYS:
            yield "/".join(path), value
        return
    for key, child in value.items():
        if key not in SKIP_KEYS:
            yield from counters(child, (*path, str(key)), depth + 1)


def scalars(value, path=(), depth=0):
    if isinstance(value, dict) and depth <= 4:
        if len(value) >= 2 and all(type(count) is int for count in value.values()):
            return
        for key, child in value.items():
            if key not in SKIP_KEYS:
                yield from scalars(child, (*path, str(key)), depth + 1)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield "/".join(path), value


def distribution(path: str, counter: dict) -> dict:
    top = sorted(counter.items(), key=lambda entry: -entry[1])[:TOP_ENTRIES]
    return {"path": path, "total": sum(counter.values()), "size": len(counter), "top": top}


def contact_label(group: str, info: dict, sample_ids: list) -> str:
    # Chat sample IDs read "<person>:<chat file>:<record>:<segment>", the chat file being named after the contact.
    if len(info.get("contacts", [])) == 1 and info.get("group") == info["contacts"][0] and sample_ids:
        name = str(sample_ids[0]).split(":", 1)[-1].rsplit(":", 2)[0]
        return secure.logical_path(name).stem or info["group"]
    return str(info.get("group", group.removeprefix("group-")))


def compact(value):
    """Four significant digits are plenty for display and keep the response small."""
    if isinstance(value, float):
        return float(f"{value:.4g}")
    if isinstance(value, dict):
        return {key: compact(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact(child) for child in value]
    return value


def build_view(run: Path) -> dict:
    manifest = secure.read_json(run / "manifest.json")
    person = secure.read_json(run / "person" / "metrics.json")
    variation_path = run / "group_variation.json"
    variation = secure.read_json(variation_path) if secure.file_exists(variation_path) else {}
    sampling_path = run / "sampling.json"
    sampling = secure.read_json(sampling_path) if secure.file_exists(sampling_path) else {}
    corpora = (
        manifest.get("grouping", {}).get("corpora", {}) if manifest.get("grouping", {}).get("enabled") else {}
    )
    groups = {name: secure.read_json(run / name / "metrics.json") for name in corpora}
    collected = {PRIMARY: collect(person), **{name: collect(metrics) for name, metrics in groups.items()}}
    group_ids = sampling.get("group_ids", {})
    contacts = sorted(
        (
            {
                "id": name,
                "label": contact_label(name, info, group_ids.get(name, [])),
                "messages": info.get("messages"),
                "windows": info.get("windows"),
                "semantic_windows": sampling.get("groups", {}).get(name, {}).get("selected"),
            }
            for name, info in corpora.items()
        ),
        key=lambda contact: -(contact["messages"] or 0),
    )

    items = {}
    for item, coverage in manifest["coverage"].items():
        entry = person.get(item, {})
        measurements = entry.get("measurements", {})
        tested = variation.get("items", {}).get(item, {})
        leaves = []
        for path, (hits, messages, strata) in collected[PRIMARY].get(item, {}).items():
            test = tested.get("leaves", {}).get(path, {})
            cells = {}
            for name in groups:
                found = collected[name].get(item, {}).get(path)
                if not found:
                    continue
                cell = rate_cell(groups[name][item]["measurements"], path, *found)
                ratio = test.get("groups", {}).get(name, {})
                if "length_adjusted_ratio" in ratio:
                    cell.update(ratio=ratio["length_adjusted_ratio"], expected=ratio["expected_hits"])
                cell["tested"] = name in test.get("groups_tested", [])
                cells[name] = cell
            leaves.append(
                {
                    "path": path,
                    "source": next(
                        (part for part in path.split("/") if part in ("surface", "contextual")), "other"
                    ),
                    "person": rate_cell(measurements, path, hits, messages, strata),
                    "groups": cells,
                    "test": {name: test[name] for name in TEST_KEYS if name in test} if test else None,
                }
            )
        items[item] = {
            "id": item,
            "name": coverage.get("name", entry.get("name", item)),
            "category": coverage.get("category"),
            "state": coverage.get("state"),
            "reason": coverage.get("reason") or entry.get("reason"),
            "extractor": coverage.get("extractor"),
            "variation_class": tested.get("class"),
            "leaves": leaves,
            "distributions": [distribution(path, counter) for path, counter in counters(measurements)],
            "numbers": dict(scalars(measurements)) if not leaves else {},
        }

    scope = next(
        (entry["measurement_scope"] for entry in person.values() if "measurement_scope" in entry), {}
    )
    return compact(
        {
            "run": {
                "id": run.name,
                "protocol": manifest.get("protocol"),
                "window": manifest.get("window", {}),
                "scope": {key: scope.get(key) for key in ("surface", "semantic")},
                "sampling": {
                    key: sampling.get(key) for key in ("budget", "selected", "pooled_selected", "group_floor")
                },
            },
            "buckets": manifest.get("length_buckets", {}),
            "categories": [
                {"id": key, "name": value["name"], "items": value["items"]}
                for key, value in manifest.get("categories", {}).items()
            ],
            "contacts": contacts,
            "items": items,
        }
    )
