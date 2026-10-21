"""Which per-message rates stay stable across contact groups and which follow the relationship."""

from .data import PRIMARY_CORPUS

DEFAULT_MIN_EXPECTED = 5
DEFAULT_MIN_EVENTS = 30
DEFAULT_RATIO_THRESHOLD = 1.5
DEFAULT_FDR = 0.01
CLASS_ORDER = ("relationship_dependent", "uncertain", "stable", "insufficient")
RATE_KEYS = (
    "message_rate",
    "message_occurrence_rate",
    "message_advice_occurrence_rate",
    "message_newline_burst_rate",
    "message_long_rate",
    "no_punctuation_rate",
    "bubble_no_punctuation_rate",
    "unpunctuated_boundary_rate",
    "applicability_rate",
)
RATE_MAP_KEYS = {
    "message_rates": "message_counts",
    "label_message_rates": None,
    "message_weighted_proportions": "message_label_counts",
    "message_length_shares": "message_length_counts",
}
# These rates are message length itself, so within one length bucket there is nothing left to compare.
LENGTH_DEFINED = {"message_long_rate", "message_length_shares"}
NUMERATOR_KEYS = ("hit_messages", "matched_messages")
DENOMINATOR_KEYS = (
    "valid_messages",
    "message_denominator",
    "nonempty_messages",
    "n_messages",
    "labeled_messages",
    "eligible_messages",
)
SPECIAL = {
    "message_advice_occurrence_rate": (("advice_hit_messages",), ("advice_valid_messages",)),
    "applicability_rate": ((), ("applicability_denominator",)),
}


def matching(numerator, denominator, rate):
    return denominator > 0 and abs(numerator / denominator - rate) < 1e-9


def count_pair(node, rate, numerators, denominators, count=None):
    """(hits, messages) consistent with the reported rate, or None when no sibling fits."""
    candidates = [node[k] for k in denominators if isinstance(node.get(k), int) and node[k] > 0]
    known = (
        [count] if isinstance(count, int) else [node[k] for k in numerators if isinstance(node.get(k), int)]
    )
    for denominator in candidates:
        for numerator in known or [round(rate * denominator)]:
            if matching(numerator, denominator, rate):
                return numerator, denominator
    return None


def strata_of(node, key, label=None):
    """Per-bucket (hits, messages) kept under `by_length` next to a rate, if any."""
    value = node.get("by_length")
    value = value.get(key) if isinstance(value, dict) else None
    if label is not None:
        value = value.get(label) if isinstance(value, dict) else None
    if not isinstance(value, dict) or not value:
        return None
    return {bucket: (cell["hits"], cell["messages"]) for bucket, cell in value.items()}


def recover(node, rate, numerators, denominators, strata, count=None):
    """(hits, messages, strata): exact from the strata when they reproduce the rate, else from siblings."""
    if strata:
        hits, messages = sum(k for k, _ in strata.values()), sum(n for _, n in strata.values())
        if matching(hits, messages, rate) and (not isinstance(count, int) or hits == count):
            return hits, messages, strata
    pair = count_pair(node, rate, numerators, denominators, count)
    return (*pair, None) if pair else None


def leaves(value, path=()):
    """Yield (path, hits, messages, strata or None) for every per-message rate whose counts can be recovered."""
    if not isinstance(value, dict):
        return
    for key in RATE_KEYS:
        rate = value.get(key)
        if isinstance(rate, (int, float)) and not isinstance(rate, bool):
            numerators, denominators = SPECIAL.get(key, (NUMERATOR_KEYS, DENOMINATOR_KEYS))
            found = recover(value, rate, numerators, denominators, strata_of(value, key))
            if found:
                yield ((*path, key), *found)
    for key, count_key in RATE_MAP_KEYS.items():
        rates = value.get(key)
        if not isinstance(rates, dict):
            continue
        counts = value.get(count_key) if count_key else None
        counts = counts if isinstance(counts, dict) else {}
        for label, rate in rates.items():
            if isinstance(rate, (int, float)) and not isinstance(rate, bool):
                strata = strata_of(value, key, label)
                found = recover(value, rate, (), DENOMINATOR_KEYS, strata, counts.get(label))
                if found:
                    yield ((*path, key, str(label)), *found)
    for key, child in value.items():
        if isinstance(child, dict) and key not in RATE_MAP_KEYS and key not in ("by_length", "reliability"):
            yield from leaves(child, (*path, str(key)))


def collect(metrics):
    """{item: {path: (hits, messages, strata)}} from one corpus' metrics.json content."""
    result = {}
    for item, entry in metrics.items():
        found = {"/".join(path): (k, n, s) for path, k, n, s in leaves(entry.get("measurements", {}))}
        if found:
            result[item] = found
    return result


def chi_square(table):
    from scipy.stats import chi2

    hits, total = sum(k for k, _ in table), sum(n for _, n in table)
    rate = hits / total
    statistic = sum(
        (observed - expected) ** 2 / expected
        for k, n in table
        for observed, expected in ((k, n * rate), (n - k, n * (1 - rate)))
    )
    return statistic, len(table) - 1, float(chi2.sf(statistic, len(table) - 1))


def cmh(strata):
    """Generalized Cochran-Mantel-Haenszel test of equal hit rates across groups within length buckets.

    `strata` maps group to {bucket: (hits, messages)}. A bucket where every message hits, or none does, adds nothing.
    """
    import numpy as np
    from scipy.stats import chi2

    groups = list(strata)
    deviation = np.zeros(len(groups))
    variance = np.zeros((len(groups), len(groups)))
    # Sorted, so float sums do not follow per-process string hash order.
    for bucket in sorted({b for cells in strata.values() for b in cells}):
        hits = np.array([strata[g].get(bucket, (0, 0))[0] for g in groups], dtype=float)
        sizes = np.array([strata[g].get(bucket, (0, 0))[1] for g in groups], dtype=float)
        n, k = sizes.sum(), hits.sum()
        if n < 2 or k in (0, n):
            continue
        deviation += hits - sizes * k / n
        variance += k * (n - k) / (n**2 * (n - 1)) * (np.diag(sizes) * n - np.outer(sizes, sizes))
    dof = int(np.linalg.matrix_rank(variance))
    if not dof:
        return 0.0, 0, 1.0
    statistic = float(deviation @ np.linalg.pinv(variance) @ deviation)
    return statistic, dof, float(chi2.sf(statistic, dof))


def reference_rates(person, strata):
    """Per-bucket hit rate of the person corpus, or of all groups pooled where the person has no such message."""
    pooled = {}
    for cells in strata.values():
        for bucket, (k, n) in (cells or {}).items():
            cell = pooled.setdefault(bucket, [0, 0])
            cell[0] += k
            cell[1] += n
    own = person or {}
    return {
        bucket: own[bucket][0] / own[bucket][1] if own.get(bucket, (0, 0))[1] else k / n
        for bucket, (k, n) in pooled.items()
    }


def expected_hits(strata, reference):
    return sum(n * reference[bucket] for bucket, (_, n) in strata.items())


def benjamini_hochberg(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    q, running = [None] * len(p_values), 1.0
    for rank in range(len(order), 0, -1):
        i = order[rank - 1]
        running = min(running, p_values[i] * len(p_values) / rank)
        q[i] = running
    return q


def assess(
    values, strata=None, reference=None, min_expected=DEFAULT_MIN_EXPECTED, min_events=DEFAULT_MIN_EVENTS
):
    """Homogeneity test over groups, within length buckets when every tested group has them.

    The rarer outcome sets the effect ratio, so 0.95 vs 0.90 counts as a 2x difference.
    """
    observed = {name: pair for name, pair in values.items() if pair[1] > 0}
    hits, total = sum(k for k, _ in observed.values()), sum(n for _, n in observed.values())
    if not total:
        return {"testable_groups": 0, "reason": "no observed messages"}
    pooled = hits / total
    testable = {
        name: (k, n)
        for name, (k, n) in observed.items()
        if n * pooled >= min_expected and n * (1 - pooled) >= min_expected
    }
    rare_hits = sum(k for k, _ in testable.values())
    rare_misses = sum(n - k for k, n in testable.values())
    result = {"pooled_rate": pooled, "testable_groups": len(testable)}
    if len(testable) < 2 or min(rare_hits, rare_misses) < min_events:
        return dict(result, reason=f"fewer than 2 testable groups or {min_events} events/non-events")
    flip = pooled > 0.5

    def rare(k, n):
        return n - k if flip else k

    def spread(ratios):
        return {
            "rate_ratio": max(ratios.values()) / min(ratios.values()),
            "highest_group": max(ratios, key=ratios.get),
            "lowest_group": min(ratios, key=ratios.get),
        }

    # Jeffreys-smoothed rate of the rarer outcome.
    smoothed = {name: (rare(k, n) + 0.5) / (n + 1) for name, (k, n) in testable.items()}
    statistic, dof, p_value = chi_square(list(testable.values()))
    unadjusted = dict(spread(smoothed), chi2=statistic, dof=dof, p_value=p_value)
    result.update(rare_outcome="non_occurrence" if flip else "occurrence", groups_tested=sorted(testable))
    if reference is None or not all((strata or {}).get(name) for name in testable):
        return dict(result, test="chi_square", **unadjusted)
    # Observed over expected rarer outcomes, the expectation following each group's own length mix.
    adjusted = {}
    for name, (k, n) in testable.items():
        expected = expected_hits(strata[name], reference)
        adjusted[name] = (rare(k, n) + 0.5) / ((n - expected if flip else expected) + 0.5)
    statistic, dof, p_value = cmh({name: strata[name] for name in testable})
    return dict(
        result,
        test="length_stratified_cmh",
        **spread(adjusted),
        chi2=statistic,
        dof=dof,
        p_value=p_value,
        unadjusted=unadjusted,
    )


def classify(leaf, threshold=DEFAULT_RATIO_THRESHOLD, fdr=DEFAULT_FDR):
    if "p_value" not in leaf:
        return "insufficient"
    if leaf["rate_ratio"] < threshold:
        return "stable"
    return "relationship_dependent" if leaf["q_value"] < fdr else "uncertain"


def cell(hits, messages, strata=None):
    value = {"hits": hits, "messages": messages, "rate": hits / messages}
    if strata:
        value["by_length"] = {b: {"hits": k, "messages": n, "rate": k / n} for b, (k, n) in strata.items()}
    return value


def group_variation(corpus_metrics, grouping, *, threshold=DEFAULT_RATIO_THRESHOLD, fdr=DEFAULT_FDR):
    """Compare every recovered per-message rate across group corpora; the person corpus is reference only."""
    groups = [name for name in corpus_metrics if name != PRIMARY_CORPUS]
    collected = {name: collect(metrics) for name, metrics in corpus_metrics.items()}
    paths = sorted(
        {(item, path) for name in groups for item, found in collected[name].items() for path in found}
    )
    leaves_out = []
    for item, path in paths:
        found = {
            name: collected[name][item][path] for name in groups if path in collected[name].get(item, {})
        }
        person = collected.get(PRIMARY_CORPUS, {}).get(item, {}).get(path)
        defined = bool(LENGTH_DEFINED & set(path.split("/")))
        strata = None if defined else {name: s for name, (_, _, s) in found.items()}
        reference = reference_rates(person[2] if person else None, strata) if strata else None
        cells = {name: cell(k, n) for name, (k, n, _) in found.items()}
        for name, (k, _, s) in found.items():
            if reference and s:
                expected = expected_hits(s, reference)
                cells[name].update(expected_hits=expected, length_adjusted_ratio=(k + 0.5) / (expected + 0.5))
        leaf = {
            "item": item,
            "path": path,
            "person": cell(*person) if person else None,
            "groups": cells,
            **assess({name: (k, n) for name, (k, n, _) in found.items()}, strata, reference),
        }
        if leaf.get("test") == "chi_square":
            leaf["length_adjustment"] = (
                "rate is message length itself" if defined else "a tested group has no by_length counts"
            )
        leaves_out.append(leaf)
    tested = [leaf for leaf in leaves_out if "p_value" in leaf]
    for leaf, q in zip(tested, benjamini_hochberg([leaf["p_value"] for leaf in tested]), strict=True):
        leaf["q_value"] = q
    items = {}
    for leaf in leaves_out:
        leaf["class"] = classify(leaf, threshold, fdr)
        entry = items.setdefault(leaf["item"], {"leaf_classes": dict.fromkeys(CLASS_ORDER, 0), "leaves": {}})
        entry["leaf_classes"][leaf["class"]] += 1
        entry["leaves"][leaf.pop("path")] = {k: v for k, v in leaf.items() if k != "item"}
    for entry in items.values():
        entry["class"] = next(c for c in CLASS_ORDER if entry["leaf_classes"][c])
    tests = {}
    for leaf in tested:
        tests[leaf["test"]] = tests.get(leaf["test"], 0) + 1
    return {
        "method": {
            "unit": "per-message rates (one target turn); counts from by_length when they reproduce the rate, "
            "else recovered from the rate and its sibling denominator",
            "test": "generalized Cochran-Mantel-Haenszel over message length buckets (stats.LENGTH_BUCKETS, or "
            "stats.WINDOW_LENGTH_BUCKETS for 233 window labels) when every tested group has by_length counts, "
            "else chi-square homogeneity; groups enter with expected "
            f"occurrences and non-occurrences >= {DEFAULT_MIN_EXPECTED}; Benjamini-Hochberg across all tested rates",
            "rate_ratio": "max/min over tested groups of the rarer outcome: observed over expected given each group's "
            "length mix and the reference per-bucket rates when length-stratified, else the Jeffreys-smoothed rate; "
            "the unstratified ratio and test stay under unadjusted",
            "reference": "person corpus per-bucket rates, all groups pooled for a bucket without person messages; "
            "the person corpus is reported per leaf but not tested",
            "length_adjusted_ratio": "per group (hits + 0.5) / (expected hits + 0.5); a group's target rate in a "
            "length bucket is person.by_length[bucket].rate times this ratio, capped at 1",
            "length_defined": sorted(LENGTH_DEFINED),
            "classes": {
                "relationship_dependent": f"rare-outcome rate ratio >= {threshold} and q < {fdr}",
                "uncertain": f"ratio >= {threshold} but q >= {fdr}",
                "stable": f"ratio < {threshold}",
                "insufficient": f"fewer than 2 testable groups or fewer than {DEFAULT_MIN_EVENTS} events or non-events",
            },
            "item_class": "most group-dependent leaf class, in order " + " > ".join(CLASS_ORDER),
        },
        "groups": grouping.get("corpora", {}),
        "summary": {
            "items": {c: sorted(k for k, v in items.items() if v["class"] == c) for c in CLASS_ORDER},
            "leaves": {c: sum(v["leaf_classes"][c] for v in items.values()) for c in CLASS_ORDER},
            "tests": tests,
        },
        "items": items,
    }


def summary_counts(variation):
    return {c: len(v) for c, v in variation["summary"]["items"].items()} if variation else None
