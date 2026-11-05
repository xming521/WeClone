"""Reproducible contact-by-month sampling of chat windows for semantic labels."""

from collections import defaultdict

from .data import digest, record_time

DEFAULT_LLM_SAMPLES = 2000
DEFAULT_SAMPLE_SEED = 42
DEFAULT_GROUP_FLOOR = 100


def ranked(rows, seed):
    return sorted(rows, key=lambda r: (digest([seed, r["sample_id"]]), r["sample_id"]))


def stratum(row):
    moment = record_time(row)
    return row.get("contact_id") or "unknown", moment.strftime("%Y-%m") if moment else "unknown"


def proportional(strata, count, total):
    quotas = {key: count * len(rows) // total for key, rows in strata.items()}
    order = sorted(strata, key=lambda key: (-(count * len(strata[key]) % total), key))
    for key in order[: count - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def sample_records(
    records, count=DEFAULT_LLM_SAMPLES, seed=DEFAULT_SAMPLE_SEED, groups=None, floor=DEFAULT_GROUP_FLOOR
):
    """Proportional person sample plus within-group top-ups, together within `count` windows.

    `groups` maps sample_id to its group corpus. The person corpus keeps only the proportional
    windows, so it stays self-weighting; top-ups serve group corpora only.
    """
    groups = groups or {}
    total, budget = len(records), min(count, len(records))
    strata = defaultdict(list)
    for row in records:
        strata[stratum(row)].append(row)
    strata = {key: ranked(rows, seed) for key, rows in strata.items()}
    members = defaultdict(list)
    for row in records:
        if row["sample_id"] in groups:
            members[groups[row["sample_id"]]].append(row)
    members = {name: ranked(rows, seed) for name, rows in sorted(members.items())}
    # Floors may use at most half of the budget, so the person sample keeps the other half.
    floor = min(floor, budget // (2 * len(members))) if members else 0

    def draw(pooled_count):
        quotas = proportional(strata, pooled_count, total)
        pooled = {r["sample_id"] for key, rows in strata.items() for r in rows[: quotas[key]]}
        topups = {}
        for name, rows in members.items():
            need = max(0, min(floor, len(rows)) - sum(r["sample_id"] in pooled for r in rows))
            topups[name] = [r["sample_id"] for r in rows if r["sample_id"] not in pooled][:need]
        return quotas, pooled, topups

    def size(state):
        return len(state[1]) + sum(map(len, state[2].values()))

    pooled_count = budget
    state = draw(pooled_count)
    while pooled_count and size(state) > budget:
        pooled_count = max(0, pooled_count - (size(state) - budget))
        state = draw(pooled_count)
    while pooled_count < budget:
        trial = draw(pooled_count + 1)
        if size(trial) > budget:
            break
        pooled_count, state = pooled_count + 1, trial
    quotas, pooled, topups = state
    extra = {name: set(ids) for name, ids in topups.items()}
    selected = pooled.union(*extra.values())
    sampled = [row for row in records if row["sample_id"] in selected]
    group_ids = {
        name: [
            r["sample_id"]
            for r in records
            if groups.get(r["sample_id"]) == name
            and (r["sample_id"] in pooled or r["sample_id"] in extra[name])
        ]
        for name in members
    }
    return sampled, {
        "method": "proportional_contact_month_stratified_hash_sample_with_group_floor",
        "seed": seed,
        "population": total,
        "budget": budget,
        "selected": len(sampled),
        "pooled_selected": len(pooled),
        "group_floor": floor,
        "population_digest": digest(records),
        "sample_ids": [r["sample_id"] for r in sampled],
        "pooled_ids": [r["sample_id"] for r in records if r["sample_id"] in pooled],
        "group_ids": group_ids,
        "groups": {
            name: {
                "population": len(rows),
                "pooled": sum(r["sample_id"] in pooled for r in rows),
                "top_up": len(extra[name]),
                "selected": len(group_ids[name]),
                "inclusion_probability": len(group_ids[name]) / len(rows),
            }
            for name, rows in members.items()
        },
        "strata": {
            f"{contact}|{month}": {
                "population": len(strata[(contact, month)]),
                "selected": quotas[(contact, month)],
                "inclusion_probability": quotas[(contact, month)] / len(strata[(contact, month)]),
            }
            for contact, month in sorted(strata)
        },
        "estimator": "person corpus uses only the proportional windows, unweighted; each group corpus adds hash-ranked top-ups inside the group and is unweighted within the group",
    }
