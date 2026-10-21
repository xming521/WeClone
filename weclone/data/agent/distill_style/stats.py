"""Shared statistics helpers for style chapters."""

import functools

DEFAULT_TOP_K = 100
DEFAULT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
# Upper bounds of the message length buckets in nonspace characters; the last bucket is open.
LENGTH_EDGES = (2, 4, 8, 16, 32)
# Window labels (233) depend on how much text the whole window has, so they use wider window length buckets.
WINDOW_LENGTH_EDGES = (8, 16, 32, 64, 128)


def bucket_labels(edges):
    return ("0", *(f"{low + 1}-{high}" for low, high in zip((0, *edges), edges)), f"{edges[-1] + 1}+")


LENGTH_BUCKETS = bucket_labels(LENGTH_EDGES)
WINDOW_LENGTH_BUCKETS = bucket_labels(WINDOW_LENGTH_EDGES)
LENGTH_STRATA = {
    "length": "nonspace characters of one message (Unicode code points; platform codes such as [捂脸] as written)",
    "buckets": LENGTH_BUCKETS,
    "by_length": "hits, messages and rate per bucket of each per-message rate's own denominator; bucket 0 holds blank messages",
    "window_length": "233 window labels only: nonspace characters of all target messages of the window together; "
    "each nonempty message of a labeled window counts once in its window's bucket",
    "window_buckets": WINDOW_LENGTH_BUCKETS,
}
# A rate is estimable once its rarer outcome has this many effective events.
MIN_EVENTS = 30
# Below MIN_EVENTS, a rarer outcome whose upper bound stays under this share of messages is still informative.
RARE_UPPER = 0.01
CONFIDENCE = 0.95
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 0
RELIABILITY = {
    "unit": "each per-message rate with by_length counts, plus 218 message_long_rate and message_length_shares",
    "rare_outcome": "occurrence unless the rate exceeds 0.5, then non_occurrence; events count the rarer outcome",
    "interval": f"{CONFIDENCE:.0%} interval of the rate: the wider end of a window (session) cluster bootstrap "
    f"({BOOTSTRAP_SAMPLES} resamples, seed {BOOTSTRAP_SEED}, percentile) and the exact Clopper-Pearson interval "
    "over messages",
    "effective_events": "events divided by the design effect, i.e. bootstrap variance over binomial variance, "
    "floored at 1; needs at least 2 windows",
    "rare_upper": "upper end of the interval for the rarer outcome's rate",
    "bounds": "LLM-annotated rates only: unknown_messages and failure_messages are target messages left out of "
    "the denominator because their window or message was judged unknown or its annotation failed; "
    "bounds = [hits, hits + missing] / (messages + missing), i.e. every missing message counted absent or present",
    "classes": {
        "estimable": f"effective_events >= {MIN_EVENTS}: the rate and the distributions inside the item can be used",
        "rare": f"otherwise rare_upper <= {RARE_UPPER}: the rarer outcome almost never occurs; use only as a bound",
        "undetermined": "otherwise: too little evidence either way; describe only",
    },
}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def quantiles(values, points=DEFAULT_QUANTILES):
    """Descriptive distribution with linearly interpolated quantiles."""
    values = sorted(values)
    if not values:
        return {"n": 0, "mean": None, "quantiles": {}}

    def at(p):
        position = p * (len(values) - 1)
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)

    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "quantiles": {f"p{round(100 * p)}": at(p) for p in points},
    }


def nonspace_length(text):
    return sum(not c.isspace() for c in text)


def length_bucket(length, edges=LENGTH_EDGES):
    return bucket_labels(edges)[0 if not length else 1 + sum(length > e for e in edges)]


def length_strata(pairs, edges=LENGTH_EDGES):
    """Hits and messages per length bucket from the (nonspace length, hit) pairs of one rate's denominator."""
    cells = {}
    for length, hit in pairs:
        cell = cells.setdefault(length_bucket(length, edges), [0, 0])
        cell[0] += bool(hit)
        cell[1] += 1
    return {
        label: {
            "hits": cells[label][0],
            "messages": cells[label][1],
            "rate": cells[label][0] / cells[label][1],
        }
        for label in bucket_labels(edges)
        if label in cells
    }


def by_length(texts, hits):
    """Length strata of a rate whose denominator maps message keys to texts and whose hits are keys."""
    return length_strata((nonspace_length(text), key in hits) for key, text in texts.items())


def message_texts(rows):
    """{(sample_id, message_id): text} of the nonempty target messages in `rows`."""
    return {
        (row["sample_id"], m["message_id"]): m["text"]
        for row in rows
        for m in row["messages"]
        if m["text"].strip()
    }


@functools.lru_cache(maxsize=4)
def resample_weights(windows):
    """Times each of `windows` clusters is drawn in each bootstrap resample.

    Integer matrices keep the product out of multithreaded BLAS, which would spin every core for a 5 ms job.
    """
    import numpy as np

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, windows, size=(BOOTSTRAP_SAMPLES, windows))
    draws += np.arange(BOOTSTRAP_SAMPLES)[:, None] * windows
    counts = np.bincount(draws.ravel(), minlength=BOOTSTRAP_SAMPLES * windows)
    return counts.reshape(BOOTSTRAP_SAMPLES, windows).astype(np.int32)


def window_reliability(pairs):
    """Counts, interval and evidence class of one per-message rate from the (window, hit) pairs of its denominator."""
    import numpy as np
    from scipy.stats import beta

    cells = {}
    for window, hit in pairs:
        cell = cells.setdefault(window, [0, 0])
        cell[0] += bool(hit)
        cell[1] += 1
    hits, messages = sum(k for k, _ in cells.values()), sum(n for _, n in cells.values())
    result = {"hits": hits, "messages": messages, "windows": len(cells)}
    if not messages:
        return dict(result, **{"class": "undetermined"})
    rate = hits / messages
    flip = rate > 0.5
    events = messages - hits if flip else hits
    alpha = 1 - CONFIDENCE
    low = float(beta.ppf(alpha / 2, hits, messages - hits + 1)) if hits else 0.0
    high = float(beta.ppf(1 - alpha / 2, hits + 1, messages - hits)) if hits < messages else 1.0
    effective = None
    if events and len(cells) > 1:
        sampled = resample_weights(len(cells)) @ np.array(list(cells.values()), dtype=np.int32)
        rates = sampled[:, 0].astype(np.float64) / sampled[:, 1]
        bootstrap_low, bootstrap_high = np.quantile(rates, [alpha / 2, 1 - alpha / 2])
        low, high = min(low, float(bootstrap_low)), max(high, float(bootstrap_high))
        design = max(1.0, float(rates.var()) / (rate * (1 - rate) / messages))
        effective = events / design
    elif not events:
        effective = 0.0
    rare_upper = 1 - low if flip else high
    if effective is not None and effective >= MIN_EVENTS:
        label = "estimable"
    elif rare_upper <= RARE_UPPER:
        label = "rare"
    else:
        label = "undetermined"
    return dict(
        result,
        rare_outcome="non_occurrence" if flip else "occurrence",
        events=events,
        effective_events=effective,
        interval=[low, high],
        rare_upper=rare_upper,
        **{"class": label},
    )


def reliability(texts, hits):
    """`window_reliability` of a rate whose denominator maps (sample_id, message_id) keys to texts."""
    return window_reliability((key[0], key in hits) for key in texts)


def missing_messages(pairs, blank=False):
    """(unknown, failure) target messages of the windows in (row, state) pairs that are unknown or failed.

    Blank messages count only with `blank`, for rates whose denominator keeps them.
    """
    counts = {"unknown": 0, "failure": 0}
    for row, state in pairs:
        if state in counts:
            counts[state] += sum(blank or bool(m["text"].strip()) for m in row["messages"])
    return counts["unknown"], counts["failure"]


def with_missing(record, unknown, failure):
    """Add to a reliability record the bounds of its rate if every unknown or failed message hit, or none did."""
    total = record["messages"] + unknown + failure
    record.update(
        unknown_messages=unknown,
        failure_messages=failure,
        missing_share=ratio(unknown + failure, total),
        bounds=[record["hits"] / total, (record["hits"] + unknown + failure) / total] if total else None,
    )
    return record


def missing_entries(entries):
    """(unknown, failure) nonblank messages among per-message annotation entries with `nonspace_characters`."""
    counts = {"unknown": 0, "failure": 0}
    for entry in entries:
        if entry["state"] in counts and entry["nonspace_characters"]:
            counts[entry["state"]] += 1
    return counts["unknown"], counts["failure"]
