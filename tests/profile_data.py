"""Builders shared by the tests that serve a hand-made profile."""


def fact(dim, attr, value, sensitivity, sids, *, scope="current", support=1):
    return {
        "dim": dim,
        "attr": attr,
        "value": value,
        "sensitivity": sensitivity,
        "time_scope": scope,
        "support_count": support,
        "source_ids": sids,
    }
