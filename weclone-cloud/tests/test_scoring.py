from __future__ import annotations

from dataclasses import dataclass

import pytest
from weclone_cloud.api import Completion
from weclone_cloud.errors import ApiError
from weclone_cloud.scoring import (
    ScoreOutcome,
    apply_outcomes,
    build_report,
    filter_records,
    score_records,
)

from .conftest import record


class FakeClient:
    def __init__(self, score: int = 5, *, fail: bool = False) -> None:
        self.score = score
        self.fail = fail
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, json_schema=None, use_cache=True, **_kwargs) -> Completion:
        self.calls.append(messages)
        if self.fail:
            raise ApiError(429, "rate limited")
        data = {"score": self.score, "reason": "有说话风格"}
        return Completion(text=str(data), data=data, usage={"total_tokens": 12}, model="m")


def test_scores_and_writes_back(records):
    client = FakeClient(score=5)
    outcomes = score_records(records, client, max_workers=2)

    assert [o.score for o in outcomes] == [5, 5, 5]
    assert len(client.calls) == 3

    applied, failed = apply_outcomes(records, outcomes)
    assert (applied, failed) == (3, 0)
    assert all(r["score"] == 5 and r["score_source"] == "cloud" for r in records)
    assert records[0]["original_score"] == 4


def test_skips_already_scored_unless_rescore(records):
    for row in records:
        row["score_source"] = "cloud"
        row["score"] = 4
    client = FakeClient()
    outcomes = score_records(records, client)
    assert client.calls == []
    assert [o.score for o in outcomes] == [4, 4, 4]

    score_records(records, client, rescore=True)
    assert len(client.calls) == 3


def test_failures_are_recorded_not_raised(records):
    client = FakeClient(fail=True)
    outcomes = score_records(records, client, max_workers=2)
    assert all(not o.ok for o in outcomes)
    assert "rate limited" in outcomes[0].error

    applied, failed = apply_outcomes(records, outcomes)
    assert (applied, failed) == (0, 3)
    assert "score_source" not in records[0]


def test_filter_and_report(records):
    for row, score in zip(records, [5, 2, 4]):
        row["score"] = score
    kept = filter_records(records, min_score=4)
    assert len(kept) == 2

    outcomes = [ScoreOutcome(index=i, record_id=str(i), score=records[i]["score"]) for i in range(3)]
    report = build_report(records, outcomes, min_score=4, model="m")
    assert report["kept"] == 2
    assert report["distribution"]["5"] == 1
    assert report["mean_score"] == pytest.approx(11 / 3, rel=1e-3)


def test_bad_model_output_counts_as_failure(records):
    @dataclass
    class BadClient:
        def chat(self, *args, **kwargs):
            return Completion(text="我觉得还行", data=None)

    outcomes = score_records(records, BadClient())  # type: ignore[arg-type]
    assert all(not o.ok for o in outcomes)
    assert "无法解析评分" in outcomes[0].error
