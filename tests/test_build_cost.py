import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from weclone.core.inference import llm_client
from weclone.core.inference.llm_client import LLMResponse
from weclone.core.inference.model_prices import Price, in_windows, lookup
from weclone.core.inference.usage import Meter, UsageLimitReached, read_usage
from weclone.data.agent import build_cost, distill_profile
from weclone.data.agent import profile_pipeline as pipeline
from weclone.data.agent.llm_cache import LLMCache

DEEPSEEK = "https://api.deepseek.com"


def test_a_named_service_gets_its_own_price_and_any_other_the_median():
    official = lookup(DEEPSEEK, "deepseek-flash")
    assert (official.source, official.vendor) == ("official", "deepseek") and official.off_peak
    # A relay, or the model id a router gives it, gets what the providers listing it charge in the middle.
    relayed = lookup("https://relay.example/v1", "DeepSeek-AI/DeepSeek-Flash")
    assert relayed.source == "market" and relayed.vendor is None and relayed.off_peak is None
    assert lookup(DEEPSEEK, "no-such-model-anywhere") is None and lookup(DEEPSEEK, "") is None


def test_off_peak_windows_run_in_utc_on_iso_weekdays():
    windows = lookup(DEEPSEEK, "deepseek-flash").off_peak["windows"]
    # Monday 11:00 and 15:00 in Beijing are peak, 13:00 and the evening are not, nor is a Saturday.
    assert not in_windows(windows, datetime(2026, 11, 9, 3, tzinfo=timezone.utc))
    assert not in_windows(windows, datetime(2026, 11, 9, 7, tzinfo=timezone.utc))
    assert in_windows(windows, datetime(2026, 11, 9, 5, tzinfo=timezone.utc))
    assert in_windows(windows, datetime(2026, 11, 9, 23, 30, tzinfo=timezone.utc))
    assert in_windows(windows, datetime(2026, 11, 14, 3, tzinfo=timezone.utc))
    # A span past midnight belongs to the day it starts on: Friday night runs into Saturday, not Monday into Tuesday.
    friday_night = [{"hours_utc": "22:00-06:00", "weekdays": [5]}]
    assert in_windows(friday_night, datetime(2026, 11, 13, 23, tzinfo=timezone.utc))
    assert in_windows(friday_night, datetime(2026, 11, 14, 2, tzinfo=timezone.utc))
    assert not in_windows(friday_night, datetime(2026, 11, 13, 2, tzinfo=timezone.utc))


def test_a_request_is_priced_by_cache_hits_input_and_output():
    price = Price(1e-6, 4e-6, 1e-7, "official", "deepseek", "2026-11-08", {"input": 5e-7, "windows": []})
    assert price.cost(1000, 600, 100) == pytest.approx(600e-7 + 400e-6 + 400e-6)
    assert price.cost(1000, 600, 100, off_peak=True) == pytest.approx(600e-7 + 400 * 5e-7 + 400e-6)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
            (10, 4, 5, 3),
        ),
        ({"prompt_tokens": 10, "completion_tokens": 5, "prompt_cache_hit_tokens": 6}, (10, 6, 5, 0)),
        (
            {"input_tokens": 9, "cached_input_tokens": 2, "output_tokens": 4, "reasoning_output_tokens": 1},
            (9, 2, 4, 1),
        ),
        ({"prompt_tokens": 3, "completion_tokens": 1, "prompt_tokens_details": None}, (3, 0, 1, 0)),
        ({"total_tokens": 4}, None),
        (None, None),
    ],
)
def test_usage_is_read_from_each_service_shape(raw, expected):
    assert read_usage(raw) == (tuple(expected) if expected else None)


def test_the_meter_prices_each_attempt_at_the_time_it_ends():
    meter = Meter()
    usage = {"prompt_tokens": 1000, "completion_tokens": 100}
    meter.record(usage)
    assert meter.totals()["cost"] is None
    meter.price = lookup(DEEPSEEK, "deepseek-flash")
    meter.record(usage, at=datetime(2026, 11, 9, 3, tzinfo=timezone.utc))
    meter.record(usage, at=datetime(2026, 11, 14, 3, tzinfo=timezone.utc))
    meter.record(None)
    totals = meter.totals()
    peak = meter.price.cost(1000, 0, 100)
    assert totals["cost"] == pytest.approx(peak * 1.5)
    assert (totals["requests"], totals["unmetered"], totals["input_tokens"]) == (4, 1, 3000)


def chat(count, name="a"):
    """Samples of distinct text, all older than the chat's last two weeks but the last one."""
    return [
        {
            "id": str(index),
            "time": f"2026-01-{1 if index < count - 1 else 28:02d}T10:00:00",
            "messages": [{"role": "assistant", "content": f"{name}{index}" + "字" * 30}],
        }
        for index in range(count)
    ]


def settings(**extra):
    return SimpleNamespace(
        **{
            **vars(distill_profile.default_args()),
            "llm_provider": "api",
            "base_url": DEEPSEEK,
            "model_name": "deepseek-flash",
            "batch_size": 10,
            "launch_interval": 0.0,
            "config_path": None,
            **extra,
        }
    )


def test_requests_are_the_windows_of_samples_without_an_answer():
    args = settings()
    plan, pending = build_cost.count([chat(20), chat(3, "b")], args, "assistant")
    # Windows of at most eight samples, with each chat's last sample on its own as the only recent one: 19 + 1 and
    # 2 + 1 samples make 3 + 1 and 1 + 1 windows, for profile and event alike.
    assert plan["distill_profile"] == plan["distill_event"] == len(pending["state"]) == 6
    assert plan["later"] == round(12 * build_cost.LATER_REQUESTS) and plan["style"] == 0
    window = pending["state"][0]
    LLMCache().put_many(
        (distill_profile.sample_cache_key(sample, task="state", target_role="assistant"), "t", [], "m")
        for sample in window
    )
    plan, _ = build_cost.count([chat(20), chat(3, "b")], args, "assistant")
    assert plan["distill_profile"] == 5 and plan["distill_event"] == 6


def probe_windows(seconds=4.0, output=200, reasoning=0, metered=True):
    return [
        {
            "task": "state",
            "chars": 2000,
            "metered": metered,
            "input_tokens": 1000,
            "cached_tokens": 500,
            "output_tokens": output + index * 20,
            "reasoning_tokens": reasoning,
            "seconds": seconds + index * 0.2,
        }
        for index in range(10)
    ]


def test_the_estimate_shows_only_what_is_known():
    args = settings()
    plan = {"distill_profile": 100, "distill_event": 2, "later": 22, "later_chars": 48000, "style": 0}
    before = build_cost.describe(plan, None, args)
    assert before["requests"] == 124 and before["probe_requests"] == 5 + 2
    assert before["probe"] is None and before["cost"] is None and before["seconds"] is None
    probe = {"at": "2026-11-09T00:00:00+00:00", "windows": probe_windows(reasoning=100)}
    # DeepSeek's weekday peak runs 01:00–04:00 UTC, a Saturday is off-peak all day.
    after = build_cost.describe(plan, probe, args, start=datetime(2026, 11, 9, 1, tzinfo=timezone.utc))
    assert after["probe"]["requests"] == 10 and after["seconds"]["low"] < after["seconds"]["high"]
    cost = after["cost"]
    assert 0 < cost["low"] < cost["high"] and cost["per_request"]["low"] < cost["per_request"]["high"]
    assert (
        cost["source"] == "official"
        and cost["vendor"] == "deepseek"
        and cost["off_peak_share"] == 0
        and cost["off_peak_saving"] == pytest.approx(0.5, abs=0.05)
    )
    weekend = build_cost.describe(plan, probe, args, start=datetime(2026, 11, 14, 1, tzinfo=timezone.utc))[
        "cost"
    ]
    assert weekend["off_peak_share"] == 1 and weekend["high"] == pytest.approx(cost["high"] / 2, rel=0.05)
    # Two hours from an hour before the peak ends: half of it off-peak.
    price = build_cost.price_of(args)
    assert build_cost.off_peak_share(price, datetime(2026, 11, 9, 3, tzinfo=timezone.utc), 7200) == 0.5
    # A model the table does not have keeps its tokens and time; a service without usage keeps its time.
    unpriced = build_cost.describe(plan, probe, settings(model_name="house-model"))
    assert unpriced["cost"] is None and unpriced["input_tokens"] > 0 and unpriced["seconds"]
    silent = build_cost.describe(plan, {"at": "x", "windows": probe_windows(metered=False)}, args)
    assert silent["unmetered"] and silent["input_tokens"] is None and silent["seconds"]
    # Codex has tokens and no money, and starts a request a second however many may run.
    codex = build_cost.describe(
        plan, probe, settings(llm_provider="codex_exec", batch_size=40, launch_interval=1.0)
    )
    assert codex["cost"] is None and codex["input_tokens"] == after["input_tokens"]
    assert codex["seconds"]["low"] == codex["seconds"]["high"] == 124


class FakeClient:
    """Answers each window with an empty result per sample and DeepSeek-shaped usage."""

    def __init__(self, fail=False, answers=None):
        self.fail = fail
        # How many requests are answered before the usage limit, if there is one.
        self.answers = answers
        self.limit = None
        self.sent = []

    def generate_batch(self, requests):
        responses = []
        for request in requests:
            self.sent.append(request)
            if self.answers is not None and len(self.sent) > self.answers:
                self.limit = UsageLimitReached("You've hit your usage limit.")
                responses.append(LLMResponse(ok=False, error=str(self.limit)))
                continue
            if self.fail:
                responses.append(LLMResponse(ok=False, error="AuthenticationError: bad key"))
                continue
            task = request.metadata["task"].removesuffix("_distill")
            empty = [] if task == "state" else {"surface_events": []}
            results = [
                {"sample_id": sid, f"{task}_memories": empty} for sid in request.metadata["sample_ids"]
            ]
            responses.append(
                LLMResponse(
                    ok=True,
                    text=json.dumps({"results": results}),
                    parsed_json={"results": results},
                    model="deepseek-flash",
                    elapsed_s=3.0,
                    metadata={
                        "usage": {
                            "prompt_tokens": 1200,
                            "completion_tokens": 300,
                            "prompt_cache_hit_tokens": 800,
                        }
                    },
                )
            )
        return responses

    def close(self):
        pass


@pytest.fixture
def configured(storage, tmp_path, monkeypatch):
    config = storage(
        extra={
            "agent_distill_args": {"llm_provider": "api"},
            "make_dataset_args": {
                "platform": "chat",
                "base_url": DEEPSEEK,
                "llm_api_key": "k",
                "model_name": "deepseek-flash",
            },
        }
    )
    monkeypatch.setattr(build_cost, "chats_of_config", lambda config_path, folder: [chat(30), chat(12, "b")])
    return config


def test_a_probe_measures_real_windows_and_caches_their_answers(configured, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(llm_client, "build_llm_client", lambda *args, **kwargs: client)
    before = build_cost.estimate(configured, "assistant")
    assert before["probe"] is None
    probed = build_cost.probe(configured, "assistant")
    assert len(client.sent) == 2 * build_cost.PROBE_WINDOWS
    assert probed["probe"]["input_tokens"] == 1200 and probed["cost"]["source"] == "official"
    # The probed windows are answered now, and the estimate keeps the probe for these settings only.
    after = build_cost.estimate(configured, "assistant")
    assert after["requests"] == before["requests"] - 2 * build_cost.PROBE_WINDOWS
    assert after["probe"] == probed["probe"]
    stored = json.loads(build_cost.probes_path().read_text())
    assert len(stored) == 1 and "AuthenticationError" not in json.dumps(stored)
    configured.write_text(configured.read_text().replace("deepseek-flash", "deepseek-v4-pro"))
    assert build_cost.estimate(configured, "assistant")["probe"] is None


def test_a_probe_that_gets_no_answer_says_why(configured, monkeypatch):
    monkeypatch.setattr(llm_client, "build_llm_client", lambda *args, **kwargs: FakeClient(fail=True))
    with pytest.raises(RuntimeError, match="bad key"):
        build_cost.probe(configured, "assistant")
    assert not build_cost.probes_path().exists()


def test_a_probe_that_meets_the_usage_limit_keeps_the_answers_it_got(configured, monkeypatch):
    monkeypatch.setattr(llm_client, "build_llm_client", lambda *args, **kwargs: FakeClient(answers=3))
    before = build_cost.estimate(configured, "assistant")["requests"]
    with pytest.raises(UsageLimitReached, match="usage limit"):
        build_cost.probe(configured, "assistant")
    assert build_cost.estimate(configured, "assistant")["requests"] == before - 3


def test_a_stage_that_met_the_usage_limit_ends_the_build(tmp_path, monkeypatch, storage):
    storage()
    meter = Meter()
    monkeypatch.setattr(pipeline, "METER", meter)
    meter.limit = "left over from an earlier build"
    build = pipeline.Build(tmp_path, "clone")
    assert meter.limit is None
    # A stage that goes on without the answers, as the summaries do, still ends the build.
    with pytest.raises(UsageLimitReached, match="3:05 PM"), build.running():
        with build.stage("snapshot"):
            pass
        with build.stage("distill_profile"):
            meter.limit = "You've hit your usage limit. Try again at 3:05 PM."
        with build.stage("distill_event"):
            pass
    status = pipeline.read_status(tmp_path)
    assert status["state"] == "failed" and status["error"]["key"] == "distill_profile"
    assert status["error"]["type"] == "UsageLimitReached" and "3:05 PM" in status["error"]["message"]
    assert status["stages"][2]["state"] == "pending"
    # A stage that fails over the missing answers reports the limit, not its own error.
    meter.limit = None
    build = pipeline.Build(tmp_path, "clone")
    with pytest.raises(UsageLimitReached) as raised, build.running(), build.stage("organize"):
        meter.limit = "You've hit your usage limit."
        raise RuntimeError("3 LLM tasks failed")
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_the_status_keeps_what_each_stage_used(tmp_path, monkeypatch, storage):
    storage()
    meter = Meter()
    monkeypatch.setattr(pipeline, "METER", meter)
    meter.price = lookup(DEEPSEEK, "deepseek-flash")
    meter.record({"prompt_tokens": 1, "completion_tokens": 1})  # before the build
    build = pipeline.Build(tmp_path, "assistant")
    with build.stage("distill_profile"):
        build.progress(0, 4)
        meter.record({"prompt_tokens": 1000, "completion_tokens": 100})
        meter.record(None)
        build.progress(4, 4)
    status = pipeline.read_status(tmp_path)
    stage = status["stages"][1]
    assert (stage["done"], stage["total"], stage["unit"]) == (4, 4, "requests")
    assert stage["usage"]["requests"] == status["usage"]["requests"] == 2
    assert stage["usage"]["unmetered"] == 1 and stage["usage"]["cost"] > 0
    assert status["stages"][0]["usage"] is None
