import json
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import httpx
import pytest
from openai import BadRequestError, RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

from weclone.core.inference import (
    CodexExecClient,
    LLMAuditLogger,
    LLMRequest,
    OpenAICompatibleClient,
    RetryPolicy,
)
from weclone.core.inference.llm_client import make_request


class Score(BaseModel):
    score: int


def completion(text='{"score": 3}', finish="stop", **message_extra):
    return ChatCompletion(
        id="fixture",
        created=0,
        model="fixture-model",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": text, **message_extra},
            }
        ],
        usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    )


@pytest.fixture
def api(tmp_path, monkeypatch):
    client = OpenAICompatibleClient(
        api_key="fixture-key",
        base_url="http://example.invalid/v1",
        model="fixture-model",
        max_workers=3,
        retry_policy=RetryPolicy(max_retries=0),
        audit_logger=LLMAuditLogger(tmp_path / "audit", strict=True),
    )
    create = Mock(return_value=completion())
    monkeypatch.setattr(client.client.chat.completions, "create", create)
    yield client, create
    client.close()


def test_parameters_multimodal_and_explicit_none(api):
    client, create = api
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ],
        }
    ]
    result = client.chat(
        messages, max_tokens=None, temperature=0, timeout=20.0, extra_body={"thinking": {"type": "disabled"}}
    )
    params = create.call_args.kwargs
    assert result.ok and result.text == '{"score": 3}'
    assert "max_tokens" not in params
    assert params["messages"] == messages and params["temperature"] == 0
    assert params["timeout"] == 20.0 and params["extra_body"]["thinking"]["type"] == "disabled"
    assert LLMRequest.from_prompt("x").max_tokens == 1024
    assert make_request(LLMRequest.from_prompt("x"), max_tokens=None).max_tokens is None


def test_response_model_validation_and_audit(api, tmp_path):
    client, create = api
    result = client.chat("score", response_model=Score, metadata={"api_key": "do-not-log"})
    assert result.ok and result.parsed_model == Score(score=3)
    assert result.parsed_json == {"score": 3}
    assert create.call_args.kwargs["response_format"] == {"type": "json_object"}
    events = [
        json.loads(line)
        for path in (tmp_path / "audit").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert [e["event"] for e in events] == [
        "call.started",
        "attempt.started",
        "attempt.finished",
        "call.finished",
    ]
    assert events[0]["request"]["response_model"]["schema"]["required"] == ["score"]
    assert events[0]["request"]["metadata"]["api_key"] == "[REDACTED]"
    assert events[-1]["result"]["parsed_model"] == {"score": 3}


@pytest.mark.parametrize(
    "text,finish,error_type",
    [
        ('{"wrong": 1}', "stop", "ValidationError"),
        ("not JSON", "stop", "ValueError"),
        ('{"score": 3}', "length", "ValueError"),
        ('{"score": 3}', "content_filter", "ValueError"),
    ],
)
def test_invalid_structured_responses_retain_text(api, text, finish, error_type):
    client, create = api
    create.return_value = completion(text, finish)
    result = client.chat("score", response_model=Score)
    assert not result.ok and result.parsed_model is None
    assert result.text == text and result.finish_reason == finish
    assert result.metadata["error_type"] == error_type
    assert create.call_count == 1


def test_usage_and_reasoning(api):
    client, create = api
    create.return_value = completion("answer", reasoning_content="reason")
    result = client.chat("question")
    assert result.metadata["usage"]["total_tokens"] == 10
    assert result.metadata["reasoning"] == "reason"
    assert result.metadata["http_status"] == 200


def test_provider_error_and_retry_override(api):
    client, create = api
    error = RateLimitError(
        "limited",
        response=httpx.Response(429, request=httpx.Request("POST", "http://example.invalid")),
        body=None,
    )
    create.side_effect = [error, completion()]
    result = client.chat("score", retry_policy=RetryPolicy(max_retries=1, base_delay=0))
    assert result.ok and create.call_count == 2
    create.reset_mock(side_effect=True)
    create.side_effect = error
    result = client.chat("score")
    assert not result.ok and create.call_count == 1
    assert result.metadata == {"error_type": "RateLimitError", "http_status": 429}


def test_schema_fallback_and_last_attempt_empty(api, tmp_path):
    client, create = api
    error = BadRequestError(
        "response_format type is unavailable",
        response=httpx.Response(400, request=httpx.Request("POST", "http://example.invalid")),
        body=None,
    )
    create.side_effect = [error, completion("")]
    result = client.chat(
        "score", json_schema=Score.model_json_schema(), retry_policy=RetryPolicy(max_retries=1, base_delay=0)
    )
    assert not result.ok and result.error == "empty response"
    assert create.call_count == 2
    assert create.call_args_list[1].kwargs["response_format"] == {"type": "json_object"}
    events = [
        json.loads(line)
        for path in (tmp_path / "audit").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert events[-1]["event"] == "call.finished"
    assert events[-2]["will_retry"] is False


def test_empty_json_retry(api):
    client, create = api
    create.side_effect = [completion("  "), completion()]
    result = client.chat("score", json_mode=True, retry_policy=RetryPolicy(max_retries=1))
    assert result.ok and result.parsed_json == {"score": 3}
    assert create.call_count == 2


def test_stream_is_merged_and_closed(api):
    client, create = api
    chunks = [
        ChatCompletionChunk(
            id="fixture",
            created=0,
            model="fixture-model",
            object="chat.completion.chunk",
            choices=[
                {"index": 0, "delta": {"content": text, "reasoning_content": reason}, "finish_reason": finish}
            ],
        )
        for text, reason, finish in [('{"score":', "first", None), ("3}", "second", "stop")]
    ]
    chunks.append(
        ChatCompletionChunk(
            id="fixture",
            created=0,
            model="fixture-model",
            object="chat.completion.chunk",
            choices=[],
            usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        )
    )

    class Stream:
        closed = False

        def __iter__(self):
            return iter(chunks)

        def close(self):
            self.closed = True

    stream = Stream()
    create.return_value = stream
    result = client.chat("score", stream=True, response_model=Score)
    assert result.ok and result.parsed_model == Score(score=3)
    assert result.metadata["reasoning"] == "firstsecond"
    assert result.metadata["usage"]["total_tokens"] == 10
    assert stream.closed


def test_batch_order_failures_and_callback_once(api):
    client, create = api

    def respond(**params):
        value = int(params["messages"][0]["content"])
        time.sleep((3 - value) * 0.005)
        if value == 1:
            raise RuntimeError("fixture failure")
        return completion(str(value))

    create.side_effect = respond
    callbacks = []
    results = client.chat_batch(["0", "1", "2"], callback=lambda i, result: callbacks.append((i, result)))
    assert [r.text for r in results] == ["0", None, "2"]
    assert not results[1].ok and results[0].ok and results[2].ok
    assert sorted(i for i, _ in callbacks) == [0, 1, 2]
    assert all(result is results[i] for i, result in callbacks)
    assert client.chat_async("2").result(timeout=2).text == "2"


def test_stream_failure_preserves_partial_output(api):
    client, create = api

    class BrokenStream:
        closed = False

        def __iter__(self):
            yield ChatCompletionChunk(
                id="fixture",
                created=0,
                model="fixture-model",
                object="chat.completion.chunk",
                choices=[{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}],
            )
            raise TimeoutError("fixture interrupted stream")

        def close(self):
            self.closed = True

    stream = BrokenStream()
    create.return_value = stream
    result = client.chat("x", stream=True)
    assert not result.ok and result.text == "partial"
    assert result.metadata["error_type"] == "TimeoutError" and stream.closed


def test_close_interrupts_retry(api):
    client, create = api
    attempted = threading.Event()

    def fail(**kwargs):
        attempted.set()
        raise RuntimeError("retry me")

    create.side_effect = fail
    future = client.chat_async("x", retry_policy=RetryPolicy(max_retries=10, base_delay=30))
    assert attempted.wait(2)
    client.close(wait=False, cancel_futures=True)
    result = future.result(timeout=2)
    assert not result.ok and create.call_count == 1
    assert not client.chat("after close").ok


def test_cancel_queued_batch_does_not_hang(api):
    client, create = api
    release = threading.Event()
    started = threading.Event()

    def respond(**params):
        started.set()
        assert release.wait(3)
        return completion()

    create.side_effect = respond
    callbacks = []
    with ThreadPoolExecutor(max_workers=1) as runner:
        future = runner.submit(client.chat_batch, ["x"] * 10, callback=lambda i, r: callbacks.append(i))
        try:
            assert started.wait(2)
            client.close(wait=False, cancel_futures=True)
        finally:
            release.set()
        results = future.result(timeout=3)
    assert len(results) == 10 and sorted(callbacks) == list(range(10))
    assert any(not result.ok for result in results)


@pytest.fixture
def codex(tmp_path):
    script = tmp_path / "fake-codex"
    script.write_text(
        f"#!{sys.executable}\n"
        + """import json, sys, time, tomllib
from pathlib import Path
if sys.argv[1:3] == ["mcp", "list"]:
    print(json.dumps([{"name": "fixture.mcp"}, {"name": "future-server"}]))
    sys.exit(0)
config = dict(arg.split("=", 1) for i, arg in enumerate(sys.argv) if i and sys.argv[i - 1] == "-c")
instructions = Path(json.loads(config["model_instructions_file"])).read_text()
assert instructions == "You are a helpful assistant. Follow the user's instructions and answer concisely.\\n"
servers = tomllib.loads("servers=" + config["mcp_servers"])["servers"]
assert servers["fixture.mcp"]["enabled"] is False
assert servers["future-server"]["enabled"] is False
prompt = sys.stdin.read()
if prompt == "timeout": time.sleep(10)
if prompt == "fail": sys.stderr.write("fixture failed"); sys.exit(1)
if prompt == "capacity": sys.stderr.write("Insufficient model capacity"); sys.exit(1)
if prompt in {"capacity-event", "capacity-timeout"}:
    print(json.dumps({"type": "turn.failed", "error": {"code": "model_capacity_exceeded"}}), flush=True)
    if prompt == "capacity-timeout": time.sleep(10)
    sys.exit(1)
if prompt == "capacity-text":
    print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "rate limit 429"}}))
output = Path(sys.argv[sys.argv.index("--output-last-message") + 1])
output.write_text('{"score": 4}')
print(json.dumps({"type": "item.completed", "item": {"type": "web_search"}}))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 12, "cached_input_tokens": 5, "output_tokens": 4}}))
"""
    )
    script.chmod(0o700)
    client = CodexExecClient(
        model="fixture-model",
        command=str(script),
        timeout=2,
        enable_web_search=True,
        request_interval_seconds=0,
        audit_logger=LLMAuditLogger(tmp_path / "audit", strict=True),
    )
    client.requests_dir = tmp_path / "requests"
    client._disabled_mcp_args(tmp_path)
    yield client
    client.close()


def test_codex_success_schema_usage_search_and_cleanup(codex):
    result = codex.chat("score", response_model=Score, json_schema=Score.model_json_schema())
    assert result.ok and result.parsed_model == Score(score=4)
    assert result.metadata["usage"]["cached_input_tokens"] == 5
    assert result.metadata["web_search_calls"] == 1
    assert not list(codex.requests_dir.iterdir())


@pytest.mark.parametrize("web_search", [False, True])
def test_codex_minimal_context_defaults_and_explicit_web_search(codex, monkeypatch, web_search):
    codex.enable_web_search = web_search
    build = Mock(wraps=codex._build_command)
    monkeypatch.setattr(codex, "_build_command", build)
    result = codex.chat("score")
    assert result.ok
    kwargs = build.call_args.kwargs
    cmd = codex._build_command(build.call_args.args[0], **kwargs)
    config = dict(arg.split("=", 1) for i, arg in enumerate(cmd) if i and cmd[i - 1] == "-c")
    assert json.loads(config["web_search"]) == ("live" if web_search else "disabled")
    for key in (
        "features.memories", "skills.include_instructions", "features.shell_tool",
        "features.plugins", "features.apps", "features.image_generation", "agents.enabled",
        "include_environment_context", "include_collaboration_mode_instructions",
    ):
        assert config[key] == "false"
    assert config["project_doc_max_bytes"] == "0"
    assert json.loads(config["developer_instructions"]) == ""
    assert not kwargs["output_path"].parent.exists()
    assert "--ignore-user-config" not in cmd


def test_codex_mcp_discovery_runs_once_for_concurrent_requests(codex, monkeypatch):
    from weclone.core.inference import llm_client

    codex._mcp_overrides = None
    run = Mock(wraps=llm_client.subprocess.run)
    monkeypatch.setattr(llm_client.subprocess, "run", run)
    assert all(r.ok for r in codex.chat_batch(["a", "b", "c"]))
    assert run.call_count == 1
    assert run.call_args.args[0] == [codex.command, "mcp", "list", "--json"]


def test_codex_paces_concurrent_launches(codex, monkeypatch):
    from weclone.core.inference import llm_client

    codex.request_interval_seconds = 0.08
    launches = []
    popen = llm_client.subprocess.Popen

    def launch(*args, **kwargs):
        launches.append(time.monotonic())
        return popen(*args, **kwargs)

    monkeypatch.setattr(llm_client.subprocess, "Popen", launch)
    results = codex.chat_batch(["a", "b", "c", "d"])
    assert all(result.ok for result in results)
    assert len(launches) == 4
    assert all(b - a >= 0.065 for a, b in zip(launches, launches[1:]))


@pytest.mark.parametrize("prompt", ["capacity", "capacity-event", "capacity-timeout"])
def test_codex_capacity_delays_waiting_requests(codex, monkeypatch, prompt):
    from weclone.core.inference import llm_client

    codex.request_interval_seconds = 0.3
    codex.capacity_cooldown_seconds = 0.4
    launches, cooldowns = [], []
    popen, cool_down = llm_client.subprocess.Popen, codex._cool_down

    def launch(*args, **kwargs):
        launches.append(time.monotonic())
        return popen(*args, **kwargs)

    def cool():
        cooldowns.append(time.monotonic())
        cool_down()

    monkeypatch.setattr(llm_client.subprocess, "Popen", launch)
    monkeypatch.setattr(codex, "_cool_down", cool)
    first = codex.chat_async(prompt, timeout=0.15 if prompt == "capacity-timeout" else 2)
    deadline = time.monotonic() + 2
    while not launches and time.monotonic() < deadline:
        time.sleep(0.005)
    assert launches
    second = codex.chat_async("score")
    assert not first.result(timeout=2).ok
    assert second.result(timeout=2).ok
    assert len(cooldowns) == 1
    assert launches[1] - cooldowns[0] >= 0.385


def test_codex_does_not_throttle_on_normal_answer_text(codex, monkeypatch):
    cool_down = Mock()
    monkeypatch.setattr(codex, "_cool_down", cool_down)
    assert codex.chat("capacity-text").ok
    assert not codex.chat("fail").ok
    cool_down.assert_not_called()


def test_codex_can_cancel_during_launch_cooldown(codex, monkeypatch):
    from weclone.core.inference import llm_client

    popen = Mock()
    monkeypatch.setattr(llm_client.subprocess, "Popen", popen)
    codex._next_launch_at = time.monotonic() + 60
    future = codex.chat_async("score")
    deadline = time.monotonic() + 2
    while codex._active_calls == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert codex._active_calls == 1
    started = time.monotonic()
    codex.close()
    assert time.monotonic() - started < 1
    assert future.result().metadata["error_type"] == "CancelledError"
    popen.assert_not_called()


def test_codex_pacing_config_and_explicit_override(tmp_path):
    from weclone.core.inference import build_llm_client

    config = tmp_path / "settings.jsonc"
    config.write_text(json.dumps({"codex_exec_args": {
        "request_interval_seconds": 0.25, "capacity_cooldown_seconds": 7,
    }}))
    with build_llm_client("codex_exec", model="fixture", config_path=config) as client:
        assert (client.request_interval_seconds, client.capacity_cooldown_seconds) == (0.25, 7)
    with CodexExecClient(config_path=config, request_interval_seconds=0) as client:
        assert (client.request_interval_seconds, client.capacity_cooldown_seconds) == (0, 7)


def test_codex_batch_refills_submitted_work_before_slowest_finishes(codex, monkeypatch):
    started = [threading.Event() for _ in range(3)]
    release = threading.Event()

    def generate(request):
        index = int(request.messages[0]["content"])
        started[index].set()
        if index == 0:
            assert release.wait(timeout=2)
        return llm_response(ok=True, text=str(index))

    from weclone.core.inference import LLMResponse as llm_response

    codex._executor.shutdown()
    codex._executor = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(codex, "generate", generate)
    with ThreadPoolExecutor(max_workers=1) as runner:
        result = runner.submit(codex.chat_batch, ["0", "1", "2"])
        try:
            assert started[0].wait(timeout=1)
            assert started[2].wait(timeout=1)
        finally:
            release.set()
        assert [r.text for r in result.result(timeout=2)] == ["0", "1", "2"]


@pytest.mark.parametrize(
    "prompt,timeout,error", [("fail", 2, "CodexExecError"), ("timeout", 0.1, "TimeoutExpired")]
)
def test_codex_failure_and_timeout(codex, prompt, timeout, error):
    result = codex.chat(prompt, timeout=timeout)
    assert not result.ok and result.metadata["error_type"] == error
    assert not list(codex.requests_dir.iterdir())


def test_codex_rejects_multimodal_and_stream_before_launch(codex):
    result = codex.chat([{"role": "user", "content": [{"type": "image_url"}]}])
    assert not result.ok and "text messages only" in result.error
    assert not codex.requests_dir.exists()
    assert not codex.chat("x", stream=True).ok


def test_codex_cancellation(codex):
    future = codex.chat_async("timeout")
    deadline = time.monotonic() + 2
    while not codex.requests_dir.exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    codex.close(wait=False, cancel_futures=True)
    result = future.result(timeout=2)
    assert not result.ok and result.metadata["error_type"] == "CancelledError"


def test_isolated_codex_uses_directory_outside_project(codex, monkeypatch):
    from weclone.core.inference import build_llm_client

    with build_llm_client(
        "codex_exec",
        model="fixture-model",
        command=codex.command,
        isolated_cwd=True,
        audit_logger=LLMAuditLogger(enabled=False),
    ) as client:
        build = Mock(wraps=client._build_command)
        monkeypatch.setattr(client, "_build_command", build)
        result = client.chat("score", response_model=Score)
        assert result.ok
        cwd = build.call_args.kwargs["cwd"]
        assert not cwd.is_relative_to(Path.cwd())
        assert not cwd.exists()


def test_retry_status_filter_and_exception_filter():
    policy = RetryPolicy(retry_statuses=(429,), retry_exceptions=(TimeoutError,))
    assert policy.accepts(TimeoutError())
    assert not policy.accepts(ValueError())
    error = BadRequestError(
        "bad",
        response=httpx.Response(400, request=httpx.Request("POST", "http://example.invalid")),
        body=None,
    )
    assert not policy.accepts(error)
