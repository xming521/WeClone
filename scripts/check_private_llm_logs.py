"""Synthetic checks for private LLM logging; no model calls or user chats."""

import io
import json
import logging
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from openai.types.chat import ChatCompletion

from weclone.core.inference import CodexExecClient, LLMAuditLogger, OpenAICompatibleClient, RetryPolicy
from weclone.data.agent import profile_pipeline
from weclone.utils import secure_storage as secure
from weclone.utils.log import InterceptHandler


class PrivateLogChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="weclone-private-log-check-")
        self.root = Path(self.directory.name)
        self.original_config = os.environ.get("WECLONE_CONFIG_PATH")
        config = self.root / "settings.jsonc"
        config.write_text(
            json.dumps(
                {
                    "security_args": {
                        "storage_mode": "encrypted",
                        "state_dir": str(self.root / "state"),
                    }
                }
            )
        )
        secure.configure(config)
        secure.lock()
        secure.ensure_initialized("1")
        self.marker = "synthetic-private-chat-never-in-logs"

    def tearDown(self):
        secure.lock()
        if self.original_config is None:
            os.environ.pop("WECLONE_CONFIG_PATH", None)
        else:
            os.environ["WECLONE_CONFIG_PATH"] = self.original_config
        self.directory.cleanup()

    def test_api_audit_keeps_usage_and_status_without_request_or_reply(self):
        audit = self.root / "audit"
        response = ChatCompletion(
            id="synthetic-request",
            created=0,
            model="synthetic-model",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": self.marker,
                    },
                }
            ],
            usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        )
        with OpenAICompatibleClient(
            api_key="synthetic-key",
            base_url="http://example.invalid/v1",
            model="synthetic-model",
            max_workers=1,
            retry_policy=RetryPolicy(max_retries=0),
            audit_logger=LLMAuditLogger(audit, strict=True),
        ) as client:
            with patch.object(client.client.chat.completions, "create", return_value=response):
                result = client.chat(self.marker)
        self.assertTrue(result.ok)
        self.assertEqual(result.text, self.marker)
        raw = b"".join(path.read_bytes() for path in audit.glob("*.jsonl"))
        self.assertNotIn(self.marker.encode(), raw)
        self.assertNotIn(b"synthetic-key", raw)
        self.assertIn(b"call.finished", raw)
        self.assertIn(b'"total_tokens": 10', raw)

    def test_library_terminal_logs_preserve_original_messages_in_encrypted_mode(self):
        handler = InterceptHandler(level=logging.DEBUG)
        output = io.StringIO()
        with redirect_stderr(output):
            for name in ("openai.synthetic", "presidio-analyzer"):
                for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR):
                    handler.handle(logging.LogRecord(name, level, "", 0, self.marker, (), None))
        self.assertEqual(output.getvalue().count(self.marker), 8)
        self.assertNotIn("details omitted", output.getvalue())
        self.assertEqual(output.getvalue().count("presidio-analyzer"), 4)

    def test_plaintext_mode_preserves_request_reply_and_error_details(self):
        config = self.root / "plaintext.jsonc"
        config.write_text(
            json.dumps(
                {
                    "security_args": {
                        "storage_mode": "plaintext",
                        "state_dir": str(self.root / "plaintext-state"),
                    }
                }
            )
        )
        secure.lock()
        secure.configure(config)
        response = ChatCompletion(
            id="synthetic-request",
            created=0,
            model="synthetic-model",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": self.marker,
                    },
                }
            ],
        )
        audit = self.root / "audit"
        with OpenAICompatibleClient(
            api_key="synthetic-key",
            base_url="http://example.invalid/v1",
            model="synthetic-model",
            max_workers=1,
            retry_policy=RetryPolicy(max_retries=0),
            audit_logger=LLMAuditLogger(audit, strict=True),
        ) as client:
            with patch.object(client.client.chat.completions, "create", return_value=response):
                result = client.chat(self.marker, metadata={"api_key": "synthetic-key"})
        self.assertTrue(result.ok)
        events = [
            json.loads(line) for path in audit.glob("*.jsonl") for line in path.read_text().splitlines()
        ]
        self.assertEqual(events[0]["request"]["messages"][0]["content"], self.marker)
        self.assertEqual(events[0]["request"]["metadata"]["api_key"], "[REDACTED]")
        self.assertEqual(events[-1]["result"]["text"], self.marker)
        call = LLMAuditLogger(audit, strict=True).start_call(
            request=self.marker,
            provider="synthetic",
            model="synthetic-model",
        )
        call.finish_exception(ValueError(self.marker))
        events = [
            json.loads(line) for path in audit.glob("*.jsonl") for line in path.read_text().splitlines()
        ]
        self.assertEqual(events[-1]["result"]["error"]["message"], self.marker)
        output = io.StringIO()
        with redirect_stderr(output):
            InterceptHandler().handle(
                logging.LogRecord(
                    "openai.synthetic",
                    logging.INFO,
                    "",
                    0,
                    self.marker,
                    (),
                    None,
                )
            )
        self.assertIn(self.marker, output.getvalue())

    def test_codex_logs_and_state_are_removed_after_success_and_failure(self):
        source_home = self.root / "source-home"
        source_home.mkdir()
        outside_logs, outside_state = self.root / "global-log", self.root / "global-state"
        (source_home / "config.toml").write_text(
            f"log_dir = {json.dumps(str(outside_logs))}\nsqlite_home = {json.dumps(str(outside_state))}\n"
        )
        runtime_paths = []
        marker = self.marker

        class FakeProcess:
            def __init__(self, command, **kwargs):
                self.command, self.env = command, kwargs["env"]
                self.returncode = 0

            def communicate(self, input=None, timeout=None):
                overrides = {
                    value.split("=", 1)[0]: json.loads(value.split("=", 1)[1])
                    for index, value in enumerate(self.command)
                    if index and self.command[index - 1] == "-c"
                }
                output = Path(self.command[self.command.index("--output-last-message") + 1])
                runtime_paths.append(output.parent)
                for field in ("log_dir", "sqlite_home"):
                    path = Path(overrides[field])
                    path.mkdir(parents=True)
                    (path / "synthetic-trace").write_text(marker)
                assert overrides["history.persistence"] == "none"
                assert overrides["otel.log_user_prompt"] is False
                assert overrides["otel.exporter"] == "none"
                assert overrides["otel.trace_exporter"] == "none"
                assert all(self.env[key] == str(output.parent) for key in ("TMPDIR", "TMP", "TEMP"))
                assert Path(self.env["CODEX_HOME"]).parent == output.parent
                output.write_text(marker)
                if input == "fail":
                    self.returncode = 1
                return json.dumps({"type": "turn.completed", "usage": {"input_tokens": 3}}), marker

        with patch.dict(os.environ, {"CODEX_HOME": str(source_home)}):
            for prompt in (self.marker, "fail"):
                with CodexExecClient(
                    model="synthetic-model",
                    max_workers=1,
                    request_interval_seconds=0,
                    audit_logger=LLMAuditLogger(self.root / "audit", strict=True),
                ) as client:
                    with (
                        patch.object(client, "_disabled_mcp_args", return_value=[]),
                        patch("weclone.core.inference.llm_client.subprocess.Popen", FakeProcess),
                    ):
                        result = client.chat(prompt)
                self.assertEqual(result.ok, prompt != "fail")
                self.assertNotIn(self.marker, result.error or "")
        self.assertFalse(outside_logs.exists())
        self.assertFalse(outside_state.exists())
        self.assertTrue(runtime_paths)
        self.assertTrue(all(not path.exists() for path in runtime_paths))
        for path in (self.root / "audit").glob("*.jsonl"):
            self.assertNotIn(self.marker.encode(), path.read_bytes())

    def test_stage_logs_show_failure_and_duration_without_exception_body(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(ValueError):
            with profile_pipeline._stage("启动：读取模型上下文"):
                raise ValueError(self.marker)
        text = output.getvalue()
        self.assertIn("启动：读取模型上下文：开始", text)
        self.assertIn("失败（ValueError）", text)
        self.assertIn("耗时", text)
        self.assertNotIn(self.marker, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
