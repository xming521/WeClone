import json
from pathlib import Path
from unittest.mock import Mock

import click
import pytest
from click.testing import CliRunner

from weclone.core.inference import llm_client
from weclone.data.agent import distill_event, distill_profile


def test_template_preserves_codex_distill_settings():
    config = distill_profile.load_codex_exec_config(Path("settings.template.jsonc"))
    required = {
        "model", "effort", "batch_size", "timeout", "command", "sandbox",
        "request_interval_seconds", "capacity_cooldown_seconds",
    }
    assert required <= config.keys()
    assert all(config[key] not in (None, "") for key in required)
    assert (config["request_interval_seconds"], config["capacity_cooldown_seconds"]) == (0.5, 20.0)
    assert "max_tokens" not in config
    args = distill_profile.default_args()
    args.config_path = Path("settings.template.jsonc")
    resolved = distill_profile.resolve_llm_args(args)
    assert (resolved.max_tokens, resolved.timeout) == (None, 300)


def test_api_provider_does_not_require_codex_exec_settings(tmp_path):
    config_path = tmp_path / "settings.jsonc"
    config_path.write_text(
        json.dumps(
            {
                "agent_distill_args": {
                    "llm_provider": "api",
                    "batch_size": 3,
                    "max_tokens": 2048,
                    "timeout": 45,
                }
            }
        ),
        encoding="utf-8",
    )
    args = distill_profile.default_args()
    args.config_path = config_path

    resolved = distill_profile.resolve_llm_args(args)

    assert resolved.llm_provider == "api"
    assert (resolved.batch_size, resolved.max_tokens, resolved.timeout) == (3, 2048, 45)
    assert resolved.model is None
    assert resolved.codex_command is None


@pytest.mark.parametrize("module", [distill_profile, distill_event])
@pytest.mark.parametrize("answer,approved", [("", False), ("\n", False), ("n\n", False), ("y\n", True)])
def test_distillation_requires_consent_before_writes_or_model_calls(
    tmp_path, monkeypatch, module, answer, approved,
):
    config_path = tmp_path / "settings.jsonc"
    config_path.write_text(json.dumps({
        "codex_exec_args": {
            "model": "fixture", "effort": "medium", "batch_size": 1,
            "timeout": 30, "command": "codex", "sandbox": "read-only",
        },
        "agent_distill_args": {"llm_provider": "codex_exec"},
    }), encoding="utf-8")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "chat.json").write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "output"

    client = Mock()
    build_client = Mock(return_value=client)
    process_file = Mock(return_value=distill_profile.FileStats())
    monkeypatch.setattr(llm_client, "build_llm_client", build_client)
    monkeypatch.setattr(distill_profile, "process_task_file", process_file)

    @click.command()
    def distill():
        module.main(input_dir=input_dir, output_dir=output_dir, config_path=config_path)

    result = CliRunner().invoke(distill, input=answer)

    assert "聊天内容可能包含身份、联系方式和私密对话" in result.output
    assert "已经抽取过的聊天样本直接读取缓存结果" in result.output
    assert str(input_dir) in result.output
    if approved:
        assert result.exit_code == 0, result.output
        build_client.assert_called_once()
        process_file.assert_called_once()
        assert process_file.call_args.kwargs["task"] == module.TASK
        assert output_dir.exists()
    else:
        assert result.exit_code == 1, result.output
        build_client.assert_not_called()
        process_file.assert_not_called()
        assert not output_dir.exists()
