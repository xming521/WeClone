import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
    assert (config["request_interval_seconds"], config["capacity_cooldown_seconds"]) == (0.5, 10.0)
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


@pytest.mark.parametrize(
    "command,module",
    [("distill-profile", distill_profile), ("distill-event", distill_event)],
)
@pytest.mark.parametrize("answer,approved", [("", False), ("\n", False), ("n\n", False), ("y\n", True)])
def test_distillation_requires_consent_before_writes_or_model_calls(
    tmp_path, monkeypatch, command, module, answer, approved,
):
    cli_module = importlib.import_module("weclone.cli")
    monkeypatch.setattr(cli_module, "_check_project_root", lambda: None)
    monkeypatch.setattr(cli_module, "_check_versions", lambda: None)
    monkeypatch.setattr(cli_module, "load_config", lambda **_: SimpleNamespace(full_log=False))
    monkeypatch.setattr(cli_module, "configure_log_level_from_config", lambda: None)
    monkeypatch.setenv("WECLONE_CONFIG_PATH", "settings.jsonc")

    config_path = tmp_path / "settings.jsonc"
    config_path.write_text(json.dumps({
        "codex_exec_args": {
            "model": "fixture", "effort": "medium", "batch_size": 1,
            "timeout": 30, "command": "codex", "sandbox": "read-only",
        },
        "agent_distill_args": {"overwrite": True, "llm_provider": "codex_exec"},
    }), encoding="utf-8")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "chat.json").write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "output"

    client = Mock()
    build_client = Mock(return_value=client)
    process_file = Mock(return_value=(0, 0))
    monkeypatch.setattr(llm_client, "build_llm_client", build_client)
    monkeypatch.setattr(module, "process_file", process_file)

    result = CliRunner().invoke(
        cli_module.cli,
        ["--config-path", str(config_path), command,
         "--input-dir", str(input_dir), "--output-dir", str(output_dir)],
        input=answer,
    )

    assert "聊天内容可能包含身份、联系方式和私密对话" in result.output
    assert "overwrite=true" in result.output
    assert str(input_dir) in result.output
    if approved:
        assert result.exit_code == 0, result.output
        build_client.assert_called_once()
        process_file.assert_called_once()
        assert output_dir.exists()
    else:
        assert result.exit_code == 1, result.output
        build_client.assert_not_called()
        process_file.assert_not_called()
        assert not output_dir.exists()
