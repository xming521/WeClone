"""Tests for MiniMax LLM provider integration.

Unit tests validate provider presets, temperature clamping, and response_format
handling. Integration tests (skipped without MINIMAX_API_KEY) verify actual
MiniMax API calls.
"""

import os
import re
import sys
from unittest.mock import MagicMock, patch

import pytest


def _extract_json_from_text(text: str) -> str:
    """Re-implementation of extract_json_from_text for test use.

    Strips ``<think>`` blocks and markdown code fences, then finds the
    first JSON object so that the payload can be parsed by Pydantic in
    integration tests.
    """
    # Strip <think>…</think> blocks (MiniMax M2.5 thinking output)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    m = re.search(r"```json\s*(.*?)\s*```", cleaned, re.DOTALL)
    if m:
        return m.group(1).strip()
    if cleaned:
        return cleaned
    # Fallback: find first JSON object in original text (may be inside <think>)
    m = re.search(r"\{[^{}]*\}", text)
    if m:
        return m.group(0)
    return text.strip()


# Mock heavy dependencies (torch, llamafactory, vllm, pyjson5) that are not
# installed in the test environment.  We only need OnlineLLM which lives in
# online_infer.py; its sole dependency on offline_infer is for
# extract_json_from_text which we provide a working re-implementation for.
_mock_offline = MagicMock()
_mock_offline.extract_json_from_text = _extract_json_from_text
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("pyjson5", MagicMock())
sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.lora", MagicMock())
sys.modules.setdefault("vllm.lora.request", MagicMock())
sys.modules.setdefault("vllm.outputs", MagicMock())
sys.modules.setdefault("vllm.sampling_params", MagicMock())
sys.modules.setdefault("llamafactory", MagicMock())
sys.modules.setdefault("llamafactory.data", MagicMock())
sys.modules.setdefault("llamafactory.extras", MagicMock())
sys.modules.setdefault("llamafactory.extras.misc", MagicMock())
sys.modules.setdefault("llamafactory.hparams", MagicMock())
sys.modules.setdefault("llamafactory.model", MagicMock())
sys.modules.setdefault("weclone.core.inference.offline_infer", _mock_offline)

from weclone.core.inference.online_infer import OnlineLLM
from weclone.utils.config_models import (
    LLM_PROVIDER_PRESETS,
    LLMProvider,
    MakeDatasetArgs,
)

# ---------------------------------------------------------------------------
# Unit Tests – Provider presets
# ---------------------------------------------------------------------------

class TestLLMProviderPresets:
    """Test provider preset configuration."""

    def test_minimax_preset_values(self):
        preset = LLM_PROVIDER_PRESETS[LLMProvider.MINIMAX]
        assert preset["base_url"] == "https://api.minimax.io/v1"
        assert preset["model_name"] == "MiniMax-M2.5"

    def test_openai_preset_values(self):
        preset = LLM_PROVIDER_PRESETS[LLMProvider.OPENAI]
        assert preset["base_url"] == "https://api.openai.com/v1"
        assert preset["model_name"] == "gpt-4o-mini"

    def test_deepseek_preset_values(self):
        preset = LLM_PROVIDER_PRESETS[LLMProvider.DEEPSEEK]
        assert preset["base_url"] == "https://api.deepseek.com/v1"
        assert preset["model_name"] == "deepseek-chat"

    def test_custom_provider_not_in_presets(self):
        assert LLMProvider.CUSTOM not in LLM_PROVIDER_PRESETS


class TestMakeDatasetArgsProviderPresets:
    """Test that MakeDatasetArgs applies provider presets correctly."""

    _base_args = {
        "platform": "chat",
        "language": "zh",
    }

    def test_minimax_provider_fills_base_url_and_model(self):
        args = MakeDatasetArgs(llm_provider="minimax", **self._base_args)
        assert args.base_url == "https://api.minimax.io/v1"
        assert args.model_name == "MiniMax-M2.5"

    def test_explicit_base_url_overrides_preset(self):
        args = MakeDatasetArgs(
            llm_provider="minimax",
            base_url="https://api.minimaxi.com/v1",
            **self._base_args,
        )
        assert args.base_url == "https://api.minimaxi.com/v1"
        assert args.model_name == "MiniMax-M2.5"

    def test_explicit_model_name_overrides_preset(self):
        args = MakeDatasetArgs(
            llm_provider="minimax",
            model_name="MiniMax-M2.5-highspeed",
            **self._base_args,
        )
        assert args.base_url == "https://api.minimax.io/v1"
        assert args.model_name == "MiniMax-M2.5-highspeed"

    def test_no_provider_leaves_fields_none(self):
        args = MakeDatasetArgs(**self._base_args)
        assert args.base_url is None
        assert args.model_name is None
        assert args.llm_provider is None

    def test_custom_provider_does_not_fill_defaults(self):
        args = MakeDatasetArgs(
            llm_provider="custom",
            base_url="https://my-server/v1",
            model_name="my-model",
            **self._base_args,
        )
        assert args.base_url == "https://my-server/v1"
        assert args.model_name == "my-model"


# ---------------------------------------------------------------------------
# Unit Tests – Temperature clamping
# ---------------------------------------------------------------------------

class TestTemperatureClamping:
    """Test OnlineLLM.clamp_temperature for MiniMax constraints."""

    def test_zero_temperature_clamped_for_minimax(self):
        result = OnlineLLM.clamp_temperature(0.0, "https://api.minimax.io/v1")
        assert result == 0.01

    def test_negative_temperature_clamped_for_minimax(self):
        result = OnlineLLM.clamp_temperature(-1.0, "https://api.minimax.io/v1")
        assert result == 0.01

    def test_positive_temperature_unchanged_for_minimax(self):
        result = OnlineLLM.clamp_temperature(0.7, "https://api.minimax.io/v1")
        assert result == 0.7

    def test_temperature_one_unchanged_for_minimax(self):
        result = OnlineLLM.clamp_temperature(1.0, "https://api.minimax.io/v1")
        assert result == 1.0

    def test_zero_temperature_unchanged_for_openai(self):
        result = OnlineLLM.clamp_temperature(0.0, "https://api.openai.com/v1")
        assert result == 0.0

    def test_china_endpoint_also_clamped(self):
        result = OnlineLLM.clamp_temperature(0.0, "https://api.minimaxi.com/v1")
        assert result == 0.01

    def test_above_one_clamped_for_minimax(self):
        result = OnlineLLM.clamp_temperature(1.5, "https://api.minimax.io/v1")
        assert result == 1.0

    def test_above_one_unchanged_for_openai(self):
        result = OnlineLLM.clamp_temperature(1.5, "https://api.openai.com/v1")
        assert result == 1.5

    def test_clamp_temperature_none_url(self):
        """Temperature passes through unchanged when base_url is None."""
        result = OnlineLLM.clamp_temperature(0.0, None)
        assert result == 0.0

    def test_clamp_temperature_empty_url(self):
        """Temperature passes through unchanged when base_url is empty string."""
        result = OnlineLLM.clamp_temperature(0.0, "")
        assert result == 0.0


# ---------------------------------------------------------------------------
# Unit Tests – response_format handling
# ---------------------------------------------------------------------------

class TestResponseFormatHandling:
    """Test that response_format is disabled for MiniMax."""

    @patch("weclone.core.inference.online_infer.OpenAI")
    def test_minimax_disables_response_format(self, mock_openai_cls):
        llm = OnlineLLM(
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
            model_name="MiniMax-M2.5",
        )
        assert llm.response_format == ""
        assert llm._supports_response_format is False

    @patch("weclone.core.inference.online_infer.OpenAI")
    def test_openai_keeps_response_format(self, mock_openai_cls):
        llm = OnlineLLM(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o-mini",
        )
        assert llm.response_format == "json_object"
        assert llm._supports_response_format is True

    @patch("weclone.core.inference.online_infer.OpenAI")
    def test_china_minimax_disables_response_format(self, mock_openai_cls):
        llm = OnlineLLM(
            api_key="test-key",
            base_url="https://api.minimaxi.com/v1",
            model_name="MiniMax-M2.5",
        )
        assert llm.response_format == ""

    @patch("weclone.core.inference.online_infer.OpenAI")
    def test_minimax_overrides_explicit_response_format(self, mock_openai_cls):
        """Even when response_format is explicitly passed, MiniMax should clear it."""
        llm = OnlineLLM(
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
            model_name="MiniMax-M2.5",
            response_format="json_object",
        )
        assert llm.response_format == ""
        assert llm._supports_response_format is False

    @patch("weclone.core.inference.online_infer.OpenAI")
    def test_chat_omits_response_format_for_minimax(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        llm = OnlineLLM(
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
            model_name="MiniMax-M2.5",
        )
        llm.chat("Hello")

        _, kwargs = mock_client.chat.completions.create.call_args
        # response_format should NOT be in the call params
        assert "response_format" not in kwargs

    @patch("weclone.core.inference.online_infer.OpenAI")
    def test_chat_includes_response_format_for_openai(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        llm = OnlineLLM(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o-mini",
        )
        llm.chat("Hello")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "response_format" in call_kwargs


# ---------------------------------------------------------------------------
# Unit Tests – Provider detection
# ---------------------------------------------------------------------------

class TestProviderDetection:
    """Test _is_no_response_format_provider detection."""

    def test_detects_minimax_io(self):
        assert OnlineLLM._is_no_response_format_provider("https://api.minimax.io/v1") is True

    def test_detects_minimaxi_com(self):
        assert OnlineLLM._is_no_response_format_provider("https://api.minimaxi.com/v1") is True

    def test_not_detected_for_openai(self):
        assert OnlineLLM._is_no_response_format_provider("https://api.openai.com/v1") is False

    def test_not_detected_for_empty(self):
        assert OnlineLLM._is_no_response_format_provider("") is False

    def test_case_insensitive(self):
        assert OnlineLLM._is_no_response_format_provider("https://API.MINIMAX.IO/v1") is True


# ---------------------------------------------------------------------------
# Integration Tests – MiniMax API (skipped without API key)
# ---------------------------------------------------------------------------

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")


@pytest.mark.skipif(not MINIMAX_API_KEY, reason="MINIMAX_API_KEY not set")
class TestMiniMaxIntegration:
    """Integration tests that call the real MiniMax API."""

    def test_basic_chat_completion(self):
        """Send a simple message and verify a response is returned."""
        llm = OnlineLLM(
            api_key=MINIMAX_API_KEY,
            base_url="https://api.minimax.io/v1",
            model_name="MiniMax-M2.5",
            response_format="",
        )
        response = llm.chat(
            'Respond with exactly: {"status": "ok"}',
            temperature=0.5,
            max_tokens=30,
        )
        assert response is not None
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_json_extraction_without_response_format(self):
        """Verify JSON can be extracted from MiniMax response without response_format."""
        llm = OnlineLLM(
            api_key=MINIMAX_API_KEY,
            base_url="https://api.minimax.io/v1",
            model_name="MiniMax-M2.5",
            response_format="",
        )
        response = llm.chat(
            'You must respond with valid JSON only, no other text. '
            'Output: {"id": 1, "score": 4}',
            temperature=0.5,
            max_tokens=50,
        )
        content = response.choices[0].message.content
        assert content is not None
        json_text = _extract_json_from_text(content)
        assert '"id"' in json_text or '"score"' in json_text

    def test_temperature_clamping_in_api_call(self):
        """Verify that temperature=0 is clamped and doesn't cause API error."""
        llm = OnlineLLM(
            api_key=MINIMAX_API_KEY,
            base_url="https://api.minimax.io/v1",
            model_name="MiniMax-M2.5",
            response_format="",
        )
        # This would fail without clamping since MiniMax rejects temperature=0
        response = llm.chat(
            "Say hello",
            temperature=0,
            max_tokens=10,
        )
        assert response is not None
        assert response.choices[0].message.content is not None

    def test_batch_chat_with_guided_decoding(self):
        """Test batch processing with JSON validation (mimics data cleaning)."""
        from pydantic import BaseModel as PydanticBaseModel

        class SimpleScore(PydanticBaseModel):
            id: int
            score: int

        llm = OnlineLLM(
            api_key=MINIMAX_API_KEY,
            base_url="https://api.minimax.io/v1",
            model_name="MiniMax-M2.5",
            response_format="",
        )
        prompts = [
            'You must respond with valid JSON only. Output: {"id": 1, "score": 4}',
            'You must respond with valid JSON only. Output: {"id": 2, "score": 3}',
        ]
        parsed_results, failed = llm.chat_batch(
            prompts,
            temperature=0.5,
            max_tokens=50,
            guided_decoding_class=SimpleScore,
        )
        # At least one result should parse successfully
        successful = [r for r in parsed_results if r is not None]
        assert len(successful) > 0
        assert hasattr(successful[0], "id")
        assert hasattr(successful[0], "score")
