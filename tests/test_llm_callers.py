import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from weclone.core.inference import LLMResponse, RetryPolicy


ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    """Load caller definitions without executing CLI configuration or model initialization."""
    tree = ast.parse((ROOT / path).read_text())
    nodes = [
        node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            node.decorator_list = [
                d for d in node.decorator_list if isinstance(d, ast.Name) and d.id == "dataclass"
            ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
    return namespace


class ScoreWithId(BaseModel):
    id: int
    score: int


def test_cleaning_scores_preserve_alignment_and_parameters():
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.chat_batch.return_value = [
        LLMResponse(ok=True, parsed_model=ScoreWithId(id=11, score=4)),
        LLMResponse(ok=False, error="schema failure"),
    ]
    factory = Mock(return_value=client)
    namespace = definitions(
        "weclone/data/clean/strategies.py",
        {"OlineLLMCleaningStrategy"},
        {
            "dataclass": dataclass,
            "CleaningStrategy": object,
            "OpenAICompatibleClient": factory,
            "QaPairScoreWithId": ScoreWithId,
            "PromptTemplate": PromptTemplate,
            "CLEAN_PROMPT": "id={id}; messages={messages}",
            "pd": pd,
            "logger": Mock(),
            "tqdm": lambda value, **kwargs: value,
        },
    )
    strategy = namespace["OlineLLMCleaningStrategy"]()
    strategy.make_dataset_config = SimpleNamespace(
        llm_api_key="fixture",
        base_url="http://example.invalid",
        model_name="fixture-model",
        clean_batch_size=2,
    )
    rows = [
        SimpleNamespace(
            id=i, images=[], score=None, messages=[SimpleNamespace(role="user", content=f"question {i}")]
        )
        for i in (11, 22)
    ]
    strategy.judge(rows)
    assert [r.score for r in rows] == [4, 0]
    assert factory.call_args.kwargs["max_workers"] == 7
    assert client.chat_batch.call_args.kwargs == {"temperature": 0, "response_model": ScoreWithId}
    assert len(client.chat_batch.call_args.args[0]) == 2
    client.__exit__.assert_called_once()


def test_vision_payload_and_retry_contract(tmp_path):
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.chat.return_value = LLMResponse(ok=True, text="description")
    factory = Mock(return_value=client)
    import base64
    import os
    from openai import APIConnectionError

    namespace = definitions(
        "weclone/data/utils.py",
        {"ImageToTextProcessor"},
        {
            "base64": base64,
            "os": os,
            "Path": Path,
            "logger": Mock(),
            "OpenAICompatibleClient": factory,
            "RetryPolicy": RetryPolicy,
            "APIConnectionError": APIConnectionError,
        },
    )
    image_path = tmp_path / "fixture.png"
    image_path.write_bytes(b"fixture image")
    processor = namespace["ImageToTextProcessor"](
        "http://example.invalid/v1", "fixture", "vision", SimpleNamespace()
    )
    assert processor.describe_image(str(image_path)) == "description"
    params = client.chat.call_args.kwargs
    messages = client.chat.call_args.args[0]
    assert params == {"max_tokens": 1000, "temperature": 0.1}
    assert messages[0]["content"][0]["text"] == processor.prompt
    assert messages[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    policy = factory.call_args.kwargs["retry_policy"]
    assert (policy.max_retries, policy.base_delay, policy.max_delay) == (5, 15.0, 300.0)
    assert factory.call_args.kwargs["timeout"] == 60
