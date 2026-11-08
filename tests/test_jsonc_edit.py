from pathlib import Path

import pyjson5
import pytest

from weclone.utils.jsonc_edit import remove_value, set_value

TEMPLATE = (Path(__file__).resolve().parents[1] / "settings.template.jsonc").read_text(encoding="utf-8")

TEXT = """{
    // backend
    "agent_distill_args": {
        "llm_provider": "codex_exec", // codex_exec or api
        "batch_size": 8
    },
    "make_dataset_args": {
        "base_url": "https://example.com/v1", // a URL is not a comment
        "model_name": "m",
    },
    "empty": {},
    "inline": {"a": 1}
}
"""


def changed_lines(before: str, after: str) -> list[str]:
    old = set(before.splitlines())
    return [line for line in after.splitlines() if line not in old]


def test_replacing_a_value_keeps_the_rest_of_the_file():
    text = set_value(TEMPLATE, ("agent_distill_args", "llm_provider"), "api")
    assert pyjson5.loads(text)["agent_distill_args"]["llm_provider"] == "api"
    assert changed_lines(TEMPLATE, text) == ['        "llm_provider": "api",']
    assert len(text.splitlines()) == len(TEMPLATE.splitlines())


def test_a_new_key_goes_after_the_last_one_and_its_comment():
    text = set_value(TEXT, ("agent_distill_args", "timeout"), 300)
    assert '        "batch_size": 8,\n        "timeout": 300\n    },' in text
    assert "// codex_exec or api" in text
    text = set_value(TEXT, ("make_dataset_args", "llm_api_key"), "sk-1")
    assert '        "model_name": "m",\n        "llm_api_key": "sk-1"\n    },' in text
    assert pyjson5.loads(text)["make_dataset_args"]["base_url"] == "https://example.com/v1"


def test_missing_objects_are_added_on_the_way():
    text = set_value(TEXT, ("api_args", "headers", "X-Title"), "WeClone")
    assert pyjson5.loads(text)["api_args"] == {"headers": {"X-Title": "WeClone"}}
    assert (
        '    "api_args": {\n        "headers": {\n            "X-Title": "WeClone"\n        }\n    }\n}'
        in text
    )


def test_empty_and_one_line_objects():
    text = set_value(TEXT, ("empty", "a"), True)
    assert '    "empty": {\n        "a": true\n    },' in text
    text = set_value(TEXT, ("inline", "b"), "x")
    assert '"inline": {"a": 1, "b": "x"}' in text


def test_objects_are_written_at_their_depth():
    text = set_value(TEXT, ("agent_distill_args", "batch_size"), {"a": [1, 2]})
    assert (
        '        "batch_size": {\n            "a": [\n                1,\n                2\n            ]\n        }\n'
        in text
    )


def test_removing_a_key_takes_its_line():
    text = remove_value(TEXT, ("agent_distill_args", "llm_provider"))
    assert "llm_provider" not in text and "// codex_exec or api" not in text
    assert pyjson5.loads(text)["agent_distill_args"] == {"batch_size": 8}
    text = remove_value(TEXT, ("agent_distill_args", "batch_size"))
    assert '"llm_provider": "codex_exec" // codex_exec or api\n    },' in text
    assert remove_value(TEXT, ("missing", "key")) == TEXT


def test_unreadable_text_is_refused():
    with pytest.raises(ValueError):
        set_value('{"a": 1,, }', ("a",), 2)
    with pytest.raises(ValueError):
        set_value(TEXT, ("agent_distill_args", "llm_provider", "x"), 1)
