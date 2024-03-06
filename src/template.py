from llmtuner.data.formatter import FunctionFormatter, StringFormatter
from llmtuner.data.template import _register_template


def template_register():
    _register_template(
        name="chatglm3-weclone",
        default_system=(
            "请你扮演一名人类，不要说自己是人工智能"
        ),
        format_user=StringFormatter(slots=[{"token": "<|user|>"}, "\n", "{{content}}", {"token": "<|assistant|>"}]),
        format_assistant=StringFormatter(slots=["\n", "{{content}}"]),
        format_system=StringFormatter(slots=[{"token": "[gMASK]"}, {"token": "sop"}, {"token": "<|system|>"}, "\n", "{{content}}"]),
        format_function=FunctionFormatter(slots=["{{name}}\n{{arguments}}"]),
        format_observation=StringFormatter(slots=[{"token": "<|observation|>"}, "\n", "{{content}}"]),
        stop_words=["<|user|>", "<|observation|>"],
        efficient_eos=True,
        force_system=True
    )
