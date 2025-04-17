from llamafactory.data.formatter import FunctionFormatter, StringFormatter, ToolFormatter, EmptyFormatter
from llamafactory.data.template import register_template

default_prompt = "请你扮演一名人类，不要说自己是人工智能"


def template_register():
    register_template(
        name="chatglm3-weclone",
        default_system=(
            default_prompt
        ),
        format_user=StringFormatter(slots=[{"token": "<|user|>"}, "\n", "{{content}}", {"token": "<|assistant|>"}]),
        format_assistant=StringFormatter(slots=["\n", "{{content}}"]),
        format_system=StringFormatter(slots=[{"token": "<|system|>"}, "\n", "{{content}}"]),
        format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
        format_observation=StringFormatter(
            slots=[{"token": "<|observation|>"}, "\n", "{{content}}", {"token": "<|assistant|>"}]
        ),
        format_tools=ToolFormatter(tool_format="glm4"),
        format_prefix=EmptyFormatter(slots=[{"token": "[gMASK]"}, {"token": "sop"}]),
        stop_words=["<|user|>", "<|observation|>"],
        efficient_eos=True,
    )
