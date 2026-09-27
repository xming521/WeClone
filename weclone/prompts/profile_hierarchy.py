"""Attribute-only hierarchy generation."""

import json

from weclone.prompts.memory_organization import DIMENSIONS

SUBGROUP = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "items": {"type": "array", "items": {"type": "integer"}}},
    "required": ["name", "items"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {"anyOf": [{"type": "integer"}, SUBGROUP]},
                    },
                },
                "required": ["name", "items"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["groups"],
    "additionalProperties": False,
}
RULES = """将当前维度的画像属性组织成便于人浏览的层级结构，当前维度已是一级中心。
按属性的实际含义形成多个二级主题；某主题内部存在清晰子主题时再设三级主题，否则直接挂属性。主题名须概括成员，不用某个狭窄属性代替整个主题；不预设类别或数量。
根据生活领域和对象归组，不因共有“目标、计划、偏好”等表达而归组。保留对象、阶段和范围区别；含义不明时不猜测，不强塞到具体主题。
输入id是属性编号，attr是原属性名。所有属性编号在整棵树中各出现一次，不合并、改写或新增属性。
只输出JSON对象，groups是二级主题列表；主题的name是简洁中文组名，items是成员列表，成员可为属性编号或三级主题对象；三级主题的items只能包含属性编号。各主题非空。
格式：{"groups":[{"name":"二级主题","items":[1,{"name":"三级主题","items":[2,3]}]}]}。
"""


def prompt(dim: int, attributes: list[dict]) -> str:
    return (
        RULES
        + "\n当前维度："
        + DIMENSIONS[dim].split("：")[0]
        + "\n属性：\n"
        + json.dumps(attributes, ensure_ascii=False, separators=(",", ":"))
    )
