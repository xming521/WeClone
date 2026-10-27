"""Prompts that turn organized topics into the browsable profile hierarchy."""

from weclone.prompts.memory_organization import dimension_name

VALUE_CHARS = 60

SPLIT_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "items": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["name", "items"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["topics"],
    "additionalProperties": False,
}
DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "domains": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "description": {"type": "string"}},
                "required": ["id", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["domains"],
    "additionalProperties": False,
}

SPLIT_RULES = """你在整理 B 的画像。画像按领域、主题、条目三层组织，查找时先选领域，再选主题，然后读主题下的条目。
下面这个主题的条目太多，不便浏览和选取。把它拆成几个更具体的子主题，子主题替代原主题，与同领域其他主题并列。
1. 按条目实际在聊的对象、事情或阶段拆分，同一件事的条目放在一起，不按维度分开。不按人拆分：和同一个人有关的条目，按在聊的事情放进相应子主题。
2. 每个子主题尽量不超过 {leaf_facts} 条，但不要为了凑数拆散同一件事。
3. 子主题名用简洁中文，不超过 12 个字，在本领域内能单独看懂，不与原主题或本领域其他主题重名，不含人名或 P1 这类聊天对象编号。
4. 至少拆成两个子主题。每个条目编号只放进一个子主题，全部条目都要放进子主题。

只输出 JSON：{{"topics":[{{"name":"子主题名","items":[条目编号]}}]}}
"""

DESCRIBE_RULES = """你在整理 B 的画像。查找画像时，先根据要查的内容和领域说明判断话题属于 B 生活的哪个领域。
下面是 B 画像的各个领域，以及每个领域下的主题和条目数。为每个领域写一句说明，概括其中包括哪些内容，带上 B 在这个领域的具体对象或话题词，便于根据聊天内容识别。
说明不超过 50 个字，只根据列出的主题概括，不写列表外的内容。每个领域编号只出现一次。

只输出 JSON：{"domains":[{"id":领域编号,"description":"说明"}]}

## 领域
格式：领域编号 领域名：主题（条目数）、主题（条目数）
"""


def split_prompt(domain: dict, others: list[str], topic: str, facts: list[dict], leaf_facts: int) -> str:
    return (
        SPLIT_RULES.format(leaf_facts=leaf_facts)
        + f"\n## 领域\n{domain['name']}：{domain['description']}\n"
        + "\n## 本领域其他主题\n"
        + ("、".join(others) or "无")
        + f"\n\n## 原主题\n{topic}\n"
        + "\n## 条目\n格式：条目编号 属性：内容（维度）\n"
        + "\n".join(
            f"{i} {fact['attr']}：{' '.join(fact['value'].split())[:VALUE_CHARS]}（{dimension_name(fact['dim'])}）"
            for i, fact in enumerate(facts, 1)
        )
    )


def describe_prompt(domains: list[dict]) -> str:
    return DESCRIBE_RULES + "\n".join(
        f"{i} {domain['name']}："
        + "、".join(f"{topic['name']}（{len(topic['fact_indices'])}）" for topic in domain["topics"])
        for i, domain in enumerate(domains, 1)
    )
