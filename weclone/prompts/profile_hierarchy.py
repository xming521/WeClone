"""Prompts that group organized profile facts into the topics of the browsable profile hierarchy."""

from weclone.prompts.memory_organization import dimension_name, domain_block

VALUE_CHARS = 60

TOPIC_ITEMS = {
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
TOPIC_SCHEMA = {
    "type": "object",
    "properties": {"description": {"type": "string"}, "topics": TOPIC_ITEMS},
    "required": ["description", "topics"],
    "additionalProperties": False,
}
SPLIT_SCHEMA = {
    "type": "object",
    "properties": {"topics": TOPIC_ITEMS},
    "required": ["topics"],
    "additionalProperties": False,
}

TOPIC_RULES = """你在整理 B 的画像。画像按领域、主题、条目三层组织，查找时先根据领域说明选领域，再选主题，然后读主题下的条目。
现在把其中一个领域的条目分成主题，并为这个领域写一句说明。
1. 按条目实际在聊的对象、事情或阶段分主题，同一件事的条目放在一起，不按维度分开。不按人分：和同一个人有关的条目，按在聊的事情放进相应主题。
2. 每个主题不超过 {max_facts} 条，超过的话题拆成几个并列主题；主题一般不少于 {min_facts} 条，零散条目并入最接近的主题，不单独成主题。整个领域条目很少时可以只有一个主题。
3. 主题名用简洁中文，不超过 12 个字，在本领域内能单独看懂，互不重名，不含人名或 P1 这类聊天对象编号。
4. 每个条目编号只放进一个主题，全部条目都要放进主题。
5. 说明概括这个领域包括哪些内容，带上 B 在这个领域的具体对象或话题词，便于根据聊天内容识别，不超过 50 个字，只根据列出的条目概括。

只输出 JSON：{{"description":"说明","topics":[{{"name":"主题名","items":[条目编号]}}]}}
"""

SPLIT_RULES = """你在整理 B 的画像。画像按领域、主题、条目三层组织，查找时先选领域，再选主题，然后读主题下的条目。
下面这个主题的条目太多，不便浏览和选取。把它拆成几个更具体的子主题，子主题替代原主题，与同领域其他主题并列。
1. 按条目实际在聊的对象、事情或阶段拆分，同一件事的条目放在一起，不按维度分开。不按人拆分：和同一个人有关的条目，按在聊的事情放进相应子主题。
2. 每个子主题尽量不超过 {leaf_facts} 条，但不要为了凑数拆散同一件事。
3. 子主题名用简洁中文，不超过 12 个字，在本领域内能单独看懂，不与原主题或本领域其他主题重名，不含人名或 P1 这类聊天对象编号。
4. 至少拆成两个子主题。每个条目编号只放进一个子主题，全部条目都要放进子主题。

只输出 JSON：{{"topics":[{{"name":"子主题名","items":[条目编号]}}]}}
"""


def fact_lines(facts: list[dict]) -> str:
    return "\n## 条目\n格式：条目编号 属性：内容（维度）\n" + "\n".join(
        f"{i} {fact['attr']}：{' '.join(fact['value'].split())[:VALUE_CHARS]}（{dimension_name(fact['dim'])}）"
        for i, fact in enumerate(facts, 1)
    )


def topic_prompt(domains: list[dict], domain: dict, facts: list[dict], min_facts: int, max_facts: int) -> str:
    # Rules and the domain list come first so every domain's request shares the cached prefix.
    return (
        TOPIC_RULES.format(min_facts=min_facts, max_facts=max_facts)
        + domain_block(domains)
        + f"\n## 当前领域\n{domain['name']}：{domain['description']}\n"
        + fact_lines(facts)
    )


def split_prompt(domain: dict, others: list[str], topic: str, facts: list[dict], leaf_facts: int) -> str:
    return (
        SPLIT_RULES.format(leaf_facts=leaf_facts)
        + f"\n## 领域\n{domain['name']}：{domain['description']}\n"
        + "\n## 本领域其他主题\n"
        + ("、".join(others) or "无")
        + f"\n\n## 原主题\n{topic}\n"
        + fact_lines(facts)
    )
