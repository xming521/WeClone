"""Prompts that turn the facts of one life domain into a summary: B's own full summary, or the shorter one a personal
assistant reads about B."""

# Characters each reader's summary asks for.
SUMMARY_CHARS = {"owner": 200, "assistant": 100}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["title", "summary"],
    "additionalProperties": False,
}

RULES = """{intro}内容按 B 生活的领域分段，每段一个标题和一段总结。
下面是 B 在其中一个领域的画像条目，按出现次数从多到少排列。为这个领域写标题和总结。
1. 标题 2～8 个字，概括这些条目涉及的方面，比如"学业与科研""饮食口味"，不写人名、机构名和数字。
2. 总结{sentences}，合计不超过 {chars} 个字，{focus}，出现次数多的优先。只出现过一次的条目，没有别的内容可写时才用。
3. 只写条目里有的内容，不推断，不评价，不把一次提到的事写成长期习惯。条目互相矛盾时，以年月更晚的为准。
4. 不写主语，不出现"B""我""你""用户"，直接写内容。不写 P1 这类聊天对象编号，提到别人时用"朋友""同事"这类称呼，或者略去。
5. 写到{recent}时带上年份，比如"2024 年起在实习"。

只输出 JSON：{{"title":"标题","summary":"总结"}}

## 条目
格式：属性：内容（维度，出现次数，出现的年月）
"""

SUMMARY_RULES = {
    "owner": RULES.format(
        intro="你在为 B 整理 B 自己的画像总结。B 会直接读这份总结，快速看到从 B 的聊天记录里归纳出了什么。",
        sentences="两到四句话",
        chars=SUMMARY_CHARS["owner"],
        focus="覆盖这个领域最主要的几个方面",
        recent="近况和计划",
    ),
    "assistant": RULES.format(
        intro="你在为 B 的个人助理整理 B 的背景资料。"
        "助理和 B 对话时会一直带着这份资料，用来了解 B 是什么人、在做什么、在意什么。",
        sentences="一到两句话",
        chars=SUMMARY_CHARS["assistant"],
        focus="优先写助理帮 B 做事时用得上的情况",
        recent="近况",
    ),
}


def summary_prompt(reader: str, lines: list[str]) -> str:
    return SUMMARY_RULES[reader] + "\n".join(lines)
