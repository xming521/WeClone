"""Prompts that turn the facts of one life domain into the summary B's personal assistant keeps in context: a long one
from every fact still true, or a short one from the facts at or below a privacy level."""

# Characters each length of summary asks for.
SUMMARY_CHARS = {"long": 300, "short": 100}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["title", "summary"],
    "additionalProperties": False,
}

RULES = """你在为 B 的个人助理整理 B 的背景资料。助理事先对 B 一无所知，之后一段时间里每次和 B 对话前都会先读这份资料，靠它了解 B 是什么人、在做什么、在意什么。内容按 B 生活的领域分段，每段一个标题和一段总结。
下面是 B 在其中一个领域的画像条目，每条是从 B 的聊天记录里整理出的一个关于 B 的事实，按出现次数从多到少排列。为这个领域写标题和总结。
1. 标题 2～8 个字，只说这一段讲 B 的哪些方面，比如"学业与科研""饮食口味"，具体内容留给总结。
2. 总结{sentences}，合计不超过 {chars} 个字，写 B 在这个领域长期成立的情况。条目写不下时按这个标准取舍：助理缺了这一条，会不会弄错 B 是谁、处境如何，或者给出不合适的建议和安排。会的先写，影响越大越靠前，不影响的细节提得再多也不写。条目涉及几个方面时，每个方面里会影响助理的都要写到，不要只写其中一两个方面。几周内就会办完的事很快过时，不写。出现次数只说明聊得多，不说明重要，只出现过一次的条目也可能最该写。
3. 只写条目里有的内容，不推断，不评价，不把一次提到的事写成长期习惯。条目互相矛盾时，以年月更晚的为准。
4. 用陈述句直接写事实，数字、时间和程度照条目写具体，不写铺垫和空泛的形容，不转述聊天经过。不写主语，不出现"B""我""你""用户"。不写 P1 这类聊天对象编号，提到别人时用"朋友""同事"这类称呼，或者略去。
5. 写到{recent}时带上年份，比如"2024 年起在实习"。

只输出 JSON：{{"title":"标题","summary":"总结"}}

## 条目
格式：属性：内容（维度，出现次数，出现的年月）
"""

SUMMARY_RULES = {
    "long": RULES.format(
        sentences="两到五句话",
        chars=SUMMARY_CHARS["long"],
        recent="近况和计划",
    ),
    "short": RULES.format(
        sentences="一到两句话",
        chars=SUMMARY_CHARS["short"],
        recent="近况",
    ),
}


def summary_prompt(length: str, lines: list[str]) -> str:
    return SUMMARY_RULES[length] + "\n".join(lines)
