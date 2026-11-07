"""Prompt that reads B's whole profile at once for what no fact says: implicit facts about who B is, B's situation and
abilities, and deep insights, a short summary of what kind of person B is."""

LISTS = ("implicit_facts", "insights")
ITEMS = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string"},
            "basis": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["conclusion", "basis"],
        "additionalProperties": False,
    },
}
INSIGHTS_SCHEMA = {
    "type": "object",
    "properties": dict.fromkeys(LISTS, ITEMS),
    "required": list(LISTS),
    "additionalProperties": False,
}
# A profile of 1042 facts gets 12 insights: uncapped runs over one that size wrote 12 to 14. The cap grows with the
# square root of the fact count, since more facts mostly add evidence for patterns already there.
REFERENCE_FACTS, REFERENCE_INSIGHTS = 1042, 12

RULES = """下面是从 B 的聊天记录整理出的画像事实，按 B 生活的板块分组。每行开头是编号；括号里是时效：stable 长期成立，current 当前情况，superseded 已被新情况取代，dated 某个时间点的事。

请推出这些事实没有直接写出、但能由事实推出的两类结论：
- implicit_facts（隐含事实）：B 是谁、处境如何、能做什么。
- insights（深度洞察）：总结 B 是什么样的人，B 的想法和做法背后有哪些共同的原因。

要求：
1. 不重复、不改写已经写出的事实，放在大多数人身上也成立的说法不写。
2. 每条只写一个结论，一句话，用平常说话的词，不超过 30 字。
3. 洞察是总结：每条写一个共同的原因或规律，原因相同的合成一条。每条要能解释多条事实，最好是不同板块里看似无关的事实，或者能预测 B 在没出现过的情境里会怎么做。
4. 有和结论相反的事实时，要么在结论里解释它，要么不写这条。过去的情况不当作现在。
5. 只写各种解释里最说得通的那一种，更弱的推测不写。不罗列其他可能，不写某种判断不能成立之类的提醒。
6. basis 列出用到的事实编号。

只输出 JSON：{"implicit_facts":[{"conclusion":"…","basis":[编号]}],"insights":[{"conclusion":"…","basis":[编号]}]}

事实："""


def insight_limit(fact_count: int) -> int:
    return max(1, round(REFERENCE_INSIGHTS * (fact_count / REFERENCE_FACTS) ** 0.5))


def insights_prompt(groups: list[tuple[str, list[str]]], limit: int) -> str:
    """The rules, then each domain's numbered fact lines, then the cap. The rules come first and alike for everyone, so
    model services can cache them."""
    facts = "".join(f"\n\n## {domain}\n" + "\n".join(lines) for domain, lines in groups)
    return f"{RULES}{facts}\n\n深度洞察最多写 {limit} 条，优先写能解释最多事实的。"
