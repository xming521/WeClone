"""Prompts for the harness: choosing which profile facts go into the reply model's context."""

PROFILE_DOMAIN_QUESTION = "B 要回复 A 的最后一条消息。写这条回复最需要查 B 生活中哪个领域的信息？"
PROFILE_TOPIC_QUESTION = "B 要回复 A 的最后一条消息。写这条回复最需要查 B 画像里的哪一类信息？"

PROFILE_ROUTE_SCHEMA = {
    "type": "object",
    "properties": {"topics": {"type": "array", "items": {"type": "string"}}},
    "required": ["topics"],
    "additionalProperties": False,
}
PROFILE_ROUTE_RULES = """你在为一个"以 B 本人身份自动回复聊天"的系统做画像路由。

B 的画像分成若干领域，领域下面分主题。系统写 B 的下一条回复之前，先选出几个主题，把这些主题下的条目作为候选，再从候选里挑出要用的。下面先给出画像的领域和主题（只有名称和属性，没有具体条目），最后给出一段聊天，停在 A 发完消息、轮到 B 回复的地方。请选出最可能包含"写这条回复要用到的条目"的主题，按可能性从高到低排序，最多 {max_topics} 个。

判断要求：标准是"写这条回复要用到"，不是"和话题相关"。要考虑隐含的需要，A 没提到的方面也可能决定 B 怎么回。比如 A 约 B 周末出去，B 怎么回取决于 B 的时间安排、身体状况、手头的计划或经济情况。

输出 JSON，如 {{"topics": ["3|1", "0|2"]}}。topics 是主题编号的数组，按可能性从高到低排列。
"""

PROFILE_FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "must": {"type": "array", "items": {"type": "string"}},
        "helpful": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["must", "helpful"],
    "additionalProperties": False,
}
PROFILE_FILTER_RULES = """你在为一个"以 B 本人身份自动回复聊天"的系统做画像过滤。

系统写 B 的下一条回复之前，先从 B 的画像里路由出一批候选条目，再由你决定哪些放进回复模型的上下文。回复模型往往会把上下文里的条目说出来，或者据此作答。下面是一段聊天，停在 A 发完消息、轮到 B 回复的地方，后面列出这次路由出的候选条目。请选出要放进上下文的条目，分两档：
- must：不知道这条，B 的回复很可能编造事实、和 B 的真实情况矛盾，或者明显不像 B。
- helpful：不知道也能自然地回，知道了回复会更具体、更像 B。
其余条目不选。放进上下文的条目如果跑题、过时、不得体或误导，回复就会出错或明显变差，所以选之前要想清楚。

判断要求：
1. 先看最后一条消息和 B 这次要回什么，再到候选条目里找要用到的。
2. 标准是"写这条回复要用到"，不是"和话题相关"。要考虑隐含的需要，A 没提到的条目也可能决定 B 怎么回。比如 A 约 B 周末出去，B 怎么回取决于 B 的时间安排、身体状况、手头的计划或经济情况。
3. 上文已经明说、B 不知道这条也能照样回的，不选。寒暄、附和、表情、确认收到这类回复一般用不到画像，两档都留空。
4. 不选这几种：和当前话题无关的，按聊天时间看很可能已经不成立的（条目后面的日期是它在聊天里出现过的时间范围），这个场合不适合主动说出的私密或敏感信息，会诱导回复编造条目里没有的细节的。
5. 一个条目只能放在一档，每档按重要性排序，一般不超过 5 条。

输出 JSON，如 {"must": ["F12"], "helpful": ["F3", "F40"]}。must 和 helpful 都是条目编号的数组，都用不到就输出 {"must": [], "helpful": []}。
"""


def chat_state(chat: str, now: str, contact: str | None) -> str:
    """The chat as both routers and the filter see it: time, B's name for A, then the recent messages."""
    lines = [f"聊天时间：{now}"]
    if contact:
        lines.append(f"B 给 A 的备注名：{contact}")
    return "\n".join([*lines, "上文（最后一条是 A 发的，接下来轮到 B 回复）：", chat])


def route_prompt(tree: str, state: str, max_topics: int) -> str:
    """Rules and the profile tree first (the same for every reply, so the prefix caches), the chat last."""
    return (
        PROFILE_ROUTE_RULES.format(max_topics=max_topics)
        + "\n## B 的画像结构\n格式：主题编号 主题名：属性\n\n"
        + tree
        + "\n\n## 聊天\n"
        + state
    )


def filter_prompt(state: str, candidates: list[str]) -> str:
    return PROFILE_FILTER_RULES + "\n## 聊天\n" + state + "\n\n## 候选条目\n" + "\n".join(candidates)
