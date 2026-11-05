"""Prompts and JSON schemas used by the cloud pipeline."""

from __future__ import annotations

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}

STYLE_SUMMARY_SYSTEM = """你是聊天风格分析师。用户会给你若干段同一个人的真实聊天片段（「我」= 这个人，「对方」= 聊天对象）。
请总结这个人的说话风格，供另一个模型模仿。只描述可观察的语言习惯，不评价内容、不推测性格、不做心理分析。"""

STYLE_SUMMARY_USER = """聊天片段：

{conversation}

请输出 5-8 条中文要点，每条一行，以「- 」开头，覆盖：
句长与断句、常见语气词/口头禅、标点习惯（省略号、感叹号、句号缺失等）、称呼与打招呼方式、被问住或不想聊时的回避方式、是否会主动追问。
不要写「他很幽默」「比较内向」这类评价，只写可以直接模仿的具体规则。"""

SCORE_SYSTEM = """你是聊天语料质检员。上游会把真实聊天记录切成“一问一答”样本，用来微调一个模仿「我」说话风格的数字分身。
你的任务：只判断这段对话对“学会这个人怎么说话”有多大训练价值，按 1-5 打分。

评分标准（只看训练价值，不看话题是否高级）：
5 = 高价值。「我」的回复有个性、有具体信息或情绪，能看出说话习惯（口头禅、语气、句长、标点习惯）；对话自洽、上下文完整。
4 = 较好。有明确信息或风格，但篇幅偏短或部分依赖上下文。
3 = 一般。能看懂，但回复比较模板化（“嗯”“好的”“哈哈”），风格信息有限。
2 = 低。几乎都是短应答、表情、寒暄，学不到说话方式。
1 = 无价值。乱码、纯符号、上下文缺失、无法判断谁在说什么。

必须遵守：
- 不要因为内容涉及隐私、敏感、负面情绪而扣分；脱敏已在预处理阶段完成。
- 不要因为话题日常、琐碎而扣分；日常对话恰恰是分身的语料。
- 只评价这段对话本身，不要补充、不要改写、不要评价聊天对象的水平。
- 只输出一个 json 对象，不要 markdown 代码块、不要任何解释文字。
示例 json 输出：{"score": 4, "reason": "回答有个人语气，能看出说话习惯"}
"""

SCORE_USER = """下面是「我」（训练目标）与「对方」的聊天片段：

---
{conversation}
---

请打分，并以 json 格式输出：{{"score": 1-5 的整数, "reason": "不超过 40 字的中文理由"}}。"""


def render_conversation(messages: list[dict], *, max_chars: int) -> str:
    """Render messages as ``对方:`` / ``我:`` lines, truncating from the middle."""
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        speaker = "我" if role == "assistant" else "对方" if role == "user" else None
        if speaker is None:
            continue
        lines.append(f"{speaker}: {content}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n……（中间省略 {len(text) - max_chars} 字）……\n{text[-tail:]}"


def build_score_messages(messages: list[dict], *, max_chars: int = 6000) -> list[dict]:
    return [
        {"role": "system", "content": SCORE_SYSTEM},
        {
            "role": "user",
            "content": SCORE_USER.format(conversation=render_conversation(messages, max_chars=max_chars)),
        },
    ]
