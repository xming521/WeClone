"""Numbers and lines the atlas intro opens with, read from the chats the profile was built from."""

import re
from collections import Counter

from weclone.server.source_chat import ROOT, sample_id
from weclone.utils import secure_storage as secure

# Lines about these topics never appear in the intro, nor do chats behind facts filed under them.
SENSITIVE = re.compile(
    "健康|身体|形象|心理|情绪|睡眠|精力|过敏|疾病|病|医|药|体检|监测|恋爱|亲密|婚恋|家庭|伴侣|父母|收入|经济|投资"
    "|资产|借贷|欠款|消费|理财|交易|保险|保障|补贴|存款|储蓄|账户|密码|移民|签证|党籍|宗教|政治|隐私|保密|签署|签约"
    "|出售|个人信息|身份|地域|迁居|居住|住址|地址|元|万"
)
# Stickers, links, mentions and long digit runs.
NOISE = re.compile(r"[\[\]<>@]|http|www\.|\d{5,}")
# "speaker：text" lines of sources that carry their conversation in the content.
SPEAKER_LINE = re.compile(r"^\s*([^：:\n]{1,12})[：:]\s*(.+)$")
MIN_CHARS, MAX_CHARS = 6, 18
MAX_SENTENCES = 400


def fact_paths(data: dict) -> dict[int, tuple[int | None, list[str]]]:
    """Dimension and names from the dimension down to the attribute, for every fact index."""
    paths = {}

    def visit(items, names, dim):
        for item in items:
            if "attr" in item:
                for index in item.get("fact_indices", []):
                    paths[index] = (dim, [*names, item["attr"]])
            else:
                visit(item.get("items", []), [*names, item.get("name", "")], dim)

    for dimension in data.get("dimensions", []):
        visit(dimension.get("groups", []), [dimension.get("name", "")], dimension.get("dim"))
    return paths


def shown(text: str) -> bool:
    return MIN_CHARS <= len(text) <= MAX_CHARS and not SENSITIVE.search(text) and not NOISE.search(text)


def spread(items: list, count: int) -> list:
    """`count` items evenly spaced through `items`."""
    if len(items) <= count:
        return items
    return [items[(2 * index + 1) * len(items) // (2 * count)] for index in range(count)]


def chat_lines(data, facts_of, paths):
    """Owner messages of every chat sample referenced by a source, each sample counted once.

    Returns None when no chat file can be read, e.g. for sources that carry their conversation inline.
    """
    files = {}
    linked = {}
    for source_id, source in data.get("sources", {}).items():
        for origin in source.get("origins") or []:
            if not isinstance(origin, dict) or not isinstance(origin.get("file"), str):
                continue
            path = secure.logical_path(origin["file"])
            if path.suffix != ".json" or "archive" in path.parts:
                continue
            path = path if path.is_absolute() else ROOT / path
            files[path] = None
            # State and event extraction read the same chats from files of the same name.
            key = (path.name, str(origin.get("sample_id")))
            linked.setdefault(key, set()).update(facts_of.get(source_id, ()))
    seen = set()
    total = 0
    times = []
    lines = []
    for path in files:
        try:
            samples = secure.read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(samples, list):
            continue
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            key = (path.name, sample_id(sample, index))
            if key in seen:
                continue
            seen.add(key)
            time = sample.get("time") if isinstance(sample.get("time"), str) else None
            if time:
                times.append(time)
            facts = linked.get(key, set())
            # Only chats that produced a fact lend lines, and none behind a sensitive one.
            quotable = bool(facts) and not any(
                SENSITIVE.search(name) for fact in facts for name in paths.get(fact, (None, []))[1]
            )
            dims = Counter(paths[fact][0] for fact in facts if fact in paths)
            dim = dims.most_common(1)[0][0] if dims else None
            for message in sample.get("messages") or []:
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                content = message.get("content")
                if not isinstance(content, str):
                    continue
                # Newlines separate consecutive messages merged during preprocessing.
                for part in content.replace("\r\n", "\n").split("\n"):
                    part = part.strip()
                    if not part:
                        continue
                    total += 1
                    if quotable and shown(part):
                        lines.append({"text": part, "time": time or "", "dim": dim})
    if not seen:
        return None
    return total, times, lines


def inline_lines(data, facts_of, paths):
    """Lines of the speaker who most often has the last word in the sources' own content."""
    sources = sorted(
        (
            (source_id, source)
            for source_id, source in data.get("sources", {}).items()
            if isinstance(source.get("content"), str)
        ),
        key=lambda item: str(item[1].get("sample_time") or ""),
    )
    parsed = []
    last = Counter()
    for source_id, source in sources:
        pairs = [
            (match.group(1).strip(), match.group(2).strip())
            for line in source["content"].split("\n")
            if (match := SPEAKER_LINE.match(line))
        ]
        if pairs:
            last[pairs[-1][0]] += 1
        parsed.append((source_id, source, pairs))
    owner = last.most_common(1)[0][0] if last else None
    lines = []
    for source_id, source, pairs in parsed:
        facts = facts_of.get(source_id, ())
        if any(SENSITIVE.search(name) for fact in facts for name in paths.get(fact, (None, []))[1]):
            continue
        dim = next((paths[fact][0] for fact in facts if fact in paths), None)
        lines.extend(
            {"text": text, "time": str(source.get("sample_time") or ""), "dim": dim}
            for speaker, text in pairs
            if speaker == owner and shown(text)
        )
    demo = data.get("_demo") or {}
    months = demo.get("messages_by_month") or {}
    total = demo.get("message_total") or sum(count for count in months.values() if count > 0) or None
    times = sorted(month for month, count in months.items() if count > 0) or [
        str(source["sample_time"]) for _, source in sources if source.get("sample_time")
    ]
    return total, times, lines


def profile_intro(data: dict) -> dict:
    """How many messages the owner sent, over which years, and lines they said, oldest first."""
    paths = fact_paths(data)
    facts_of = {}
    for index, fact in enumerate(data.get("facts", [])):
        # A chat behind a rejected fact only counts towards the total.
        if fact.get("status") == "rejected":
            continue
        for source_id in fact.get("source_ids", []):
            facts_of.setdefault(source_id, set()).add(index)
    total, times, lines = chat_lines(data, facts_of, paths) or inline_lines(data, facts_of, paths)
    unique = {}
    for line in sorted(lines, key=lambda line: line["time"]):
        unique.setdefault(line["text"], line)
    return {
        "message_total": total,
        "first_time": min(times) if times else None,
        "last_time": max(times) if times else None,
        "sentences": spread(list(unique.values()), MAX_SENTENCES),
    }
