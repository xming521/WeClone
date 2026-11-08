"""Changes values in JSONC / JSON5 text in place, keeping comments, key order and layout. The web settings change a
few keys of settings.jsonc, a file people also edit and comment by hand."""

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pyjson5

TOKEN = re.compile(
    r"""
    (?P<space>\s+)
    | (?P<comment>//[^\n]*|/\*.*?\*/)
    | (?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
    | (?P<punct>[{}\[\]:,])
    | (?P<atom>[^\s{}\[\]:,"'/]+)
    """,
    re.VERBOSE | re.DOTALL,
)


@dataclass
class Member:
    key: str
    start: int
    value_start: int
    value_end: int
    comma: int | None = None
    node: "Node | None" = None


@dataclass
class Node:
    """An object: where its braces are, and its members in file order."""

    start: int
    end: int
    members: list[Member] = field(default_factory=list)

    def member(self, key: str) -> Member | None:
        # A key written twice counts where it is last, as when the file is read.
        return next((item for item in reversed(self.members) if item.key == key), None)


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = []
        position = 0
        while position < len(text):
            match = TOKEN.match(text, position)
            if match is None:
                raise ValueError(f"unreadable JSONC at offset {position}")
            if match.lastgroup not in ("space", "comment"):
                self.tokens.append((match.lastgroup, match.start(), match.end()))
            position = match.end()
        self.index = 0

    def take(self, expected: str | None = None) -> tuple[str, int, int]:
        if self.index >= len(self.tokens):
            raise ValueError("unexpected end of JSONC")
        token = self.tokens[self.index]
        if expected is not None and self.text[token[1] : token[2]] != expected:
            raise ValueError(f"expected {expected!r} at offset {token[1]}")
        self.index += 1
        return token

    def peek(self) -> str:
        return (
            self.text[self.tokens[self.index][1] : self.tokens[self.index][2]]
            if self.index < len(self.tokens)
            else ""
        )

    def value(self) -> tuple[int, int, Node | None]:
        kind, start, end = self.take()
        text = self.text[start:end]
        if text == "{":
            node = Node(start, -1)
            while self.peek() != "}":
                key_kind, key_start, key_end = self.take()
                raw = self.text[key_start:key_end]
                key = pyjson5.loads(raw) if key_kind == "string" else raw
                self.take(":")
                value_start, value_end, child = self.value()
                member = Member(str(key), key_start, value_start, value_end, node=child)
                if self.peek() == ",":
                    member.comma = self.take()[1]
                elif self.peek() != "}":
                    raise ValueError(f"expected ',' or '}}' at offset {value_end}")
                node.members.append(member)
            node.end = self.take("}")[1]
            return start, node.end + 1, node
        if text == "[":
            last = end
            while self.peek() != "]":
                last = self.value()[1]
                if self.peek() == ",":
                    self.take()
                elif self.peek() != "]":
                    raise ValueError(f"expected ',' or ']' at offset {last}")
            return start, self.take("]")[2], None
        if kind in ("string", "atom"):
            return start, end, None
        raise ValueError(f"unexpected {text!r} at offset {start}")


def parse(text: str) -> Node:
    parser = _Parser(text)
    _, _, root = parser.value()
    if root is None or parser.index != len(parser.tokens):
        raise ValueError("JSONC must hold one object")
    return root


def _line_start(text: str, position: int) -> int:
    return text.rfind("\n", 0, position) + 1


def _indent(text: str, position: int) -> str:
    start = _line_start(text, position)
    return re.match(r"[ \t]*", text[start:]).group()


def _unit(text: str) -> str:
    match = re.search(r"\n([ \t]+)\S", text)
    return match.group(1) if match else "    "


def _dump(value: Any, indent: str, unit: str) -> str:
    lines = json.dumps(value, ensure_ascii=False, indent=unit).split("\n")
    return "\n".join([lines[0], *(indent + line for line in lines[1:])])


def _insert(text: str, node: Node, key: str, value: Any) -> str:
    unit = _unit(text)
    if node.members:
        last = node.members[-1]
        indent = _indent(text, node.members[0].start)
        multiline = "\n" in text[node.start : node.members[0].start]
    else:
        indent = _indent(text, node.start) + unit
        multiline = True
    member = f"{json.dumps(key, ensure_ascii=False)}: {_dump(value, indent, unit)}"
    if not node.members:
        closing = _indent(text, node.start)
        return f"{text[: node.start + 1]}\n{indent}{member}\n{closing}{text[node.end :]}"
    after = last.comma + 1 if last.comma is not None else last.value_end
    comma = "" if last.comma is not None else ","
    if not multiline:
        return f"{text[:after]}{comma} {member}{text[after:]}"
    # After the rest of the last member's line when only a comment follows it there.
    line_end = text.find("\n", after)
    line_end = len(text) if line_end < 0 else line_end
    rest = text[after:line_end]
    at = line_end if line_end <= node.end and re.fullmatch(r"\s*(//.*|/\*.*\*/\s*)?", rest) else after
    return f"{text[:after]}{comma}{text[after:at]}\n{indent}{member}{text[at:]}"


def set_value(text: str, path: tuple[str, ...], value: Any) -> str:
    """Sets the value at path, adding the key, and any missing objects on the way, at the end of its object."""
    node = parse(text)
    for depth, key in enumerate(path):
        member = node.member(key)
        if member is None:
            nested = value
            for outer in reversed(path[depth + 1 :]):
                nested = {outer: nested}
            changed = _insert(text, node, key, nested)
            break
        if depth == len(path) - 1:
            changed = f"{text[: member.value_start]}{_dump(value, _indent(text, member.start), _unit(text))}{text[member.value_end :]}"
            break
        if member.node is None:
            raise ValueError(f"{'.'.join(path[: depth + 1])} is not an object")
        node = member.node
    _check(changed, path, value)
    return changed


def remove_value(text: str, path: tuple[str, ...]) -> str:
    """Removes the key at path with its comma, and its line when nothing else is on it. A missing key is left."""
    node = parse(text)
    for key in path[:-1]:
        member = node.member(key)
        if member is None or member.node is None:
            return text
        node = member.node
    member = node.member(path[-1])
    if member is None:
        return text
    start = member.start
    end = member.comma + 1 if member.comma is not None else member.value_end
    line_start = _line_start(text, start)
    line_end = text.find("\n", end)
    line_end = len(text) if line_end < 0 else line_end
    alone = not text[line_start:start].strip() and re.fullmatch(r"\s*(//.*)?", text[end:line_end])
    if alone:
        start, end = line_start, min(line_end + 1, len(text))
    if member.comma is None:
        # The last member: the comma before it goes too, unless it is the object's only member.
        before = [item for item in node.members if item is not member and item.start < member.start]
        if before and before[-1].comma is not None:
            comma = before[-1].comma
            changed = text[:comma] + text[comma + 1 : start] + text[end:]
            _check(changed, path, None, removed=True)
            return changed
    changed = text[:start] + text[end:]
    _check(changed, path, None, removed=True)
    return changed


def _check(text: str, path: tuple[str, ...], value: Any, *, removed: bool = False) -> None:
    """The edit must still read, and read back what was written."""
    data: Any = pyjson5.loads(text)
    for key in path[:-1]:
        data = data[key]
    if removed:
        if path[-1] in data:
            raise ValueError(f"{'.'.join(path)} was not removed")
    elif data[path[-1]] != value:
        raise ValueError(f"{'.'.join(path)} did not read back as written")
