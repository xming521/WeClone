"""B's profile tree as the harness reads it: build-profile's profile_hierarchy.json, domains → topics → facts.

When the review DB (profile_review.sqlite3) exists next to the profile, review decisions are applied:
rejected facts are dropped and edited attributes and values are used.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from weclone.server.profile_review import ReviewStore, review_paths
from weclone.utils import secure_storage

PROFILE_RUNS = Path("dataset/res_csv/agent/profile_runs")
TOPIC_TEXT_CHARS = 80


def latest_profile(runs: Path = PROFILE_RUNS) -> Path:
    for directory in sorted(runs.glob("*/profile_hierarchy"), reverse=True):
        path = directory / "profile_hierarchy.json"
        if secure_storage.file_exists(path):
            return path
    raise FileNotFoundError(
        f"{runs} 下没有画像，请先运行 build-profile，"
        "或在 harness_args.profile_context.profile_path 指定 profile_hierarchy.json"
    )


def local_time(value: datetime) -> datetime:
    """Naive local time, the form chat sample times are stored in."""
    return value.astimezone().replace(tzinfo=None) if value.tzinfo else value


def reviewed_profile(review: dict) -> dict:
    """Review store output as the domains → topics → facts shape ProfileTree reads: rejected facts dropped,
    edited attr and value used. Attributes without a topic (direct on a domain or a dimension) cannot be routed."""
    facts: list[dict] = []
    topics: dict[tuple[str, str], list[int]] = {}
    for dimension in review["dimensions"]:
        for group in dimension["groups"]:
            if group["id"].endswith(":direct"):
                continue
            for topic in group["items"]:
                if "attr" in topic:
                    continue
                indices = topics.setdefault((group["name"], topic["name"]), [])
                for attribute in topic["items"]:
                    for index in attribute["fact_indices"]:
                        fact = review["facts"][index]
                        if fact["status"] != "rejected":
                            indices.append(len(facts))
                            facts.append({**fact, "attr": attribute["attr"]})
    domains = {
        domain["name"]: {"name": domain["name"], "description": domain["description"], "topics": []}
        for domain in review["domains"]
    }
    order = {
        (domain["name"], name): rank
        for rank, (domain, name) in enumerate(
            (domain, name) for domain in review["domains"] for name in domain["topics"]
        )
    }
    for (domain, topic), indices in sorted(topics.items(), key=lambda item: order.get(item[0], len(order))):
        if indices:
            domains.setdefault(domain, {"name": domain, "description": "", "topics": []})["topics"].append(
                {"name": topic, "fact_indices": indices}
            )
    return {
        "facts": facts,
        "sources": review.get("sources", {}),
        "domains": [domain for domain in domains.values() if domain["topics"]],
    }


@dataclass(frozen=True)
class Topic:
    key: str  # "<domain index>|<topic index>"
    name: str
    text: str  # topic name and attribute names: all the routers see of it
    facts: tuple[int, ...]


class ProfileTree:
    def __init__(self, data: dict, path: Path | None = None):
        domains = data["domains"]
        self.path = path
        self.facts = data["facts"]
        sources = data.get("sources", {})
        # When each fact came up in chat. A fact without source times is always visible and shown undated.
        self.seen = [
            sorted(
                local_time(datetime.fromisoformat(sources[sid]["sample_time"]))
                for sid in fact.get("source_ids", [])
                if sources.get(sid, {}).get("sample_time")
            )
            for fact in self.facts
        ]
        self.domains = [(domain["name"], domain["description"]) for domain in domains]
        self.topics: dict[str, Topic] = {}
        for d, domain in enumerate(domains):
            for t, topic in enumerate(domain["topics"]):
                key = f"{d}|{t}"
                attrs = dict.fromkeys(self.facts[k]["attr"] for k in topic["fact_indices"])
                text = f"{topic['name']}：{'、'.join(attrs)}"[:TOPIC_TEXT_CHARS]
                self.topics[key] = Topic(key, topic["name"], text, tuple(topic["fact_indices"]))

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ProfileTree":
        path = Path(path) if path else latest_profile()
        database, _ = review_paths(source=path)
        if secure_storage.file_exists(database):
            return cls(reviewed_profile(ReviewStore(database, path).profile()), path)
        return cls(secure_storage.read_json(path), path)

    def domain_topics(self, d: int) -> list[Topic]:
        return [topic for key, topic in self.topics.items() if key.split("|")[0] == str(d)]

    def visible(self, k: int, now: datetime) -> bool:
        """Facts that first came up after `now` are hidden, so replaying an old chat cannot see its future."""
        return not self.seen[k] or self.seen[k][0] < now

    def fact_text(self, k: int, now: datetime) -> str:
        """attr：value, then the months in which the fact came up before now, e.g. （2024-03~2024-07）."""
        fact = self.facts[k]
        text = f"{fact['attr']}：{fact['value']}"
        seen = [t for t in self.seen[k] if t < now]
        if not seen:
            return text
        first, last = seen[0].strftime("%Y-%m"), seen[-1].strftime("%Y-%m")
        return f"{text}（{first if first == last else f'{first}~{last}'}）"

    def tree_text(self) -> str:
        """Domains and topics without facts, the router prompt's fixed part."""
        lines = []
        for d, (name, description) in enumerate(self.domains):
            lines.append(f"## {name}：{description}")
            lines += [f"{topic.key} {topic.text}" for topic in self.domain_topics(d)]
        return "\n".join(lines)

    def fill(self, ranking: list[str], now: datetime, budget: int) -> list[int]:
        """Visible facts of the top-ranked topics; a topic that would push the total over the budget is skipped."""
        chosen: set[int] = set()
        for key in ranking:
            facts = {k for k in self.topics[key].facts if self.visible(k, now)}
            if len(chosen | facts) <= budget:
                chosen |= facts
        return sorted(chosen)
