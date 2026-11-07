"""Deep insights of a build-profile run: what B's whole profile implies but no fact says, written by the LLM in one
request over every fact of the review store that is not rejected.

Implicit facts are conclusions of the same kind as profile facts. Insights summarise what kind of person B is, at most
insight_limit of them for the number of facts. Each keeps the review ids of the facts it rests on and the strictest
sensitivity among them, unknown when one of them has none. The result goes next to the profile as deep_insights.json.
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from weclone.data.agent.organize_memories import LLMTasks, save
from weclone.prompts.deep_insights import INSIGHTS_SCHEMA, LISTS, insight_limit, insights_prompt
from weclone.prompts.memory_organization import SENSITIVITY_LEVELS
from weclone.server.profile_review import ReviewStore
from weclone.server.profile_summary import clean, domain_members

OUTPUT_NAME = "deep_insights.json"


def profile_input(profile: dict) -> tuple[list[tuple[str, list[str]]], list[dict]]:
    """Each domain's fact lines for the prompt, leaving out rejected facts, and the facts in the order the lines
    number them."""
    groups, numbered = [], []
    for domain, indices in domain_members(profile).items():
        lines = []
        for index in indices:
            fact = profile["facts"][index]
            if fact["status"] == "rejected":
                continue
            # Facts added by hand have no time scope.
            scope = f"（{fact['time_scope']}）" if fact.get("time_scope") else ""
            lines.append(f"{len(numbered)}｜{clean(fact['attr'])}：{clean(fact['value'])}{scope}")
            numbered.append(fact)
        if lines:
            groups.append((clean(domain), lines))
    return groups, numbered


def validate_insights(payload, count: int, limit: int) -> dict:
    if not isinstance(payload, dict) or set(payload) != set(LISTS):
        raise ValueError("Insights require implicit_facts and insights")
    result = {}
    for name in LISTS:
        if not isinstance(payload[name], list):
            raise ValueError(f"{name} must be an array")
        rows = []
        for item in payload[name]:
            if not isinstance(item, dict) or set(item) != {"conclusion", "basis"}:
                raise ValueError("Each conclusion requires conclusion and basis")
            text = clean(item["conclusion"]) if isinstance(item["conclusion"], str) else ""
            basis = item["basis"]
            if not text or not isinstance(basis, list) or not basis:
                raise ValueError("Each conclusion needs text and the facts it rests on")
            if any(type(i) is not int or not 0 <= i < count for i in basis):
                raise ValueError("Basis must be fact numbers from the prompt")
            rows.append({"text": text, "basis": sorted(set(basis))})
        result[name] = rows
    # A model writing past the cap keeps the insights resting on the most facts, in the order it wrote them.
    kept = sorted(range(len(result["insights"])), key=lambda i: -len(result["insights"][i]["basis"]))[:limit]
    result["insights"] = [result["insights"][i] for i in sorted(kept)]
    return result


def strictest(facts: list[dict]) -> str | None:
    levels = [fact.get("sensitivity") for fact in facts]
    if any(level not in SENSITIVITY_LEVELS for level in levels):
        return None
    return max(levels, key=SENSITIVITY_LEVELS.index)


def insights_runner(max_context_tokens: int | None) -> LLMTasks:
    """build-profile's LLM task runner with the configured model. Without a token limit it takes the Codex model's
    context length."""
    return LLMTasks(
        argparse.Namespace(
            config_path=Path(os.environ.get("WECLONE_CONFIG_PATH", "settings.jsonc")),
            llm_provider=None,
            max_context_tokens=max_context_tokens,
            token_encoding="o200k_base",
            tokenizer_file=None,
        )
    )


def build_insights(directory: Path, max_context_tokens: int | None = None) -> dict:
    """Write deep_insights.json for the profile in `directory`, a run's profile_hierarchy directory, and return it."""
    store = ReviewStore(directory / "profile_review.sqlite3", directory / "profile_hierarchy.json")
    groups, numbered = profile_input(store.profile())
    if not numbered:
        raise ValueError("画像里没有可用的事实")
    limit = insight_limit(len(numbered))
    task = {"stage": "deep_insights", "schema": INSIGHTS_SCHEMA, "prompt": insights_prompt(groups, limit)}
    runner = insights_runner(max_context_tokens)
    try:
        [answer] = runner.run([task], lambda payload, _: validate_insights(payload, len(numbered), limit))
    finally:
        runner.close()

    def resolve(row: dict) -> dict:
        basis = [numbered[i] for i in row["basis"]]
        return {
            "text": row["text"],
            "fact_ids": [fact["id"] for fact in basis],
            "sensitivity": strictest(basis),
        }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **{name: [resolve(row) for row in answer[name]] for name in LISTS},
    }
    save(directory / OUTPUT_NAME, result)
    return result
