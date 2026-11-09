"""Write weclone/core/inference/model_prices.json from LiteLLM's community price table. Run before a release.

python scripts/update_model_prices.py                  # the table on LiteLLM's main branch
python scripts/update_model_prices.py --repo ../litellm # the table of a LiteLLM checkout
"""

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

TABLE = "model_prices_and_context_window.json"
REPO = "BerriAI/litellm"
OUTPUT = Path(__file__).resolve().parents[1] / "weclone/core/inference/model_prices.json"
FIELDS = {
    "input": "input_cost_per_token",
    "output": "output_cost_per_token",
    "cache_read": "cache_read_input_token_cost",
}


def fetch(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "weclone-update-model-prices"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def load(repo: Path | None) -> tuple[dict, str, str]:
    """The table, its commit and the commit's date."""
    if repo is not None:
        commit, date = subprocess.run(
            ["git", "-C", str(repo), "log", "-1", "--format=%H %cs", "--", TABLE],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        return json.loads((repo / TABLE).read_text(encoding="utf-8")), commit, date
    latest = fetch(f"https://api.github.com/repos/{REPO}/commits?path={TABLE}&per_page=1")[0]
    commit = latest["sha"]
    return (
        fetch(f"https://raw.githubusercontent.com/{REPO}/{commit}/{TABLE}"),
        commit,
        latest["commit"]["committer"]["date"][:10],
    )


def prices(entry: dict) -> dict:
    return {name: entry[key] for name, key in FIELDS.items() if isinstance(entry.get(key), (int, float))}


def snapshot(table: dict, commit: str, date: str) -> dict:
    """Chat models with an input and an output price, grouped by the model id after its last "/", lowercased, with
    one entry per provider."""
    models: dict[str, dict[str, dict]] = {}
    for key, entry in sorted(table.items()):
        if not isinstance(entry, dict) or entry.get("mode") != "chat":
            continue
        found = prices(entry)
        if "input" not in found or "output" not in found:
            continue
        off_peak = entry.get("off_peak_pricing")
        if isinstance(off_peak, dict) and off_peak.get("windows"):
            found["off_peak"] = {**prices(off_peak), "windows": off_peak["windows"]}
        name = key.lower().rsplit("/", 1)[-1]
        models.setdefault(name, {}).setdefault(str(entry.get("litellm_provider") or ""), found)
    return {
        "source": f"https://github.com/{REPO}/blob/{commit}/{TABLE}",
        "date": date,
        "models": {
            name: [{"provider": provider, **found} for provider, found in sorted(entries.items())]
            for name, entries in sorted(models.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, help="A LiteLLM checkout to read instead of GitHub")
    args = parser.parse_args()
    result = snapshot(*load(args.repo))
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{len(result['models'])} models as of {result['date']} -> {OUTPUT}")


if __name__ == "__main__":
    main()
