from __future__ import annotations

import argparse
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_FILE = PACKAGE_ROOT / "resources" / "tokenizers" / "deepseek_v3" / "tokenizer.json"
TOKENIZER_FILE_ENV = "WECLONE_TOKENIZER_FILE"


def resolve_tokenizer_file(tokenizer_file: str | Path | None = None) -> Path:
    raw_path = tokenizer_file or os.environ.get(TOKENIZER_FILE_ENV) or DEFAULT_TOKENIZER_FILE
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Tokenizer file not found: {path}. "
            f"Pass tokenizer_file or set {TOKENIZER_FILE_ENV} to a tokenizer.json path."
        )
    return path


@lru_cache(maxsize=8)
def _load_tokenizer(tokenizer_file: str):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency `tokenizers`. Activate the WeClone environment first."
        ) from exc

    return Tokenizer.from_file(tokenizer_file)


def encode_text(text: Any, tokenizer_file: str | Path | None = None) -> list[int]:
    tokenizer_path = resolve_tokenizer_file(tokenizer_file)
    tokenizer = _load_tokenizer(str(tokenizer_path))
    value = "" if text is None else str(text)
    return tokenizer.encode(value).ids


def count_tokens(text: Any, tokenizer_file: str | Path | None = None) -> int:
    return len(encode_text(text, tokenizer_file=tokenizer_file))


def count_token_batch(texts: Iterable[Any], tokenizer_file: str | Path | None = None) -> list[int]:
    tokenizer_path = resolve_tokenizer_file(tokenizer_file)
    tokenizer = _load_tokenizer(str(tokenizer_path))
    values = ["" if text is None else str(text) for text in texts]
    return [len(encoded.ids) for encoded in tokenizer.encode_batch(values)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count text tokens with a local tokenizer.json file.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Text to count.")
    input_group.add_argument("--text-file", type=Path, help="UTF-8 text file to count.")
    parser.add_argument(
        "--tokenizer-file",
        type=Path,
        default=None,
        help=f"tokenizer.json path. Defaults to {DEFAULT_TOKENIZER_FILE}.",
    )
    parser.add_argument("--show-ids", action="store_true", help="Print token ids after the count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    ids = encode_text(text, tokenizer_file=args.tokenizer_file)
    print(len(ids))
    if args.show_ids:
        print(ids)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
