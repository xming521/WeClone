#!/usr/bin/env python3
"""Apply small Windows-compatibility patches to the copied WeClone tree.

Why this exists
---------------
WeClone hardcodes ``n_process=24`` for Presidio/spaCy batch PII detection
(``weclone/core/PII/pii_detector.py``).  On Windows spaCy starts one process per
worker with ``spawn`` and pickles the language model into each child, which ends
in ``MemoryError`` / ``EOFError: Ran out of input`` before any message is
processed.  A single process is slower but reliable.

Idempotent: running it twice changes nothing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PATCHES = [
    (
        "weclone/core/PII/pii_detector.py",
        re.compile(r"n_process\s*=\s*\d+"),
        "n_process=1",
        "spaCy multiprocessing crashes with MemoryError on Windows",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("app"), help="path to the copied WeClone tree")
    args = parser.parse_args()

    root: Path = args.root
    if not (root / "weclone").is_dir():
        print(f"skip: {root} does not look like a WeClone tree")
        return 1

    changed = 0
    for relative, pattern, replacement, why in PATCHES:
        path = root / relative
        if not path.is_file():
            print(f"skip  {relative}: not found")
            continue
        text = path.read_text(encoding="utf-8")
        updated = pattern.sub(replacement, text)
        if updated == text:
            print(f"ok    {relative}: already patched or not applicable")
            continue
        path.write_text(updated, encoding="utf-8")
        print(f"patch {relative}: {why}")
        changed += 1

    print(f"done, {changed} file(s) changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
