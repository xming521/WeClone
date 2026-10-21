#!/usr/bin/env python3
"""Make LangBot's docker-compose.yaml work on Windows.

The upstream file mounts the Box data directory as

    - ${LANGBOT_BOX_ROOT:-${PWD}/data/box}:${LANGBOT_BOX_ROOT:-${PWD}/data/box}

On Windows ``${PWD}`` expands to ``C:\\...\\docker``; Docker then splits the
volume spec on the drive colon and fails with::

    mount denied: ... too many colons

The Box sandbox is optional (and a Windows host cannot run it well anyway), so
this script rewrites that mount to a plain relative path and leaves everything
else untouched. Idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

OLD = "${LANGBOT_BOX_ROOT:-${PWD}/data/box}:${LANGBOT_BOX_ROOT:-${PWD}/data/box}"
NEW = "./data/box:/app/data/box"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: fix_langbot_compose.py <docker-compose.yaml>")
        return 2
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"not found: {path}")
        return 1

    text = path.read_text(encoding="utf-8")
    if OLD not in text:
        if NEW in text:
            print("already fixed")
            return 0
        print("pattern not found; upstream compose changed, inspect it manually")
        return 1

    fixed = text.replace(OLD, NEW)
    backup = path.with_suffix(".yaml.bak")
    backup.write_text(text, encoding="utf-8")
    path.write_text(fixed, encoding="utf-8")
    print(f"fixed {path}")
    print(f"backup {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
