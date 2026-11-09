"""Stands in for weclone/cli.py in web build tests: takes the build-profile arguments, writes a one-fact profile
under the real lock and status file, and calls no model. FAKE_BUILD says how it ends: succeed, fail or wait (until
SIGTERM)."""

import os
import sys
import time
from pathlib import Path

from weclone.data.agent import profile_pipeline as pipeline
from weclone.utils import log, secure_storage  # noqa: F401  # log sets up the terminal, as for weclone/cli.py

args = sys.argv[1:]
output = Path(args[args.index("--output-dir") + 1])
purpose = args[args.index("--purpose") + 1]
pipeline.RUNS_DIR = output.parent
secure_storage.configure()
lock = pipeline.build_lock()
lock.acquire(timeout=5)
try:
    build = pipeline.Build(output, purpose)
    with build.running():
        with build.stage("snapshot"):
            build.status["chats"] = 1
        with build.stage("distill_profile"):
            build.progress(0, 2)
            if os.environ["FAKE_BUILD"] == "fail":
                print("printed private text", flush=True)
                log.logger.info(
                    "{name} {account}: windows={counts[windows]}",
                    name="private name",
                    account=987654321,
                    counts={"windows": 7},
                )
                raise RuntimeError("fake failure with private text")
            while os.environ["FAKE_BUILD"] == "wait":
                time.sleep(0.05)
            build.progress(2, 2)
        with build.stage("hierarchy"):
            value = os.environ.get("FAKE_VALUE", "新画像")
            secure_storage.write_json(
                output / "profile_hierarchy" / "profile_hierarchy.json",
                {
                    "domains": [],
                    "dimensions": [
                        {
                            "dim": 8,
                            "name": "偏好",
                            "groups": [{"name": "阅读", "items": [{"attr": "偏好", "fact_indices": [0]}]}],
                        }
                    ],
                    "facts": [{"dim": 8, "attr": "偏好", "value": value, "source_ids": []}],
                    "sources": {},
                },
            )
finally:
    lock.release()
