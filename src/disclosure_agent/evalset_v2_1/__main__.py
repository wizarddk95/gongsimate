"""Freeze v2.1 and, separately, rescore Dense-D2a. Neither step calls a judge."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from disclosure_agent.evalset_v2.spec import repo_root
from disclosure_agent.evalset_v2_1.evaluate import evaluate_dense
from disclosure_agent.evalset_v2_1.freeze import freeze_dataset


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m disclosure_agent.evalset_v2_1")
    parser.add_argument("stage", choices=["freeze", "evaluate", "run"])
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.root or repo_root()
    if args.stage == "freeze":
        frozen = freeze_dataset(root)
        print(json.dumps({"dataset_sha256": frozen["dataset_sha256"]}, ensure_ascii=False))
        return
    if args.stage == "evaluate":
        payload = evaluate_dense(root)
        print(json.dumps({"dataset_sha256": payload["dataset_sha256"]}, ensure_ascii=False))
        return
    frozen = freeze_dataset(root)
    payload = evaluate_dense(root)
    print(
        json.dumps(
            {
                "dataset_sha256": frozen["dataset_sha256"],
                "run_dataset_sha256": payload["dataset_sha256"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
