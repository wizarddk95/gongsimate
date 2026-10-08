"""python -m disclosure_agent.agent_a0"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from disclosure_agent.agent_a0.abstention_contract import write_correction
from disclosure_agent.agent_a0.dataset import write_dataset
from disclosure_agent.agent_a0.run import run_baseline
from disclosure_agent.evalset_v2.spec import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m disclosure_agent.agent_a0")
    parser.add_argument("stage", choices=["prepare", "run", "correct"])
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    root = args.root or repo_root()
    if args.stage == "prepare":
        print(json.dumps(write_dataset(root), ensure_ascii=False))
        return
    if args.stage == "correct":
        print(json.dumps(write_correction(root), ensure_ascii=False))
        return
    summary = run_baseline(root)
    print(
        json.dumps(
            {
                "dataset_sha256": summary["dataset_sha256"],
                "task": summary["task"],
                "search_decision": summary["search_decision"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
