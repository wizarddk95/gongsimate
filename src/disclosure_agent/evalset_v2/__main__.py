"""Run one stage of the v2 retrieval evaluation set."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from disclosure_agent.evalset_v2.pipeline import estimate_stage, run_all, sample_stage
from disclosure_agent.evalset_v2.spec import repo_root


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m disclosure_agent.evalset_v2")
    parser.add_argument("stage", choices=["sample", "estimate", "run"])
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.root or repo_root()
    if args.stage == "sample":
        manifest = sample_stage(root)
        print(json.dumps(manifest["realized_splits"], ensure_ascii=False))
        return
    if args.stage == "estimate":
        print(json.dumps(estimate_stage(root), ensure_ascii=False, indent=2))
        return
    result = run_all(root)
    print(json.dumps({"dataset_sha256": result["freeze"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
