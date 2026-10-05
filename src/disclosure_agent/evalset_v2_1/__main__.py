"""Freeze v2.1, rescore Dense-D2a, or compare it with BM25 on dev.

Neither freeze nor evaluate calls a judge. The complementarity stage searches
fully answerable dev only and does not retune BM25.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from disclosure_agent.evalset_v2.spec import repo_root
from disclosure_agent.evalset_v2_1.complementarity import run_complementarity
from disclosure_agent.evalset_v2_1.evaluate import evaluate_dense
from disclosure_agent.evalset_v2_1.freeze import freeze_dataset


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m disclosure_agent.evalset_v2_1")
    parser.add_argument("stage", choices=["freeze", "evaluate", "run", "complementarity"])
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
    if args.stage == "complementarity":
        payload = run_complementarity(root)
        print(
            json.dumps(
                {
                    "dataset_sha256": payload["dataset_sha256"],
                    "question_count": payload["question_count"],
                    "query_cells": payload["query_cells"],
                    "requirement_cells": payload["requirement_cells"],
                    "hybrid_decision": payload["hybrid_decision"]["primary"],
                    "test_questions_searched": payload["test_questions_searched"],
                },
                ensure_ascii=False,
            )
        )
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
