"""Freeze v2.1, rescore Dense-D2a, compare it with BM25, or fuse them on dev.

Neither freeze nor evaluate calls a judge. Complementarity and RRF search
fully answerable dev only. RRF does not sweep k and does not read test.
Truncated RRF reuses stored DEV top-20 rankings and does not search test.
The H9 diagnosis reuses that truncated ranking and does not search test.
H10 reuses the stored DEV top-20 lists and does not search test.
The test-validation stage searches fully_answerable TEST once, then rescores
the locked rankings without a second retrieval.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from disclosure_agent.evalset_v2.spec import repo_root
from disclosure_agent.evalset_v2_1.asymmetric_fusion import run_asymmetric_fusion_experiment
from disclosure_agent.evalset_v2_1.complementarity import run_complementarity
from disclosure_agent.evalset_v2_1.evaluate import evaluate_dense
from disclosure_agent.evalset_v2_1.freeze import freeze_dataset
from disclosure_agent.evalset_v2_1.rrf import run_rrf_experiment
from disclosure_agent.evalset_v2_1.test_validation import run_test_validation
from disclosure_agent.evalset_v2_1.truncated_rrf import run_truncated_rrf_experiment
from disclosure_agent.evalset_v2_1.truncated_rrf_diagnosis import (
    build_diagnosis,
    write_diagnosis,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m disclosure_agent.evalset_v2_1")
    parser.add_argument(
        "stage",
        choices=[
            "freeze",
            "evaluate",
            "run",
            "complementarity",
            "rrf",
            "truncated-rrf",
            "truncated-rrf-diagnosis",
            "asymmetric-fusion",
            "test-validation",
        ],
    )
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
    if args.stage == "rrf":
        payload = run_rrf_experiment(root)
        print(
            json.dumps(
                {
                    "dataset_sha256": payload["dataset_sha256"],
                    "question_count": payload["question_count"],
                    "verdict": payload["verdict"]["label"],
                    "rrf_hit10": payload["rrf_headline"]["at_k"]["10"]["hit"],
                    "test_questions_searched": payload["test_questions_searched"],
                },
                ensure_ascii=False,
            )
        )
        return
    if args.stage == "truncated-rrf":
        payload = run_truncated_rrf_experiment(root)
        print(
            json.dumps(
                {
                    "dataset_sha256": payload["dataset_sha256"],
                    "question_count": payload["question_count"],
                    "verdict": payload["verdict"]["label"],
                    "truncated_hit10": payload["truncated_headline"]["at_k"]["10"]["hit"],
                    "recovered_of_eight": payload["recovery"]["recovered_hit10"],
                    "test_questions_searched": payload["test_questions_searched"],
                },
                ensure_ascii=False,
            )
        )
        return
    if args.stage == "truncated-rrf-diagnosis":
        payload = build_diagnosis(root)
        write_diagnosis(root, payload)
        print(
            json.dumps(
                {
                    "dataset_sha256": payload["dataset_sha256"],
                    "question_count": payload["question_count"],
                    "verdict": payload["verdict"]["decision"],
                    "mechanisms_identical": payload["verdict"]["mechanisms_identical"],
                    "test_questions_searched": payload["test_questions_searched"],
                },
                ensure_ascii=False,
            )
        )
        return
    if args.stage == "test-validation":
        payload = run_test_validation(root)
        print(
            json.dumps(
                {
                    "dataset_sha256": payload["dataset_sha256"],
                    "question_count": payload["question_count"],
                    "retrieval_execution_count": payload["retrieval_execution_count"],
                    "verdict": payload["verdict"]["label"],
                    "reason": payload["verdict"]["reason"],
                    "selected_configuration": payload["decision"]["selected_configuration"],
                    "h10_hit10": payload["h10_headline"]["at_k"]["10"]["hit"],
                    "bm25_hit10": payload["bm25_headline"]["at_k"]["10"]["hit"],
                },
                ensure_ascii=False,
            )
        )
        return
    if args.stage == "asymmetric-fusion":
        payload = run_asymmetric_fusion_experiment(root)
        print(
            json.dumps(
                {
                    "dataset_sha256": payload["dataset_sha256"],
                    "question_count": payload["question_count"],
                    "verdict": payload["verdict"]["label"],
                    "reason": payload["verdict"]["reason"],
                    "h10_hit10": payload["h10_headline"]["at_k"]["10"]["hit"],
                    "bm25_only_lost": payload["regressions"]["bm25_only_lost"],
                    "dense_only_hit10": payload["dense_only_retention"]["hit10"],
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
