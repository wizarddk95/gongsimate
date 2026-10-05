"""Freeze retrieval benchmark v2.1 from the immutable v2 judgments.

This module only reads ``multi_company_retrieval_v2``. It writes a sibling
dataset directory and does not call a judge.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from disclosure_agent.chunking import canonical_dumps, sha256_text
from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2_1 import DATASET_ID, PARENT_DATASET_ID, PROMPT_VERSION
from disclosure_agent.evalset_v2_1.qrels import build_unit_qrels
from disclosure_agent.evalset_v2_1.split import (
    DEV_SIZE,
    SPLIT_SEED,
    TEST_SIZE,
    assign_fully_answerable_splits,
)

PARENT_DATASET_SHA256 = (
    "d3635756a57c39d6d4ef8722f5f118348c00bbc034bf5963fb7720d184e6f137"
)
METRIC_POLICY = {
    "headline_population": (
        "answerability=fully_answerable and split=test, with at least one "
        "retrieval unit whose own judged relevance is >= 1"
    ),
    "retrieval_unit": "the object the judge read and the object the retriever returns",
    "source_element": "citation provenance only. Not a relevance row.",
    "unjudged": "absent from qrels. Never stored as relevance 0.",
    "relevance_0": "the unit was read and judged irrelevant",
    "hit": "raw top-k contains a unit with its own judged relevance >= 1",
    "recall_denominator": "unique retrieval units with judged relevance >= 1",
    "recall_macro": "unweighted mean of per-query recall",
    "recall_micro": "sum of retrieved relevant units / sum of relevant units",
    "requirement_recall": (
        "requirements whose own relevant unit appears in the window, "
        "divided by the requirement count. Not by source elements."
    ),
    "all_requirements_covered": "every fully_answerable requirement of the query is covered",
    "mrr": "1 / raw rank of the first unit with its own relevance >= 1, cutoff 20",
    "ndcg_headline": (
        "condensed: skip unjudged units, keep judged relevance 0, "
        "gain 2^relevance-1, each retrieval_unit_id gains at most once, cutoff 10 "
        "on the condensed list"
    ),
    "ndcg_sensitivity_unjudged_as_zero": (
        "raw top-10, unjudged gain 0 at that rank. Same unit-once rule. "
        "Not the headline. Equals condensed NDCG when the raw top-10 is fully judged."
    ),
    "preferred": (
        "diagnostic. A seed retrieval unit that remains judged relevant. "
        "Otherwise one fallback unit. Non-preferred can still be relevant."
    ),
    "duplicate_unit_in_ranking": (
        "first occurrence keeps the gain and the recall credit. "
        "A later copy occupies its rank with gain 0."
    ),
    "shared_source_element": (
        "units that share an element stay separate. "
        "A grade is not copied onto another unit."
    ),
}


def dataset_dir(root: Path) -> Path:
    return root / "evaluation" / "datasets" / DATASET_ID


def parent_dir(root: Path) -> Path:
    return root / "evaluation" / "datasets" / PARENT_DATASET_ID


def freeze_dataset(root: Path) -> dict[str, Any]:
    parent = _load_parent(root)
    questions = _retarget_splits(parent["questions"], parent["requirements"])
    judgments = parent["judgments"]
    qrels, provenance = build_unit_qrels(judgments, questions)
    _reject_element_expansion(qrels, provenance)
    judgments_sha = sha256_text(parent["judgments_text"])
    body = {
        "dataset_id": DATASET_ID,
        "parent_dataset_id": PARENT_DATASET_ID,
        "parent_dataset_sha256": PARENT_DATASET_SHA256,
        "prompt_version": PROMPT_VERSION,
        "split_seed": SPLIT_SEED,
        "test_size": TEST_SIZE,
        "dev_size": DEV_SIZE,
        "label_source": "v2 LLM judgments reused at retrieval-unit granularity",
        "human_review": False,
        "new_judge_calls": 0,
        "judgments_sha256": judgments_sha,
        "metric_policy": METRIC_POLICY,
        "questions": questions,
        "requirements": parent["requirements"],
        "qrels": qrels,
        "provenance": provenance,
    }
    digest = sha256_text(canonical_dumps(body))
    directory = dataset_dir(root)
    if directory.resolve() == parent_dir(root).resolve():
        raise UsageError("v2.1 must not overwrite the v2 dataset")
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory / "questions.json", {"questions": questions})
    _write(directory / "requirements.json", {"requirements": parent["requirements"]})
    _write(
        directory / "qrels.json",
        {
            "schema": "retrieval_unit_qrel_v2_1",
            "note": (
                "One row per query, requirement, and retrieval unit. Unjudged units are absent."
            ),
            "qrels": qrels,
        },
    )
    _write(
        directory / "provenance.json",
        {
            "note": "Source elements locate a citation. They are not relevance grades.",
            "units": provenance,
        },
    )
    shutil.copyfile(parent["judgments_path"], directory / "judgments.jsonl")
    diff = _diff(parent, questions, qrels, judgments, judgments_sha, provenance)
    audit = _split_audit(questions, parent["requirements"])
    frozen = {
        "dataset_id": DATASET_ID,
        "dataset_sha256": digest,
        "parent_dataset_id": PARENT_DATASET_ID,
        "parent_dataset_sha256": PARENT_DATASET_SHA256,
        "prompt_version": PROMPT_VERSION,
        "split_seed": SPLIT_SEED,
        "test_size": TEST_SIZE,
        "dev_size": DEV_SIZE,
        "label_source": body["label_source"],
        "human_review": False,
        "new_judge_calls": 0,
        "judgments_reused": len(judgments),
        "judgments_sha256": judgments_sha,
        "question_count": len(questions),
        "qrel_count": len(qrels),
        "provenance_count": len(provenance),
        "metric_policy": METRIC_POLICY,
        "split_rule": (
            "Fully answerable questions are reassigned with seed "
            f"{SPLIT_SEED}. Each company contributes 11 test and 9 dev questions. "
            "Wording and query type match Hamilton targets at 55/100. "
            "Evidence form cannot match its own Hamilton target at the same time, "
            "because those query types do not cross evidence forms. "
            "Section strata are repaired until a further within-company swap would "
            "move wording or query type. Retrieval scores are not an input. "
            "Partial and unanswerable splits stay on the v2 assignment."
        ),
    }
    _write(directory / "freeze.json", frozen)
    _write(directory / "dataset_diff.json", diff)
    _write(directory / "split_audit.json", audit)
    return frozen


def _load_parent(root: Path) -> dict[str, Any]:
    directory = parent_dir(root)
    freeze = _read(directory / "freeze.json")
    if freeze.get("dataset_sha256") != PARENT_DATASET_SHA256:
        raise UsageError("v2 freeze hash changed. v2.1 will not rebuild on a modified parent")
    if freeze.get("dataset_id") != PARENT_DATASET_ID:
        raise UsageError("parent dataset id is not multi_company_retrieval_v2")
    questions = _read(directory / "questions_frozen.json")["questions"]
    requirements = _read(directory / "requirements.json")["requirements"]
    v2_qrels = _read(directory / "qrels.json")["qrels"]
    judgments_path = directory / "judgments.jsonl"
    judgments_text = judgments_path.read_text(encoding="utf-8")
    judgments = [
        json.loads(line) for line in judgments_text.splitlines() if line.strip()
    ]
    return {
        "questions": questions,
        "requirements": requirements,
        "judgments": judgments,
        "judgments_text": judgments_text,
        "judgments_path": judgments_path,
        "qrel_count": int(freeze["qrel_count"]),
        "v2_qrels": v2_qrels,
    }


def _retarget_splits(
    questions: list[dict[str, Any]], requirements: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    fully = [row for row in questions if row.get("answerability") == "fully_answerable"]
    assigned = assign_fully_answerable_splits(fully, requirements)
    retargeted = []
    for question in questions:
        copied = dict(question)
        if copied.get("answerability") == "fully_answerable":
            copied["split"] = assigned[str(copied["query_id"])]
        retargeted.append(copied)
    return retargeted


def _diff(
    parent: dict[str, Any],
    questions: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    judgments_sha: str,
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    before = {row["query_id"]: row for row in parent["questions"]}
    text_changed = 0
    added = 0
    removed = 0
    split_changed = []
    for question in questions:
        previous = before.get(question["query_id"])
        if previous is None:
            added += 1
            continue
        if previous["question"] != question["question"]:
            text_changed += 1
        if previous.get("split") != question.get("split"):
            split_changed.append(question["query_id"])
    removed = len(set(before) - {row["query_id"] for row in questions})
    counts = _positive_counts(
        parent["questions"], questions, parent["v2_qrels"], qrels, provenance
    )
    v2_test = {
        row["query_id"]
        for row in parent["questions"]
        if row.get("answerability") == "fully_answerable" and row.get("split") == "test"
    }
    v21_test = {
        row["query_id"]
        for row in questions
        if row.get("answerability") == "fully_answerable" and row.get("split") == "test"
    }
    return {
        "from_dataset_id": PARENT_DATASET_ID,
        "to_dataset_id": DATASET_ID,
        "from_dataset_sha256": PARENT_DATASET_SHA256,
        "question_text_changed": text_changed,
        "questions_added": added,
        "questions_removed": removed,
        "split_assignments_changed": len(split_changed),
        "split_changed_query_ids": split_changed,
        "partial_and_unanswerable_splits": "unchanged",
        "qrel_granularity_before": "requirement x source element, grade copied from the unit",
        "qrel_granularity_after": "requirement x retrieval unit",
        "v2_qrel_rows": parent["qrel_count"],
        "v2_1_qrel_rows": len(qrels),
        "v2_test_positive_elements": counts["v2_test"]["positive_elements_v2_copy"],
        "v2_test_positive_units": counts["v2_test"]["positive_units"],
        "v2_1_test_positive_units": counts["v21_test"]["positive_units"],
        "v2_test_ids": sorted(v2_test),
        "v2_1_test_ids": sorted(v21_test),
        "judgments_reused": len(judgments),
        "new_judgments": 0,
        "llm_api_calls": 0,
        "judgments_sha256": judgments_sha,
        "metric_semantics_changed": [
            "recall denominator is relevant retrieval units, not copied source elements",
            "a unit is judged only by its own qrel, not by a shared source element",
            "NDCG gain is awarded once per retrieval unit",
            "condensed and unjudged-as-zero NDCG match when raw top-k is fully judged",
            "requirement coverage matches retrieval_unit_id, not element_id",
            "preferred is a seed retrieval unit that remains relevant",
        ],
        "denominator_comparison": counts,
    }


def _positive_counts(
    questions_v2: list[dict[str, Any]],
    questions_v21: list[dict[str, Any]],
    qrels_v2: list[dict[str, Any]],
    qrels_v21: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    groups = {
        "v2_test": _ids(questions_v2, "test"),
        "v2_dev": _ids(questions_v2, "dev"),
        "v21_test": _ids(questions_v21, "test"),
        "v21_dev": _ids(questions_v21, "dev"),
        "fully_answerable": _ids(questions_v2, None),
    }
    report = {}
    for name, ids in groups.items():
        report[name] = {
            "positive_elements_v2_copy": _positive_elements(qrels_v2, ids),
            "positive_units": _positive_units(qrels_v21, ids),
            "positive_units_sharing_an_element": _shared_positive_elements(
                qrels_v21, provenance, ids
            ),
        }
    return report


def _ids(questions: list[dict[str, Any]], split: str | None) -> set[str]:
    return {
        row["query_id"]
        for row in questions
        if row.get("answerability") == "fully_answerable"
        and (split is None or row.get("split") == split)
    }


def _positive_elements(qrels: list[dict[str, Any]], ids: set[str]) -> int:
    found = {
        (row["query_id"], row["element_id"])
        for row in qrels
        if row["query_id"] in ids and int(row["relevance"]) >= 1
    }
    return len(found)


def _positive_units(qrels: list[dict[str, Any]], ids: set[str]) -> int:
    found = {
        (row["query_id"], row["retrieval_unit_id"])
        for row in qrels
        if row["query_id"] in ids and int(row["relevance"]) >= 1
    }
    return len(found)


def _shared_positive_elements(
    qrels: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    ids: set[str],
) -> int:
    """Positive units that contain an element also contained by another positive unit.

    Those units stay separate in the denominator. This count is the overlap, not a merge.
    """
    positive = {
        (row["query_id"], row["retrieval_unit_id"])
        for row in qrels
        if row["query_id"] in ids and int(row["relevance"]) >= 1
    }
    element_units: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in provenance:
        if row["query_id"] not in ids:
            continue
        if (row["query_id"], row["retrieval_unit_id"]) not in positive:
            continue
        for element_id in row["source_element_ids"]:
            element_units[(row["query_id"], element_id)].add(row["retrieval_unit_id"])
    shared = {
        (query_id, unit_id)
        for (query_id, _), units in element_units.items()
        if len(units) > 1
        for unit_id in units
    }
    return len(shared)


def _reject_element_expansion(
    qrels: list[dict[str, Any]], provenance: list[dict[str, Any]]
) -> None:
    keys = {(row["query_id"], row["requirement_id"], row["retrieval_unit_id"]) for row in qrels}
    if len(keys) != len(qrels):
        raise UsageError("duplicate retrieval-unit qrel")
    if any("element_id" in row for row in qrels):
        raise UsageError("a v2.1 qrel must not be keyed by element_id")
    by_unit = {(row["query_id"], row["retrieval_unit_id"]): row for row in provenance}
    for row in qrels:
        if (row["query_id"], row["retrieval_unit_id"]) not in by_unit:
            raise UsageError("qrel is missing source provenance")


def _split_audit(
    questions: list[dict[str, Any]], requirements: list[dict[str, Any]]
) -> dict[str, Any]:
    multi = _multi_ids(requirements)
    fully = [row for row in questions if row.get("answerability") == "fully_answerable"]
    fields = (
        "company_name",
        "query_type",
        "evidence_form",
        "wording_style",
        "period_sensitive",
        "stratum_id",
    )
    report: dict[str, Any] = {}
    for name, group in (
        ("all", fully),
        ("dev", [row for row in fully if row["split"] == "dev"]),
        ("test", [row for row in fully if row["split"] == "test"]),
    ):
        block: dict[str, Any] = {"count": len(group)}
        for field in fields:
            block[field] = dict(Counter(str(row.get(field)) for row in group))
        block["multi_evidence"] = {
            "true": sum(1 for row in group if row["query_id"] in multi),
            "false": sum(1 for row in group if row["query_id"] not in multi),
        }
        block["difficulty_tags"] = dict(
            Counter(
                ",".join(row.get("difficulty_tags") or []) or "(none)" for row in group
            )
        )
        report[name] = block
    report["note"] = (
        "difficulty_tags are generator annotations, not a sampling quota. "
        "stratum_id is the document/section category."
    )
    return report


def _multi_ids(requirements: list[dict[str, Any]]) -> set[str]:
    counts: Counter[str] = Counter()
    for row in requirements:
        if row.get("answerability") == "fully_answerable":
            counts[str(row["query_id"])] += 1
    return {query_id for query_id, count in counts.items() if count > 1}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
