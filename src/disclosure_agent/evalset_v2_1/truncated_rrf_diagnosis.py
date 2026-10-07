"""Cause attribution for the three residual truncated-RRF BM25 regressions.

The ranker is the existing candidate-truncated RRF. N, k, and the 1:1 weights
stay fixed. Counterfactual ranks delete units from that order on paper. They
are not new retrieval configurations, and they are not scored as systems.
v2.1 TEST is not read.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2_1 import DATASET_ID
from disclosure_agent.evalset_v2_1.freeze import dataset_dir
from disclosure_agent.evalset_v2_1.rrf import HIT_K, movement_label, validate_source_bundle
from disclosure_agent.evalset_v2_1.truncated_rrf import (
    CANDIDATE_N,
    candidate_truncated_rrf,
    intersection_min_score,
    single_list_max_score,
)
from disclosure_agent.evalset_v2_1.truncated_rrf import (
    RUN_DIR_NAME as TRUNCATED_RUN_DIR,
)
from disclosure_agent.evaluation import git_revision
from disclosure_agent.hybrid import RRF_K

RUN_DIR_NAME = "multi_company_retrieval_v2_1_dev_h9_diagnosis"
WORKLOG_NAME = "2026-10-07_v21-dev-h9-rank-pressure.md"
CATEGORIES = ("intersection", "dense_only", "bm25_only")
MECHANISM_BOTH = "intersection_dominance_and_symmetric_competition"
MECHANISM_DOMINANCE = "intersection_dominance"
MECHANISM_SYMMETRIC = "symmetric_rank_competition"
MECHANISM_UNEXPLAINED = "unexplained"
MECHANISM_INTERSECTION_POSITIVE = "positive_is_intersection"
ASSOCIATION_NOTE = (
    "이 계수는 이 DEV 조각의 기술 통계다. 표본이 작고 질의가 한 벤치마크에 묶여 있다. "
    "인과 추정치로 읽지 않는다."
)
RESIDUALS = (
    {
        "query_id": "V2-00121941-business_paragraph-03",
        "label": "대상 지분증권",
        "bm25_rank": 8,
        "truncated_rank": 15,
        "intersection_count": 0,
    },
    {
        "query_id": "V2-00121941-finance_table-02",
        "label": "대상 특수관계자",
        "bm25_rank": 7,
        "truncated_rank": 14,
        "intersection_count": 8,
    },
    {
        "query_id": "V2-00989619-governance_paragraph-02",
        "label": "알테오젠 우선주",
        "bm25_rank": 4,
        "truncated_rank": 11,
        "intersection_count": 8,
    },
)
QUERY_COLUMNS = (
    "query_id",
    "company",
    "movement_vs_bm25",
    "bm25_positive_rank",
    "bm25_positive_truncated_rank",
    "truncated_first_relevant_rank",
    "same_unit_rank_change",
    "intersection_count",
    "dense_only_count",
    "bm25_only_count",
    "intersection_ahead",
    "dense_only_ahead",
    "bm25_only_ahead",
    "incremental_intersection_pressure",
    "symmetric_competition_pressure",
    "structural_property_holds",
    "bm25_hit10",
    "truncated_hit10",
    "positive_category",
    "mechanism",
)
UNIT_COLUMNS = (
    "query_id",
    "retrieval_unit_id",
    "final_rank",
    "dense_rank",
    "bm25_rank",
    "dense_contribution",
    "bm25_contribution",
    "fused_score",
    "category",
    "relevance",
    "judgment_status",
    "relevant_requirement_ids",
    "is_primary_bm25_positive",
    "ahead_of_primary_bm25_positive",
    "in_fused_top20",
)


def unit_category(*, dense_rank: int | None, bm25_rank: int | None) -> str:
    """Classify one fused row by which top-20 lists contain it."""
    if dense_rank is not None and bm25_rank is not None:
        return "intersection"
    if dense_rank is not None:
        return "dense_only"
    if bm25_rank is not None:
        return "bm25_only"
    raise UsageError("a fused unit is outside both top-20 lists")


def judgments_by_unit(qrels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Max judged grade per unit. Missing units stay out of the map."""
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrels:
        if row.get("judgment_status") not in (None, "judged"):
            continue
        if row.get("relevance") is None:
            continue
        unit_id = str(row["retrieval_unit_id"])
        requirement_id = str(row["requirement_id"])
        relevance = int(row["relevance"])
        previous = grouped[unit_id].get(requirement_id)
        if previous is None or relevance > previous:
            grouped[unit_id][requirement_id] = relevance
    resolved: dict[str, dict[str, Any]] = {}
    for unit_id, requirements in grouped.items():
        judgments = [
            {"requirement_id": requirement_id, "relevance": relevance}
            for requirement_id, relevance in sorted(requirements.items())
        ]
        resolved[unit_id] = {
            "relevance": max(item["relevance"] for item in judgments),
            "requirement_judgments": judgments,
            "relevant_requirement_ids": [
                item["requirement_id"] for item in judgments if item["relevance"] >= 1
            ],
        }
    return resolved


def decompose_fused_ranking(
    fused: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy the truncated ranking and attach list category plus qrel fields."""
    judgments = judgments_by_unit(qrels)
    rows: list[dict[str, Any]] = []
    for hit in fused:
        unit_id = str(hit["retrieval_unit_id"])
        judged = judgments.get(unit_id)
        dense_rank = _optional_rank(hit.get("dense_rank"))
        bm25_rank = _optional_rank(hit.get("bm25_rank"))
        relevance = None if judged is None else int(judged["relevance"])
        rows.append(
            {
                "retrieval_unit_id": unit_id,
                "final_rank": int(hit["rank"]),
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
                "dense_contribution": float(hit["dense_contribution"]),
                "bm25_contribution": float(hit["bm25_contribution"]),
                "fused_score": float(hit["score"]),
                "category": unit_category(dense_rank=dense_rank, bm25_rank=bm25_rank),
                "relevance": relevance,
                "judgment_status": "unjudged" if judged is None else "judged",
                "relevant_requirement_ids": []
                if judged is None
                else list(judged["relevant_requirement_ids"]),
                "requirement_judgments": []
                if judged is None
                else list(judged["requirement_judgments"]),
                "in_fused_top20": int(hit["rank"]) <= CANDIDATE_N,
            }
        )
    return rows


def structural_property(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every intersection unit outranks every single-list unit at N=20, k=60, 1:1."""
    intersection = [row for row in rows if row["category"] == "intersection"]
    single = [row for row in rows if row["category"] != "intersection"]
    worst = min((float(row["fused_score"]) for row in intersection), default=None)
    best = max((float(row["fused_score"]) for row in single), default=None)
    if intersection and single:
        order_holds = worst > best and max(int(row["final_rank"]) for row in intersection) < min(
            int(row["final_rank"]) for row in single
        )
    else:
        order_holds = True
    floor = intersection_min_score()
    ceiling = single_list_max_score()
    floor_ok = worst is None or worst >= floor - 1e-12
    ceiling_ok = best is None or best <= ceiling + 1e-12
    return {
        "holds": bool(order_holds and floor_ok and ceiling_ok),
        "empirical_order_holds": bool(order_holds),
        "scores_within_formula_bounds": bool(floor_ok and ceiling_ok),
        "intersection_count": len(intersection),
        "single_list_count": len(single),
        "dense_only_count": sum(1 for row in rows if row["category"] == "dense_only"),
        "bm25_only_count": sum(1 for row in rows if row["category"] == "bm25_only"),
        "worst_intersection_score": worst,
        "best_single_list_score": best,
        "formula_intersection_floor": floor,
        "formula_single_list_ceiling": ceiling,
        "vacuous_intersection": not intersection,
    }


def ahead_category_counts(rows: list[dict[str, Any]], *, before_rank: int) -> dict[str, int]:
    counts = {category: 0 for category in CATEGORIES}
    for row in rows:
        if int(row["final_rank"]) < int(before_rank):
            counts[str(row["category"])] += 1
    return counts


def intersection_dominance_weight_ratio(bm25_rank: int) -> float:
    """w_bm25/w_dense where a BM25-only rank beats the worst intersection unit.

    This is algebra of k=60 and N=20. No weighted list is scored.
    """
    rank = int(bm25_rank)
    if rank < 1 or rank >= CANDIDATE_N:
        raise UsageError(f"dominance boundary rank {bm25_rank} is outside 1..{CANDIDATE_N - 1}")
    return (RRF_K + rank) / (CANDIDATE_N - rank)


def dense_rank1_weight_ratio(bm25_rank: int) -> float:
    """w_bm25/w_dense where a BM25-only rank beats a Dense-only rank 1.

    Algebra only. No weighted list is scored.
    """
    rank = int(bm25_rank)
    if rank < 1 or rank > CANDIDATE_N:
        raise UsageError(f"dense-rank-1 boundary rank {bm25_rank} is outside 1..{CANDIDATE_N}")
    return (RRF_K + rank) / (RRF_K + 1)


def attribute_rank_pressure(
    rows: list[dict[str, Any]],
    *,
    positive_unit_id: str,
    bm25_rank: int,
) -> dict[str, Any]:
    """Explain one BM25 positive's truncated rank from the units in front of it.

    For a BM25-only positive the identity is: rank change equals intersection
    queue jumps plus Dense-only units ahead. Queue jumps are intersection units
    whose own BM25 rank is worse than the positive. BM25 predecessors stay in
    the predecessor count and are not called displacement.
    """
    positive = _find_row(rows, positive_unit_id)
    if int(positive["bm25_rank"]) != int(bm25_rank):
        raise UsageError(f"{positive_unit_id} BM25 rank does not match the fused row")
    truncated_rank = int(positive["final_rank"])
    ahead = [row for row in rows if int(row["final_rank"]) < truncated_rank]
    expected = [
        row
        for row in rows
        if row["bm25_rank"] is not None and int(row["bm25_rank"]) < int(bm25_rank)
    ]
    expected_ids = {str(row["retrieval_unit_id"]) for row in expected}
    missing = [
        str(row["retrieval_unit_id"])
        for row in expected
        if str(row["retrieval_unit_id"]) not in {str(item["retrieval_unit_id"]) for item in ahead}
    ]
    displacement = [row for row in ahead if str(row["retrieval_unit_id"]) not in expected_ids]
    queue_jumps = [row for row in displacement if row["category"] == "intersection"]
    dense_disp = [row for row in displacement if row["category"] == "dense_only"]
    other = [row for row in displacement if row["category"] == "bm25_only"]
    ahead_counts = ahead_category_counts(rows, before_rank=truncated_rank)
    intersection_count = sum(1 for row in rows if row["category"] == "intersection")
    rank_change = truncated_rank - int(bm25_rank)
    applies = positive["category"] == "bm25_only"
    holds = applies and (
        len(expected) == int(bm25_rank) - 1
        and not missing
        and not other
        and len(queue_jumps) + len(dense_disp) == rank_change
        and ahead_counts["intersection"] == intersection_count
    )
    mechanism = _mechanism(
        applies=applies,
        holds=holds,
        queue_jumps=len(queue_jumps),
        dense_displacement=len(dense_disp),
    )
    minimum_exit = max(0, HIT_K + 1 - int(bm25_rank))
    removals = max(0, truncated_rank - HIT_K)
    rank_without_intersection = truncated_rank - ahead_counts["intersection"]
    rank_without_queue = truncated_rank - len(queue_jumps)
    rank_without_dense = truncated_rank - ahead_counts["dense_only"]
    rank_without_bm25 = truncated_rank - ahead_counts["bm25_only"]
    exited = int(bm25_rank) <= HIT_K and truncated_rank > HIT_K
    displacement_counts = _category_counts(displacement)
    return {
        "positive_unit_id": positive_unit_id,
        "positive_category": positive["category"],
        "bm25_rank": int(bm25_rank),
        "truncated_rank": truncated_rank,
        "same_unit_rank_change": rank_change,
        "ahead_counts": ahead_counts,
        "ahead_judgments": _judgment_counts(ahead),
        "displacement_judgments": _judgment_counts(displacement),
        "predecessor_count": len(expected),
        "predecessor_counts": _category_counts(expected),
        "predecessors_missing_ahead": missing,
        "incremental_intersection_pressure": len(queue_jumps),
        "positional_intersection_pressure": ahead_counts["intersection"],
        "symmetric_competition_pressure": len(dense_disp),
        "displacement_by_category": displacement_counts,
        "displacement_units": [_short_unit(row) for row in displacement],
        "identity_applies": applies,
        "identity_holds": holds,
        "mechanism": mechanism,
        "dense_contribution": float(positive["dense_contribution"]),
        "bm25_contribution": float(positive["bm25_contribution"]),
        "fused_score": float(positive["fused_score"]),
        "counterfactual_a_rank": rank_without_intersection,
        "counterfactual_a_removes_predecessors": sum(
            1 for row in expected if row["category"] == "intersection"
        ),
        "counterfactual_a_queue_jump_only_rank": rank_without_queue,
        "counterfactual_b_dense_only_ahead": ahead_counts["dense_only"],
        "counterfactual_c_bm25_rank": int(bm25_rank),
        "counterfactual_d": {
            "exited_top10": exited,
            "minimum_displacement_to_exit_top10": minimum_exit,
            "observed_displacement": len(displacement),
            "minimum_removals_to_restore_top10": removals,
            "rank_if_intersection_removed": rank_without_intersection,
            "rank_if_queue_jumps_removed": rank_without_queue,
            "rank_if_dense_only_removed": rank_without_dense,
            "rank_if_bm25_only_removed": rank_without_bm25,
            "removal_restores_hit10": {
                "intersection": _restores(rank_without_intersection, truncated_rank),
                "queue_jumps": _restores(rank_without_queue, truncated_rank),
                "dense_only": _restores(rank_without_dense, truncated_rank),
                "bm25_only": _restores(rank_without_bm25, truncated_rank),
            },
            "displacement_category_can_supply_exit": {
                category: bool(exited and displacement_counts[category] >= minimum_exit > 0)
                for category in CATEGORIES
            },
        },
    }


def judge_h9(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Judge H9 on the three residual Hit@10 regressions.

    Strong support requires the same two-part mechanism on every case:
    an intersection queue jump and Dense-only rank competition, with deep-tail
    contribution absent. Partial support requires every case to be explained
    by at least one of those top-20 mechanisms while the mix differs or one
    part is absent. Anything with a deep-tail contribution, a broken identity,
    or an unexplained displacement is a rejection.
    """
    if len(cases) != len(RESIDUALS):
        raise UsageError("H9 is judged on the three residual BM25 regressions")
    deep_tail_clear = all(
        case["deep_tail_rank_used"] is False
        and float(case["positive_dense_contribution"]) == 0
        and int(case["max_contributing_rank"]) <= CANDIDATE_N
        for case in cases
    )
    property_ok = all(bool(case["structural_property_holds"]) for case in cases)
    explained = all(
        bool(case["identity_holds"]) and case["mechanism"] != MECHANISM_UNEXPLAINED
        for case in cases
    )
    labels = [str(case["mechanism"]) for case in cases]
    known = {MECHANISM_BOTH, MECHANISM_DOMINANCE, MECHANISM_SYMMETRIC}
    top20 = all(label in known for label in labels)
    both_on_every = all(label == MECHANISM_BOTH for label in labels)
    if not (deep_tail_clear and property_ok and explained and top20):
        decision = "Reject"
    elif both_on_every:
        decision = "Strong Support"
    else:
        decision = "Partial Support"
    return {
        "hypothesis": (
            "잔여 BM25 regression의 직접적인 rank-level mechanism은 "
            "deep-tail이 아니라 intersection dominance 및 symmetric rank competition이다."
        ),
        "decision": decision,
        "mechanisms": labels,
        "mechanisms_identical": len(set(labels)) == 1,
        "deep_tail_clear": deep_tail_clear,
        "structural_property_holds_on_all_three": property_ok,
        "identities_hold": explained,
        "asymmetric_fusion_justified": False,
        "weighted_rrf_implemented": False,
        "weight_sweep_run": False,
        "rule": (
            "Strong Support when deep-tail contribution is 0, the intersection "
            "prefix property holds, the BM25-only displacement identity holds, "
            "and every residual case is intersection queue jumps plus Dense-only "
            "competition. Partial Support when those audits hold and every case "
            "is one or both of those mechanisms, but the three cases are not the "
            "same pair. Reject when a deep-tail rank is used, the identity fails, "
            "or a displacement is unexplained."
        ),
    }


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rho with average ranks for ties. Undefined on a constant input."""
    if len(xs) != len(ys):
        raise UsageError("spearman inputs differ in length")
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def aggregate_dev_diagnostics(queries: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe DEV patterns. Coefficients stay descriptive."""
    changes = [
        query
        for query in queries
        if query["same_unit_rank_change"] is not None and query["bm25_positive_rank"] is not None
    ]
    identity_rows = [
        query for query in changes if query["identity_applies"] and query["identity_holds"]
    ]
    regressions = [query for query in identity_rows if int(query["same_unit_rank_change"]) > 0]
    by_intersection: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for query in changes:
        by_intersection[int(query["intersection_count"])].append(query)
    histogram = []
    for count in sorted(by_intersection):
        group = by_intersection[count]
        deltas = [int(query["same_unit_rank_change"]) for query in group]
        histogram.append(
            {
                "intersection_count": count,
                "queries": len(group),
                "mean_same_unit_rank_change": _mean(deltas),
                "median_same_unit_rank_change": _median(deltas),
                "hit10_regressions": sum(
                    1 for query in group if query["bm25_hit10"] and not query["truncated_hit10"]
                ),
                "small_n": len(group) < 5,
            }
        )
    movement_groups = []
    for label in ("improvement", "neutral", "regression", "recovered", "still_missing"):
        group = [query for query in queries if query["movement_vs_bm25"] == label]
        if not group:
            continue
        intersections = [int(query["intersection_count"]) for query in group]
        deltas = [
            int(query["same_unit_rank_change"])
            for query in group
            if query["same_unit_rank_change"] is not None
        ]
        dense_ahead = [
            int(query["dense_only_ahead"])
            for query in group
            if query["dense_only_ahead"] is not None
        ]
        movement_groups.append(
            {
                "movement_vs_bm25": label,
                "queries": len(group),
                "mean_intersection_count": _mean(intersections),
                "median_intersection_count": _median(intersections),
                "min_intersection_count": min(intersections),
                "max_intersection_count": max(intersections),
                "mean_same_unit_rank_change": _mean(deltas),
                "mean_dense_only_ahead": _mean(dense_ahead),
                "small_n": len(group) < 5,
            }
        )
    bands = []
    for name, low, high in (("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), ("11-20", 11, 20)):
        group = [
            query
            for query in changes
            if low <= int(query["bm25_positive_rank"]) <= high
        ]
        if not group:
            continue
        bands.append(
            {
                "bm25_positive_rank_band": name,
                "queries": len(group),
                "mean_bm25_positive_rank": _mean(
                    [int(query["bm25_positive_rank"]) for query in group]
                ),
                "mean_truncated_rank_of_that_unit": _mean(
                    [int(query["bm25_positive_truncated_rank"]) for query in group]
                ),
                "mean_same_unit_rank_change": _mean(
                    [int(query["same_unit_rank_change"]) for query in group]
                ),
                "small_n": len(group) < 5,
            }
        )
    return {
        "question_count": len(queries),
        "structural_property_holds": sum(
            1 for query in queries if query["structural_property_holds"]
        ),
        "same_unit_pairs": len(changes),
        "same_unit_rank_change_mean": _mean(
            [int(query["same_unit_rank_change"]) for query in changes]
        ),
        "same_unit_improved": sum(
            1 for query in changes if int(query["same_unit_rank_change"]) < 0
        ),
        "same_unit_neutral": sum(
            1 for query in changes if int(query["same_unit_rank_change"]) == 0
        ),
        "same_unit_regressed": sum(
            1 for query in changes if int(query["same_unit_rank_change"]) > 0
        ),
        "first_relevant_unit_differs_from_bm25_positive": sum(
            1 for query in queries if query["same_positive_unit"] is False
        ),
        "intersection_count_distribution": histogram,
        "movement_groups": movement_groups,
        "bm25_rank_bands": bands,
        "identity_population": len(identity_rows),
        "identity_regressions": _regression_pattern(regressions),
        "associations": [
            _association(
                "intersection_count",
                "same_unit_rank_change",
                changes,
                "queries whose BM25 positive is inside the stored top-20",
            ),
            _association(
                "dense_only_count",
                "same_unit_rank_change",
                changes,
                "queries whose BM25 positive is inside the stored top-20",
            ),
            _association(
                "dense_only_ahead",
                "same_unit_rank_change",
                identity_rows,
                "BM25-only positives whose displacement identity holds",
            ),
            _association(
                "bm25_positive_rank",
                "bm25_positive_truncated_rank",
                changes,
                "queries whose BM25 positive is inside the stored top-20",
            ),
        ],
        "association_note": ASSOCIATION_NOTE,
    }


def build_diagnosis(root: Path) -> dict[str, Any]:
    """Decompose stored DEV top-20 lists with the existing truncated RRF."""
    bundle = validate_source_bundle(root)
    if any(str(question.get("split")) != "dev" for question in bundle["questions"]):
        raise UsageError("H9 diagnosis received a non-dev question")
    stored = _load_stored(root)
    _audit_recovery(stored["recovery"])
    questions = sorted(bundle["questions"], key=lambda item: str(item["query_id"]))
    qrels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bundle["qrels"]:
        qrels_by_query[str(row["query_id"])].append(row)
    residual_by_id = {str(item["query_id"]): item for item in RESIDUALS}
    movement_by_id = {str(row["query_id"]): row for row in stored["movement"]}
    truncated_by_id = {str(row["query_id"]): row for row in stored["truncated_questions"]}
    if set(movement_by_id) != {str(question["query_id"]) for question in questions}:
        raise UsageError("stored truncated movement is not the DEV 45")
    queries: list[dict[str, Any]] = []
    residuals: list[dict[str, Any]] = []
    max_rank_used = 0
    for question in questions:
        query_id = str(question["query_id"])
        fused = candidate_truncated_rrf(
            bundle["dense_by_id"][query_id]["hits"],
            bundle["bm25_by_id"][query_id]["hits"],
        )
        rows = decompose_fused_ranking(fused, qrels_by_query.get(query_id, []))
        _audit_stored_order(query_id, rows, truncated_by_id[query_id]["hits"])
        prop = structural_property(rows)
        if not prop["holds"]:
            raise UsageError(f"{query_id} broke the intersection prefix property")
        for row in rows:
            for key in ("dense_rank", "bm25_rank"):
                if row[key] is not None:
                    max_rank_used = max(max_rank_used, int(row[key]))
        summary, case = _diagnose_query(
            question,
            rows,
            movement_by_id[query_id],
            prop,
            residual_by_id.get(query_id),
            max_rank_used_query=_max_rank(rows),
        )
        queries.append(summary)
        if case is not None:
            residuals.append(case)
    if max_rank_used > CANDIDATE_N:
        raise UsageError("diagnosis used a rank outside the top-20 window")
    if len(queries) != 45 or len(residuals) != 3:
        raise UsageError("diagnosis did not cover DEV 45 and the three residuals")
    lost = [
        query["query_id"]
        for query in queries
        if query["bm25_hit10"] and not query["truncated_hit10"]
    ]
    if sorted(lost) != sorted(residual_by_id):
        raise UsageError("BM25 Hit@10 losses are not the three pinned residuals")
    residuals.sort(key=lambda item: str(item["query_id"]))
    aggregate = aggregate_dev_diagnostics(queries)
    verdict = judge_h9(residuals)
    hypotheses = candidate_hypotheses(residuals)
    boundaries = [
        {
            "query_id": case["query_id"],
            "bm25_rank": case["bm25_rank"],
            "weight_ratio_to_beat_worst_intersection": intersection_dominance_weight_ratio(
                int(case["bm25_rank"])
            ),
            "weight_ratio_to_beat_dense_rank_1": dense_rank1_weight_ratio(int(case["bm25_rank"])),
            "scored": False,
        }
        for case in residuals
    ]
    payload: dict[str, Any] = {
        "experiment_id": "v21_dev_h9_truncated_rrf_rank_pressure",
        "dataset_id": DATASET_ID,
        "dataset_sha256": bundle["frozen"]["dataset_sha256"],
        "population": "dev fully_answerable",
        "question_count": 45,
        "test_questions_searched": 0,
        "retrieval_reexecuted": False,
        "new_ranker_implemented": False,
        "input_truncated_run": TRUNCATED_RUN_DIR,
        "input_sha256": {
            "dense_dev": bundle["dense_sha256"],
            "bm25_dev": bundle["bm25_sha256"],
            "truncated_rrf_dev": _sha_file(
                root / "evaluation" / "runs" / TRUNCATED_RUN_DIR / "truncated_rrf_dev.json"
            ),
        },
        "max_contributing_rank": max_rank_used,
        "deep_tail_ranks_used": 0,
        "queries": queries,
        "residual_cases": residuals,
        "aggregate": aggregate,
        "verdict": verdict,
        "candidate_hypotheses": hypotheses,
        "formula_boundaries_not_scored": boundaries,
        "git": git_revision(root),
    }
    payload["answers"] = build_answers(payload)
    payload["controlled_experiment"] = {
        "warranted": verdict["decision"] in {"Strong Support", "Partial Support"},
        "weight_sweep_warranted": False,
        "implemented": False,
        "reason": payload["answers"]["H"],
    }
    return payload


def candidate_hypotheses(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Name a later hypothesis only when a residual case shows that pressure."""
    found: list[dict[str, Any]] = []
    symmetric = [case["query_id"] for case in cases if case["symmetric_competition_pressure"] > 0]
    jumps = [case["query_id"] for case in cases if case["incremental_intersection_pressure"] > 0]
    if symmetric:
        found.append(
            {
                "name": "BM25-primary asymmetric fusion",
                "status": "hypothesis_only",
                "implemented": False,
                "query_ids": symmetric,
                "observed_basis": (
                    "Dense-only units with a better reciprocal rank sit ahead of "
                    "the BM25 positive."
                ),
            }
        )
    if jumps:
        found.append(
            {
                "name": "weighted RRF",
                "status": "hypothesis_only",
                "implemented": False,
                "query_ids": jumps,
                "observed_basis": (
                    "Intersection units with a worse BM25 rank sit ahead of the "
                    "BM25-only positive."
                ),
                "formula_note": (
                    "At k=60 and N=20 the worst intersection unit stays ahead of a "
                    "BM25-only rank r until w_bm25/w_dense exceeds (60+r)/(20-r). "
                    "That ratio was not searched and no weighted list was scored."
                ),
            }
        )
    return found


def build_answers(payload: dict[str, Any]) -> dict[str, str]:
    """Short answers. A–E restate counts. F–H are the judgment."""
    cases = payload["residual_cases"]
    verdict = payload["verdict"]
    aggregate = payload["aggregate"]
    pattern = aggregate["identity_regressions"]
    return {
        "A": " ".join(_ahead_sentence(case) for case in cases),
        "B": " ".join(_pressure_sentence(case) for case in cases),
        "C": " ".join(_dense_sentence(case) for case in cases),
        "D": _mechanism_sameness(cases),
        "E": (
            f"교집합 prefix 성질은 {aggregate['structural_property_holds']}/"
            f"{aggregate['question_count']}에서 성립한다. "
        f"BM25-only identity가 성립하는 same-unit regression "
        f"{pattern['queries']}건 중 queue jump가 있는 질의는 {pattern['with_queue_jumps']}건, "
        f"Dense-only displacement가 있는 질의는 {pattern['with_dense_displacement']}건, "
        f"둘 다 있는 질의는 {pattern['with_both']}건이다. "
        f"{_association_sentence(aggregate)} "
        f"{ASSOCIATION_NOTE}"
    ),
        "F": (
            f"H9 판정은 {verdict['decision']}이다. "
            f"세 mechanism은 {', '.join(verdict['mechanisms'])}이다."
        ),
        "G": _answer_g(payload),
        "H": _answer_h(verdict, cases),
    }


def write_diagnosis(root: Path, payload: dict[str, Any]) -> Path:
    """Write the diagnostic artifacts. Does not rewrite the truncated-RRF run."""
    output = root / "evaluation" / "runs" / RUN_DIR_NAME
    if dataset_dir(root) in output.parents or output == dataset_dir(root):
        raise UsageError("refusing to write the diagnosis into the dataset")
    output.mkdir(parents=True, exist_ok=True)
    queries = payload["queries"]
    residuals = payload["residual_cases"]
    _dump(output / "experiment_config.json", _config(payload))
    _dump(output / "query_diagnostics.json", {"queries": queries})
    _write_csv(output / "query_diagnostics.csv", queries, QUERY_COLUMNS)
    _dump(
        output / "residual_unit_decomposition.json",
        {"cases": residuals, "counterfactuals_are_not_a_new_ranker": True},
    )
    _write_csv(output / "residual_unit_decomposition.csv", _unit_csv_rows(residuals), UNIT_COLUMNS)
    _dump(
        output / "dev_aggregate.json",
        {
            "aggregate": payload["aggregate"],
            "verdict": payload["verdict"],
            "answers": payload["answers"],
            "candidate_hypotheses": payload["candidate_hypotheses"],
            "formula_boundaries_not_scored": payload["formula_boundaries_not_scored"],
            "controlled_experiment": payload["controlled_experiment"],
            "residual_summaries": [
                {key: value for key, value in case.items() if key != "units"} for case in residuals
            ],
        },
    )
    report = render_report(payload)
    (output / "report.md").write_text(report, encoding="utf-8")
    worklog = root / "docs" / "worklogs" / WORKLOG_NAME
    worklog.parent.mkdir(parents=True, exist_ok=True)
    worklog.write_text(render_worklog(payload), encoding="utf-8")
    return output


def render_report(payload: dict[str, Any]) -> str:
    """DEV diagnosis report with FACT and INTERPRETATION separated."""
    sections = [
        "# v2.1 DEV H9 rank-pressure diagnosis",
        "",
        "Candidate-truncated RRF의 기존 순위를 분해한다. 새 랭커는 없다. TEST 검색은 0이다.",
        "",
        "## DEFINITIONS",
        "",
        _definitions(),
        "",
        "## FACT",
        "",
        _fact_property(payload),
        "",
        _fact_cases(payload["residual_cases"]),
        "",
        _fact_dev(payload["aggregate"]),
        "",
        "## INTERPRETATION",
        "",
        _interpretation(payload),
        "",
        "## Answers",
        "",
        _answers_md(payload["answers"]),
        "",
        "## Verdict",
        "",
        f"{payload['verdict']['decision']}. {payload['verdict']['rule']}",
        "",
        "## Not done",
        "",
        _not_done(),
        "",
    ]
    return "\n".join(sections)


def render_worklog(payload: dict[str, Any]) -> str:
    verdict = payload["verdict"]
    lines = [
        "# v2.1 DEV H9 rank-pressure diagnosis — 2026-10-07",
        "",
        "## Goal",
        "",
        "Candidate-truncated RRF가 남긴 BM25 Hit@10 회귀 3건의 등위가 "
        "Top-20 안 intersection dominance와 symmetric rank competition 때문인지 분해한다. "
        "새 retrieval algorithm은 만들지 않는다. TEST는 실행하지 않는다.",
        "",
        "## Setup",
        "",
        (
            f"dataset `{payload['dataset_id']}` sha256 `{payload['dataset_sha256']}`. "
            "입력은 저장된 DEV Dense/BM25 top-20과 truncated RRF 산출물이다. "
            f"함수 `candidate_truncated_rrf`. N={CANDIDATE_N}, k={RRF_K}, weights 1:1. "
            f"점수에 들어간 최대 rank {payload['max_contributing_rank']}. "
            f"deep-tail rank 사용 {payload['deep_tail_ranks_used']}. "
            f"TEST 검색 {payload['test_questions_searched']}."
        ),
        "",
        "## FACT",
        "",
        _fact_cases(payload["residual_cases"]),
        "",
        _fact_dev(payload["aggregate"]),
        "",
        "## INTERPRETATION",
        "",
        _interpretation(payload),
        "",
        "## Answers",
        "",
        _answers_md(payload["answers"]),
        "",
        "## Verdict",
        "",
        f"{verdict['decision']}.",
        "",
        "산출물: `evaluation/runs/multi_company_retrieval_v2_1_dev_h9_diagnosis/`.",
        "",
    ]
    return "\n".join(lines)


def _diagnose_query(
    question: dict[str, Any],
    rows: list[dict[str, Any]],
    movement: dict[str, Any],
    prop: dict[str, Any],
    spec: dict[str, Any] | None,
    *,
    max_rank_used_query: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    bm25_unit, bm25_rank = _first_relevant(rows, source="bm25")
    truncated_unit, truncated_rank = _first_relevant(rows, source="fused")
    if bm25_unit is None:
        attr = None
    else:
        attr = attribute_rank_pressure(rows, positive_unit_id=bm25_unit, bm25_rank=int(bm25_rank))
        if attr["identity_applies"] and not attr["identity_holds"]:
            raise UsageError(f"{question['query_id']} BM25-only displacement identity failed")
    same_unit = bm25_unit is not None and bm25_unit == truncated_unit
    summary = {
        "query_id": question["query_id"],
        "company": question.get("company_name"),
        "question": question.get("question"),
        "prior_cell": movement.get("prior_cell"),
        "movement_vs_bm25": movement_label(bm25_rank, truncated_rank, hit_k=HIT_K),
        "bm25_positive_unit_id": bm25_unit,
        "bm25_positive_rank": bm25_rank,
        "bm25_positive_truncated_rank": None if attr is None else attr["truncated_rank"],
        "truncated_first_relevant_unit_id": truncated_unit,
        "truncated_first_relevant_rank": truncated_rank,
        "same_positive_unit": same_unit if bm25_unit is not None else None,
        "same_unit_rank_change": None if attr is None else attr["same_unit_rank_change"],
        "intersection_count": prop["intersection_count"],
        "dense_only_count": prop["dense_only_count"],
        "bm25_only_count": prop["bm25_only_count"],
        "union_count": len(rows),
        "intersection_ahead": None if attr is None else attr["ahead_counts"]["intersection"],
        "dense_only_ahead": None if attr is None else attr["ahead_counts"]["dense_only"],
        "bm25_only_ahead": None if attr is None else attr["ahead_counts"]["bm25_only"],
        "incremental_intersection_pressure": None
        if attr is None
        else attr["incremental_intersection_pressure"],
        "symmetric_competition_pressure": None
        if attr is None
        else attr["symmetric_competition_pressure"],
        "structural_property_holds": prop["holds"],
        "bm25_hit10": bm25_rank is not None and int(bm25_rank) <= HIT_K,
        "truncated_hit10": truncated_rank is not None and int(truncated_rank) <= HIT_K,
        "positive_category": None if attr is None else attr["positive_category"],
        "identity_applies": False if attr is None else attr["identity_applies"],
        "identity_holds": False if attr is None else attr["identity_holds"],
        "mechanism": None if attr is None else attr["mechanism"],
    }
    _audit_movement(summary, movement)
    if spec is None:
        return summary, None
    if bm25_rank != int(spec["bm25_rank"]) or truncated_rank != int(spec["truncated_rank"]):
        raise UsageError(f"{spec['query_id']} recomputed ranks are not the pinned residual")
    if prop["intersection_count"] != int(spec["intersection_count"]):
        raise UsageError(f"{spec['query_id']} intersection count is not the pinned residual")
    if attr is None or not same_unit:
        raise UsageError(f"{spec['query_id']} positive unit is not the truncated first relevant")
    if float(attr["dense_contribution"]) != 0:
        raise UsageError(f"{spec['query_id']} positive received a Dense contribution")
    case = {
        **attr,
        "query_id": question["query_id"],
        "label": spec["label"],
        "company": question.get("company_name"),
        "question": question.get("question"),
        "structural_property_holds": True,
        "structural_property": prop,
        "deep_tail_rank_used": False,
        "positive_dense_contribution": float(attr["dense_contribution"]),
        "max_contributing_rank": max_rank_used_query,
        "bm25_relevant_units": _bm25_relevant_units(rows),
        "units": [
            {
                **row,
                "is_primary_bm25_positive": row["retrieval_unit_id"] == bm25_unit,
                "ahead_of_primary_bm25_positive": int(row["final_rank"])
                < int(attr["truncated_rank"]),
            }
            for row in rows
        ],
    }
    return summary, case


def _bm25_relevant_units(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found = []
    for row in rows:
        if row["bm25_rank"] is None or row["relevance"] is None or int(row["relevance"]) < 1:
            continue
        ahead = ahead_category_counts(rows, before_rank=int(row["final_rank"]))
        found.append(
            {
                "retrieval_unit_id": row["retrieval_unit_id"],
                "bm25_rank": int(row["bm25_rank"]),
                "truncated_rank": int(row["final_rank"]),
                "relevance": int(row["relevance"]),
                "relevant_requirement_ids": list(row["relevant_requirement_ids"]),
                "category": row["category"],
                "ahead_counts": ahead,
                "outside_top10": int(row["final_rank"]) > HIT_K,
            }
        )
    found.sort(key=lambda item: int(item["bm25_rank"]))
    return found


def _first_relevant(
    rows: list[dict[str, Any]],
    *,
    source: str,
) -> tuple[str | None, int | None]:
    ordered = rows
    if source == "bm25":
        ordered = sorted(
            (row for row in rows if row["bm25_rank"] is not None),
            key=lambda row: int(row["bm25_rank"]),
        )
    for row in ordered:
        if row["relevance"] is not None and int(row["relevance"]) >= 1:
            rank = int(row["bm25_rank"] if source == "bm25" else row["final_rank"])
            return str(row["retrieval_unit_id"]), rank
    return None, None


def _audit_movement(summary: dict[str, Any], movement: dict[str, Any]) -> None:
    query_id = summary["query_id"]
    if not _same_rank(summary["bm25_positive_rank"], movement.get("bm25_first_relevant_rank")):
        raise UsageError(f"{query_id} BM25 positive rank differs from the stored movement")
    if not _same_rank(
        summary["truncated_first_relevant_rank"],
        movement.get("truncated_first_relevant_rank"),
    ):
        raise UsageError(f"{query_id} truncated positive rank differs from the stored movement")
    if int(summary["intersection_count"]) != int(movement["intersection_count"]):
        raise UsageError(f"{query_id} intersection count differs from the stored movement")
    if summary["movement_vs_bm25"] != movement["movement_vs_bm25"]:
        raise UsageError(f"{query_id} movement label differs from the stored movement")
    if bool(summary["truncated_hit10"]) != bool(movement["truncated_hit10"]):
        raise UsageError(f"{query_id} Hit@10 flag differs from the stored movement")


def _audit_recovery(recovery: dict[str, Any]) -> None:
    expected = {str(item["query_id"]): item for item in RESIDUALS}
    if sorted(recovery.get("still_lost_query_ids") or []) != sorted(expected):
        raise UsageError("stored truncated recovery is not the three pinned residuals")
    by_id = {str(row["query_id"]): row for row in recovery["queries"]}
    for query_id, spec in expected.items():
        row = by_id[query_id]
        if int(row["bm25_first_relevant_rank"]) != int(spec["bm25_rank"]):
            raise UsageError(f"{query_id} stored BM25 rank is not the pinned residual")
        if int(row["truncated_rank"]) != int(spec["truncated_rank"]):
            raise UsageError(f"{query_id} stored truncated rank is not the pinned residual")
        if int(row["truncated_first_relevant_rank"]) != int(spec["truncated_rank"]):
            raise UsageError(f"{query_id} stored first relevant rank is not the pinned residual")
        if int(row["intersection_count"]) != int(spec["intersection_count"]):
            raise UsageError(f"{query_id} stored intersection count is not the pinned residual")


def _audit_stored_order(
    query_id: str,
    rows: list[dict[str, Any]],
    stored_hits: list[dict[str, Any]],
) -> None:
    got = [str(row["retrieval_unit_id"]) for row in rows]
    stored = [str(hit["retrieval_unit_id"]) for hit in stored_hits]
    if got != stored:
        raise UsageError(f"{query_id} recomputed truncated order differs from the stored run")


def _load_stored(root: Path) -> dict[str, Any]:
    directory = root / "evaluation" / "runs" / TRUNCATED_RUN_DIR
    if not directory.is_dir():
        raise UsageError("stored truncated RRF run is missing")
    truncated = _read(directory / "truncated_rrf_dev.json")
    movement = _read(directory / "per_query_rank_movement.json")
    recovery = _read(directory / "regression_recovery.json")
    return {
        "truncated_questions": truncated["questions"],
        "movement": movement["queries"],
        "recovery": recovery,
    }


def _mechanism(*, applies: bool, holds: bool, queue_jumps: int, dense_displacement: int) -> str:
    if not applies:
        return MECHANISM_INTERSECTION_POSITIVE
    if not holds:
        return MECHANISM_UNEXPLAINED
    if queue_jumps > 0 and dense_displacement > 0:
        return MECHANISM_BOTH
    if queue_jumps > 0:
        return MECHANISM_DOMINANCE
    if dense_displacement > 0:
        return MECHANISM_SYMMETRIC
    return MECHANISM_UNEXPLAINED


def _regression_pattern(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "queries": len(rows),
        "with_queue_jumps": sum(
            1 for row in rows if int(row["incremental_intersection_pressure"]) > 0
        ),
        "with_dense_displacement": sum(
            1 for row in rows if int(row["symmetric_competition_pressure"]) > 0
        ),
        "with_both": sum(
            1
            for row in rows
            if int(row["incremental_intersection_pressure"]) > 0
            and int(row["symmetric_competition_pressure"]) > 0
        ),
        "reading": "counts of a DEV slice; not a causal estimate",
    }


def _association(
    x_name: str,
    y_name: str,
    rows: list[dict[str, Any]],
    population: str,
) -> dict[str, Any]:
    pairs = [
        (float(row[x_name]), float(row[y_name]))
        for row in rows
        if row[x_name] is not None and row[y_name] is not None
    ]
    value = None
    if len(pairs) >= 3:
        value = spearman([item[0] for item in pairs], [item[1] for item in pairs])
    return {
        "statistic": "spearman",
        "x": x_name,
        "y": y_name,
        "n": len(pairs),
        "value": value,
        "population": population,
        "reading": ASSOCIATION_NOTE,
    }


def _ahead_sentence(case: dict[str, Any]) -> str:
    counts = case["ahead_counts"]
    judgments = case["ahead_judgments"]
    text = (
        f"{case['label']}(`{case['query_id']}`) positive 앞에는 "
        f"intersection {counts['intersection']}개, dense_only {counts['dense_only']}개, "
        f"bm25_only {counts['bm25_only']}개가 있다. "
        f"그 앞 판정은 relevant {_bucket_total(judgments, 'relevant')}, "
        f"irrelevant {_bucket_total(judgments, 'irrelevant')}, "
        f"unjudged {_bucket_total(judgments, 'unjudged')}이다."
    )
    extras = case.get("bm25_relevant_units") or []
    if len(extras) > 1:
        listed = ", ".join(
            f"{unit['bm25_rank']}위→{unit['truncated_rank']}위" for unit in extras
        )
        text += f" BM25 양성 unit의 자리는 {listed}이다."
    return text


def _association_sentence(aggregate: dict[str, Any]) -> str:
    parts = []
    for row in aggregate["associations"]:
        value = "undefined" if row["value"] is None else f"{float(row['value']):.3f}"
        parts.append(f"{row['x']} 대 {row['y']} {value} (n={row['n']})")
    return "Spearman은 " + ", ".join(parts) + "이다."


def _pressure_sentence(case: dict[str, Any]) -> str:
    detail = case["counterfactual_d"]
    return (
        f"{case['label']}의 positional intersection pressure는 "
        f"{case['positional_intersection_pressure']}칸이고, "
        f"BM25 순서 대비 추가로 앞으로 온 queue jump는 "
        f"{case['incremental_intersection_pressure']}개다. "
        f"intersection을 앞에서 뺀 자리는 {case['counterfactual_a_rank']}위, "
        f"queue jump만 뺀 자리는 {case['counterfactual_a_queue_jump_only_rank']}위다. "
        f"A가 함께 빼는 BM25 predecessor intersection은 "
        f"{case['counterfactual_a_removes_predecessors']}개다. "
        f"Top-10을 벗어나는 최소 displacement는 "
        f"{detail['minimum_displacement_to_exit_top10']}이고 "
        f"관측 displacement는 {detail['observed_displacement']}이다."
    )


def _dense_sentence(case: dict[str, Any]) -> str:
    detail = case["counterfactual_d"]
    return (
        f"{case['label']}의 Dense-only ahead는 {case['counterfactual_b_dense_only_ahead']}개이고 "
        f"그 중 displacement는 {case['symmetric_competition_pressure']}개다. "
        f"Dense-only를 빼면 자리는 {detail['rank_if_dense_only_removed']}위가 되고 "
        f"Hit@10 복원 여부는 {detail['removal_restores_hit10']['dense_only']}이다. "
        f"BM25 원래 자리는 {case['counterfactual_c_bm25_rank']}위다."
    )


def _mechanism_sameness(cases: list[dict[str, Any]]) -> str:
    labels = ", ".join(f"{case['label']}={case['mechanism']}" for case in cases)
    identical = len({case["mechanism"] for case in cases}) == 1
    state = "세 mechanism은 같다." if identical else "세 mechanism은 같지 않다."
    return f"{labels}. {state}"


def _answer_g(payload: dict[str, Any]) -> str:
    lines = [
        "이 진단은 asymmetric fusion의 순위를 만들지 않았다. "
        "관측된 pressure만으로 그 규칙을 채택할 근거는 되지 않는다. "
        "Dense rank 1을 넘는 weight ratio와 최악 intersection을 넘는 weight ratio는 "
        "크기가 다르다. 한 weight가 세 pressure를 함께 지우는지 이 진단은 점수화하지 않았다."
    ]
    for item in payload["formula_boundaries_not_scored"]:
        lines.append(
            f"`{item['query_id']}` BM25 {item['bm25_rank']}위에서 "
            f"Dense rank 1을 넘기려면 weight ratio "
            f"{item['weight_ratio_to_beat_dense_rank_1']:.3f}, "
            f"최악 intersection을 넘기려면 "
            f"{item['weight_ratio_to_beat_worst_intersection']:.3f}이다. "
            "이 비율은 식의 경계이고 점수화하지 않았다."
        )
    return " ".join(lines)


def _answer_h(verdict: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    if verdict["decision"] == "Reject":
        return (
            "H9가 기각이면 이 분해는 다음 랭킹 실험을 열 근거가 되지 않는다. "
            "이번 작업은 그 실험을 구현하지 않는다."
        )
    names = ", ".join(case["mechanism"] for case in cases)
    return (
        "Top-20 안의 등위 경쟁이 세 Hit@10 회귀의 displacement를 설명하므로, "
        "기전 하나를 미리 못 박은 DEV 실험을 설계할 근거는 있다. "
        f"세 사례의 mechanism은 {names}이다. "
        "하나의 weight sweep이 세 건을 동시에 설명한다는 실험은 이 분해의 다음 단계가 아니다. "
        "이번 작업은 그 실험을 구현하지 않는다."
    )


def _definitions() -> str:
    floor = intersection_min_score()
    ceiling = single_list_max_score()
    return "\n".join(
        [
            (
                "후보 집합은 Dense top-20과 BM25 top-20의 union이다. "
                f"score = I_dense/({RRF_K}+dense_rank) + I_bm25/({RRF_K}+bm25_rank). "
                f"교집합 최소 점수는 {floor:.6f}이고 단일 리스트 최대 점수는 {ceiling:.6f}이다."
            ),
            (
                "intersection은 양쪽 top-20에 있는 unit, dense_only는 Dense top-20에만 있는 unit, "
                "bm25_only는 BM25 top-20에만 있는 unit이다."
            ),
            (
                "primary positive는 BM25 top-20 안에서 판정 relevance가 1 이상인 첫 unit이다. "
                "unjudged는 relevance 0이 아니다."
            ),
            (
                "BM25 predecessor는 BM25 rank가 positive보다 앞인 unit이다. "
                "queue jump는 positive보다 BM25 rank가 나쁜데 truncated 순서에서는 앞에 있는 "
                "intersection unit이다. symmetric competition pressure는 positive 앞에 있는 "
                "Dense-only unit 수다."
            ),
            (
                "counterfactual A는 positive 앞의 intersection unit을 순서에서 지운 자리다. "
                "BM25가 원래 앞에 둔 intersection predecessor도 지워진다. "
                "queue jump만 지운 자리는 따로 적는다. "
                "B는 Dense-only ahead 수다. C는 BM25 원래 rank다. "
                "D의 최소 displacement는 BM25 rank를 11위로 만드는 추가 unit 수인 "
                "`11 - BM25 rank`다. Top-10으로 되돌리는 최소 제거 수는 "
                "`truncated rank - 10`이다. 이 자리는 새 랭킹이 아니다."
            ),
        ]
    )


def _fact_property(payload: dict[str, Any]) -> str:
    lines = [
        "### Structural property",
        "",
        (
            f"세 residual의 deep-tail rank 사용은 {payload['deep_tail_ranks_used']}이고 "
            f"점수에 들어간 최대 rank는 {payload['max_contributing_rank']}이다."
        ),
        "",
    ]
    for case in payload["residual_cases"]:
        prop = case["structural_property"]
        lines.append(
            f"- {case['label']}: holds={prop['holds']}, "
            f"intersection {prop['intersection_count']}, "
            f"worst intersection score {_score(prop['worst_intersection_score'])}, "
            f"best single-list score {_score(prop['best_single_list_score'])}, "
            f"vacuous_intersection={prop['vacuous_intersection']}."
        )
    return "\n".join(lines)


def _fact_cases(cases: list[dict[str, Any]]) -> str:
    lines = [
        "### Residual decompositions",
        "",
        (
            "| query | BM25 | truncated | ahead I | ahead D | ahead B | "
            "queue jump | dense displacement | A | A queue-only | "
            "dense-removed | min exit | removals to restore |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        detail = case["counterfactual_d"]
        counts = case["ahead_counts"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{case['label']}",
                    str(case["bm25_rank"]),
                    str(case["truncated_rank"]),
                    str(counts["intersection"]),
                    str(counts["dense_only"]),
                    str(counts["bm25_only"]),
                    str(case["incremental_intersection_pressure"]),
                    str(case["symmetric_competition_pressure"]),
                    str(case["counterfactual_a_rank"]),
                    str(case["counterfactual_a_queue_jump_only_rank"]),
                    str(detail["rank_if_dense_only_removed"]),
                    str(detail["minimum_displacement_to_exit_top10"]),
                    str(detail["minimum_removals_to_restore_top10"]),
                ]
            )
            + " |"
        )
    lines.append("")
    for case in cases:
        lines.extend(_case_detail(case))
    return "\n".join(lines)


def _case_detail(case: dict[str, Any]) -> list[str]:
    detail = case["counterfactual_d"]
    lines = [
        f"#### {case['label']}",
        "",
        f"`{case['query_id']}` {case['company']}: {case['question']}",
        "",
        (
            f"primary `{case['positive_unit_id']}` category {case['positive_category']}. "
            f"Dense contribution {_score(case['dense_contribution'])}, "
            f"BM25 contribution {_score(case['bm25_contribution'])}, "
            f"fused score {_score(case['fused_score'])}. "
            f"mechanism `{case['mechanism']}`."
        ),
        "",
        (
            f"predecessor {case['predecessor_count']} "
            f"`{json.dumps(case['predecessor_counts'], ensure_ascii=False)}`. "
            f"ahead judgments `{json.dumps(case['ahead_judgments'], ensure_ascii=False)}`. "
            f"displacement judgments "
            f"`{json.dumps(case['displacement_judgments'], ensure_ascii=False)}`."
        ),
        "",
        (
            "removal restores Hit@10: "
            f"`{json.dumps(detail['removal_restores_hit10'], ensure_ascii=False)}`. "
            "displacement category can supply the exit alone: "
            f"`{json.dumps(detail['displacement_category_can_supply_exit'], ensure_ascii=False)}`."
        ),
        "",
        "Displacement units:",
        "",
    ]
    if not case["displacement_units"]:
        lines.append("없음.")
    else:
        for unit in case["displacement_units"]:
            lines.append(
                f"- rank {unit['final_rank']} `{unit['retrieval_unit_id']}` "
                f"{unit['category']} dense {unit['dense_rank']} bm25 {unit['bm25_rank']} "
                f"relevance {unit['relevance']} "
                f"requirements {', '.join(unit['relevant_requirement_ids']) or '-'}"
            )
    relevant = case.get("bm25_relevant_units") or []
    lines.extend(["", f"BM25 relevant units in the union: {len(relevant)}.", ""])
    for unit in relevant:
        lines.append(
            f"- BM25 {unit['bm25_rank']} → truncated {unit['truncated_rank']} "
            f"`{unit['retrieval_unit_id']}` {unit['category']} "
            f"requirements {', '.join(unit['relevant_requirement_ids']) or '-'} "
            f"ahead `{json.dumps(unit['ahead_counts'], ensure_ascii=False)}`."
        )
    lines.append("")
    return lines


def _fact_dev(aggregate: dict[str, Any]) -> str:
    lines = [
        "### DEV aggregate",
        "",
        (
            f"질의 {aggregate['question_count']}. "
            f"교집합 prefix 성립 {aggregate['structural_property_holds']}/"
            f"{aggregate['question_count']}. "
            f"BM25 positive가 top-20 안에 있는 pair {aggregate['same_unit_pairs']}. "
            f"same-unit rank change 평균 {_num(aggregate['same_unit_rank_change_mean'])}. "
            f"앞당겨짐 {aggregate['same_unit_improved']}, "
            f"동일 {aggregate['same_unit_neutral']}, "
            f"밀림 {aggregate['same_unit_regressed']}. "
            f"truncated 첫 양성과 BM25 positive unit이 다른 질의 "
            f"{aggregate['first_relevant_unit_differs_from_bm25_positive']}. "
            "movement 라벨은 첫 양성 rank의 비교이고, same-unit change는 "
            "BM25 positive unit의 자리다. 두 unit이 다르면 라벨과 same-unit change가 어긋난다. "
            "mean dense_only ahead는 intersection positive도 포함한다. "
            "그 positive는 prefix 안에 있어 Dense-only ahead가 0인 경우가 많다."
        ),
        "",
        "Intersection count와 same-unit rank change. 칸의 n이 5 미만이면 small_n이다.",
        "",
        "| intersection | n | mean rank change | median | Hit@10 losses | small_n |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in aggregate["intersection_count_distribution"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["intersection_count"]),
                    str(row["queries"]),
                    _num(row["mean_same_unit_rank_change"]),
                    _num(row["median_same_unit_rank_change"]),
                    str(row["hit10_regressions"]),
                    str(row["small_n"]).lower(),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "| movement | n | mean intersection | median intersection | "
            "mean same-unit change | mean dense_only ahead | small_n |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in aggregate["movement_groups"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["movement_vs_bm25"]),
                    str(row["queries"]),
                    _num(row["mean_intersection_count"]),
                    _num(row["median_intersection_count"]),
                    _num(row["mean_same_unit_rank_change"]),
                    _num(row["mean_dense_only_ahead"]),
                    str(row["small_n"]).lower(),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "| BM25 rank band | n | mean BM25 rank | mean truncated rank | mean change | small_n |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in aggregate["bm25_rank_bands"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["bm25_positive_rank_band"]),
                    str(row["queries"]),
                    _num(row["mean_bm25_positive_rank"]),
                    _num(row["mean_truncated_rank_of_that_unit"]),
                    _num(row["mean_same_unit_rank_change"]),
                    str(row["small_n"]).lower(),
                ]
            )
            + " |"
        )
    pattern = aggregate["identity_regressions"]
    lines.extend(
        [
            "",
            (
                f"BM25-only identity regression {pattern['queries']}건. "
                f"queue jump {pattern['with_queue_jumps']}, "
                f"Dense-only displacement {pattern['with_dense_displacement']}, "
                f"둘 다 {pattern['with_both']}."
            ),
            "",
            "기술 상관계수. " + aggregate["association_note"],
            "",
            "| x | y | n | spearman | population |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in aggregate["associations"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["x"]),
                    str(row["y"]),
                    str(row["n"]),
                    _num(row["value"]),
                    str(row["population"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _interpretation(payload: dict[str, Any]) -> str:
    verdict = payload["verdict"]
    hypotheses = payload["candidate_hypotheses"] or []
    names = ", ".join(str(item["name"]) for item in hypotheses) or "없음"
    pieces = [
        f"판정은 {verdict['decision']}이다.",
        "세 질의 positive Dense contribution은 0이고 사용된 rank는 20 이하다.",
    ]
    for case in payload["residual_cases"]:
        pieces.append(
            f"{case['label']}의 mechanism은 {case['mechanism']}이다. "
            f"queue jump {case['incremental_intersection_pressure']}, "
            f"Dense-only displacement {case['symmetric_competition_pressure']}. "
            f"queue jump를 빼면 {case['counterfactual_a_queue_jump_only_rank']}위, "
            f"Dense-only를 빼면 {case['counterfactual_d']['rank_if_dense_only_removed']}위다."
        )
    if verdict["mechanisms_identical"]:
        pieces.append("세 사례의 mechanism 라벨은 같다.")
    else:
        pieces.append("세 사례의 mechanism 라벨은 서로 다르다.")
    pieces.extend(
        [
            "DEV 계수 표는 그 구성의 분포다.",
            "그 분포를 세 건의 원인으로 읽지 않는다.",
            f"관측이 이름을 준 다음 후보는 {names}이다.",
            "후보는 구현하지 않았고 weight도 점수화하지 않았다.",
            "lexical confidence 점수는 이 산출물에 없다.",
        ]
    )
    return " ".join(pieces)


def _answers_md(answers: dict[str, str]) -> str:
    labels = {
        "A": "A. positive 앞을 무엇이 차지했는가?",
        "B": "B. intersection dominance의 rank pressure는 얼마인가?",
        "C": "C. Dense-only competition은 얼마를 기여했는가?",
        "D": "D. 세 사례의 failure mechanism이 동일한가?",
        "E": "E. DEV 전체에서도 같은 pattern이 관찰되는가?",
        "F": "F. H9 판정은 무엇인가?",
        "G": "G. 이 결과만으로 asymmetric fusion을 정당화할 수 있는가?",
        "H": "H. 다음 controlled experiment를 시작할 근거가 있는가?",
    }
    lines = []
    for key in ("A", "B", "C", "D", "E", "F", "G", "H"):
        kind = "FACT" if key in {"A", "B", "C", "D", "E"} else "INTERPRETATION"
        if key == "F":
            kind = "VERDICT"
        lines.extend([f"### {labels[key]}", "", f"{kind}. {answers[key]}", ""])
    return "\n".join(lines).rstrip()


def _not_done() -> str:
    return (
        "weighted RRF, weight sweep, N sweep, k sweep, score fusion, alpha tuning, "
        "query routing, reranker, query expansion, embedding 변경, chunking 변경, "
        "period-aware ranking은 구현하지 않았다. TEST는 실행하지 않았다. "
        "counterfactual 자리는 진단 수치이고 시스템 점수가 아니다."
    )


def _config(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": payload["experiment_id"],
        "dataset_id": payload["dataset_id"],
        "dataset_sha256": payload["dataset_sha256"],
        "population": payload["population"],
        "question_count": payload["question_count"],
        "test_questions_searched": 0,
        "retrieval_reexecuted": False,
        "new_ranker_implemented": False,
        "input_truncated_run": payload["input_truncated_run"],
        "input_sha256": payload["input_sha256"],
        "ranker": "disclosure_agent.evalset_v2_1.truncated_rrf.candidate_truncated_rrf",
        "candidate_n": CANDIDATE_N,
        "k": RRF_K,
        "weights": "1:1",
        "forbidden": {
            "weighted_rrf": False,
            "weight_sweep": False,
            "n_sweep": False,
            "k_sweep": False,
            "score_fusion": False,
            "alpha_tuning": False,
            "query_routing": False,
            "reranker": False,
            "query_expansion": False,
            "embedding_change": False,
            "chunking_change": False,
            "period_aware_ranking": False,
            "test_execution": False,
        },
        "verdict": payload["verdict"]["decision"],
        "git_commit": payload["git"].get("git_commit"),
        "git_dirty": payload["git"].get("git_dirty"),
    }


def _unit_csv_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        for unit in case["units"]:
            rows.append({"query_id": case["query_id"], **unit})
    return rows


def _short_unit(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "retrieval_unit_id": row["retrieval_unit_id"],
        "final_rank": int(row["final_rank"]),
        "category": row["category"],
        "dense_rank": row["dense_rank"],
        "bm25_rank": row["bm25_rank"],
        "relevance": row["relevance"],
        "relevant_requirement_ids": list(row["relevant_requirement_ids"]),
    }


def _category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {category: 0 for category in CATEGORIES}
    for row in rows:
        counts[str(row["category"])] += 1
    return counts


def _judgment_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {
        category: {"relevant": 0, "irrelevant": 0, "unjudged": 0} for category in CATEGORIES
    }
    for row in rows:
        bucket = _judgment_bucket(row)
        counts[str(row["category"])][bucket] += 1
    return counts


def _judgment_bucket(row: dict[str, Any]) -> str:
    if row["relevance"] is None:
        return "unjudged"
    if int(row["relevance"]) >= 1:
        return "relevant"
    return "irrelevant"


def _bucket_total(counts: dict[str, dict[str, int]], bucket: str) -> int:
    return sum(item[bucket] for item in counts.values())


def _restores(rank: int, current: int) -> bool:
    return int(current) > HIT_K and int(rank) <= HIT_K


def _find_row(rows: list[dict[str, Any]], unit_id: str) -> dict[str, Any]:
    for row in rows:
        if str(row["retrieval_unit_id"]) == unit_id:
            return row
    raise UsageError(f"{unit_id} is not in the fused ranking")


def _optional_rank(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _same_rank(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return int(left) == int(right)


def _max_rank(rows: list[dict[str, Any]]) -> int:
    found = 0
    for row in rows:
        for key in ("dense_rank", "bm25_rank"):
            if row[key] is not None:
                found = max(found, int(row[key]))
    return found


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = (start + end) / 2 + 1
        for position in range(start, end + 1):
            ranks[order[position]] = average
        start = end + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    count = len(xs)
    if count < 3 or count != len(ys):
        return None
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    if den_x == 0 or den_y == 0:
        return None
    return numerator / ((den_x * den_y) ** 0.5)


def _mean(values: list[float] | list[int]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: list[float] | list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _score(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _num(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return str(value)
