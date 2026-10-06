"""Candidate-truncated RRF on v2.1 DEV.

One change from the rejected full-ranking run: Dense top-20 and BM25 top-20
are the only fusion candidates. A unit outside a retriever's top-20 contributes
0 from that retriever. Ranks such as 75, 283, and 705 are not score inputs.

k stays 60. Weights stay 1:1. N stays 20. There is no sweep, no routing, no
reranker, and no test search. Stored DEV rankings are reused after the
complementarity artifact and the full-RRF copy agree on all 45 top-20 lists.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from disclosure_agent.bm25 import BM25_PARAMS, TOKENIZER_VERSION
from disclosure_agent.dense import DENSE_MODEL_ID
from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2_1 import DATASET_ID
from disclosure_agent.evalset_v2_1.complementarity import (
    CELL_BM25,
    CELL_BOTH,
    CELL_DENSE,
    CELL_MISS,
    covered_requirement_ids,
    first_relevant_rank,
    relevant_ids_at,
)
from disclosure_agent.evalset_v2_1.freeze import dataset_dir
from disclosure_agent.evalset_v2_1.metrics import aggregate_headline, score_question
from disclosure_agent.evalset_v2_1.rrf import (
    RUN_DIR_NAME as FULL_RUN_DIR,
)
from disclosure_agent.evalset_v2_1.rrf import (
    movement_label,
    rank_delta,
    validate_source_bundle,
)
from disclosure_agent.evaluation import git_revision
from disclosure_agent.hybrid import RRF_K

RUN_DIR_NAME = "multi_company_retrieval_v2_1_dev_truncated_rrf"
CANDIDATE_N = 20
HIT_K = 10
INTENDED_BASE = "6c3fec85523545e3552a71930b56409b8be8e754"
STRONG_HIT10 = 41
MOST_OF_EIGHT = 6
CLEAR_GAIN_OVER_FULL = 3
RECOVERED_FOR_PARTIAL = 4
RANKING_DROP_LIMIT = 0.05
REQUIREMENT_RECALL_FLOOR = 49
ALL_REQUIREMENTS_FLOOR = 39
PINNED_FULL_HITS = {"1": 28, "5": 33, "10": 36}
PINNED_FULL_REQUIREMENT = (46, 58)
PINNED_FULL_ALL_REQUIREMENTS = (35, 45)
LOST_EIGHT = (
    "V2-00121941-business_paragraph-02",
    "V2-00121941-business_paragraph-03",
    "V2-00121941-finance_table-02",
    "V2-00121941-overview_paragraph-01",
    "V2-00126380-governance_paragraph-01",
    "V2-00126955-finance_paragraph-02",
    "V2-00989619-governance_paragraph-02",
    "V2-00989619-overview_paragraph-01",
)
NAMED_CASES = (
    {
        "query_id": "V2-00126380-governance_paragraph-01",
        "role": "bm25_lexical",
        "bm25_rank": 1,
        "excluded_dense_rank": 705,
        "full_rrf_rank": 27,
    },
    {
        "query_id": "V2-00121941-overview_paragraph-01",
        "role": "bm25_lexical",
        "bm25_rank": 1,
        "excluded_dense_rank": 283,
        "full_rrf_rank": 23,
    },
    {
        "query_id": "V2-00121941-finance_table-02",
        "role": "bm25_lexical",
        "bm25_rank": 7,
        "excluded_dense_rank": 75,
        "full_rrf_rank": 24,
    },
)
DENSE_ONLY_CASE = {
    "query_id": "V2-00126380-finance_table-02",
    "dense_rank": 1,
    "excluded_bm25_rank": 75,
    "full_rrf_rank": 6,
}
BOTH_MISS = (
    "V2-00107613-cross_period-02",
    "V2-00107613-finance_table-03",
)
MOVEMENT_COLUMNS = (
    "query_id",
    "company",
    "prior_cell",
    "bm25_first_relevant_rank",
    "dense_first_relevant_rank",
    "bm25_unit_in_dense_top20",
    "full_rrf_first_relevant_rank",
    "truncated_first_relevant_rank",
    "full_rrf_hit10",
    "truncated_hit10",
    "hit10_recovered_vs_full_rrf",
    "rank_delta_vs_bm25",
    "movement_vs_bm25",
    "intersection_count",
)


def intersection_min_score() -> float:
    """Lowest score of a unit present in both top-20 lists."""
    return 2.0 / (RRF_K + CANDIDATE_N)


def single_list_max_score() -> float:
    """Highest score of a unit present in only one top-20 list."""
    return 1.0 / (RRF_K + 1)


def implementation_audit() -> dict[str, Any]:
    """State truncated RRF. This is not reciprocal_rank_fusion."""
    return {
        "function": (
            "disclosure_agent.evalset_v2_1.truncated_rrf.candidate_truncated_rrf"
        ),
        "full_ranking_function_not_used": "disclosure_agent.hybrid.reciprocal_rank_fusion",
        "k": RRF_K,
        "candidate_n": CANDIDATE_N,
        "k_swept": False,
        "n_swept": False,
        "rank_base": 1,
        "formula": (
            "score(d) = I_dense(d)/(k + dense_rank(d)) + I_bm25(d)/(k + bm25_rank(d))"
        ),
        "I_dense": "1 when d is in the Dense top-20, else 0",
        "I_bm25": "1 when d is in the BM25 top-20, else 0",
        "weights": "1:1",
        "deep_tail_rank_used": False,
        "ranks_outside_top_n_are_not_score_inputs": True,
        "outside_union_is_not_a_candidate": True,
        "duplicate_retrieval_unit_id": "one row; the two contributions add",
        "tie_break": ["score descending", "retrieval_unit_id ascending"],
        "tie_break_reason": (
            "Stored top-20 hits have no ordinal or unit_index. "
            "retrieval_unit_id is a total order, so equal scores do not depend on "
            "dict order."
        ),
        "intersection_min_score": intersection_min_score(),
        "single_list_max_score": single_list_max_score(),
        "intersection_precedes_single_list": intersection_min_score() > single_list_max_score(),
        "weighted_score_fusion": False,
        "query_routing": False,
        "reranker": False,
    }


def candidate_truncated_rrf(
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    *,
    n: int = CANDIDATE_N,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Fuse the union of two top-n lists. Ranks beyond n are not read."""
    if n != CANDIDATE_N:
        raise UsageError(f"truncated RRF fixes N at {CANDIDATE_N}, got {n}")
    if k != RRF_K:
        raise UsageError(f"truncated RRF fixes k at {RRF_K}, got {k}")
    dense = _top_index(dense_hits, "dense", n)
    bm25 = _top_index(bm25_hits, "bm25", n)
    unit_ids = sorted(set(dense) | set(bm25))
    if not unit_ids:
        raise UsageError("truncated RRF union is empty")
    scored: list[dict[str, Any]] = []
    for unit_id in unit_ids:
        dense_rank = dense[unit_id]["rank"] if unit_id in dense else None
        bm25_rank = bm25[unit_id]["rank"] if unit_id in bm25 else None
        dense_part = _contribution(dense_rank, k, "dense")
        bm25_part = _contribution(bm25_rank, k, "bm25")
        if dense_part == 0 and bm25_part == 0:
            raise UsageError(f"{unit_id} is outside both top-{n} lists")
        source = dense.get(unit_id) or bm25[unit_id]
        other = bm25.get(unit_id) if unit_id in dense else None
        elements = _elements(source, other)
        scored.append(
            {
                "retrieval_unit_id": unit_id,
                "chunk_id": unit_id,
                "score": dense_part + bm25_part,
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
                "dense_contribution": dense_part,
                "bm25_contribution": bm25_part,
                "in_dense_top_n": dense_rank is not None,
                "in_bm25_top_n": bm25_rank is not None,
                "source_element_ids": elements,
            }
        )
    scored.sort(key=lambda item: (-float(item["score"]), str(item["retrieval_unit_id"])))
    _assert_intersection_precedes(scored)
    fused = []
    seen: set[str] = set()
    for rank, item in enumerate(scored, start=1):
        unit_id = str(item["retrieval_unit_id"])
        if unit_id in seen:
            raise UsageError(f"duplicate fused retrieval_unit_id {unit_id}")
        seen.add(unit_id)
        item["rank"] = rank
        fused.append(item)
    return fused


def judge_truncated_hypothesis(
    *,
    truncated_hit10: int,
    full_hit10: int,
    recovered_of_eight: int,
    dense_only_retained: bool,
    both_hit_lost: int,
    prior_bm25_only_kept: int,
    prior_bm25_only_kept_still: int,
    bm25_mrr: float,
    truncated_mrr: float,
    bm25_ndcg: float,
    truncated_ndcg: float,
    bm25_recall5: float,
    truncated_recall5: float,
    truncated_requirement_recall: int,
    truncated_all_requirements: int,
) -> dict[str, Any]:
    """Apply the pre-registered reading of the truncated-RRF criteria.

    Strong support needs Hit@10 at least 41/45, the Dense-only question still
    inside Hit@10, at least 6 of the 8 full-RRF BM25-only losses back inside
    Hit@10, no new both-hit loss, the two BM25-only questions that full-RRF
    already kept still kept, ranking drops of at most 0.05, and requirement
    counts within 2 of BM25. Partial support needs a gain of at least 3 Hit@10
    over full-RRF and at least 4 of those 8 recovered. Anything else is a
    rejection. These cutoffs were fixed before the DEV scores were read.
    """
    drops = {
        "mrr": float(bm25_mrr) - float(truncated_mrr),
        "ndcg@10": float(bm25_ndcg) - float(truncated_ndcg),
        "recall@5_macro": float(bm25_recall5) - float(truncated_recall5),
    }
    ranking_ok = all(value <= RANKING_DROP_LIMIT for value in drops.values())
    requirements_ok = (
        int(truncated_requirement_recall) >= REQUIREMENT_RECALL_FLOOR
        and int(truncated_all_requirements) >= ALL_REQUIREMENTS_FLOOR
    )
    coverage_ok = (
        int(truncated_hit10) >= STRONG_HIT10
        and int(both_hit_lost) == 0
        and int(prior_bm25_only_kept_still) == int(prior_bm25_only_kept)
    )
    strong = (
        coverage_ok
        and bool(dense_only_retained)
        and int(recovered_of_eight) >= MOST_OF_EIGHT
        and ranking_ok
        and requirements_ok
    )
    clearly_better = (
        int(truncated_hit10) >= int(full_hit10) + CLEAR_GAIN_OVER_FULL
        and int(recovered_of_eight) >= RECOVERED_FOR_PARTIAL
    )
    if strong:
        label = "A"
    elif clearly_better:
        label = "B"
    else:
        label = "C"
    return {
        "label": label,
        "decision": {"A": "채택", "B": "부분 지지", "C": "기각"}[label],
        "truncated_hit10": int(truncated_hit10),
        "full_hit10": int(full_hit10),
        "recovered_of_eight": int(recovered_of_eight),
        "dense_only_retained": bool(dense_only_retained),
        "both_hit_lost": int(both_hit_lost),
        "ranking_drops": drops,
        "ranking_within_limit": ranking_ok,
        "requirements_near_bm25": requirements_ok,
        "rule": (
            "A when Hit@10 is at least 41/45, Dense-only Hit@10 remains, "
            "at least 6 of the 8 full-RRF BM25-only losses return to Hit@10, "
            "both-hit losses are 0, previously kept BM25-only hits remain, "
            "MRR, NDCG@10, and Recall@5 macro each drop by at most 0.05, "
            "and requirement recall is at least 49/58 with all-requirements "
            "at least 39/45. Otherwise B when Hit@10 is at least 3 above "
            "full-RRF and at least 4 of the 8 are recovered. Otherwise C."
        ),
    }


def run_truncated_rrf_experiment(root: Path) -> dict[str, Any]:
    """Score stored DEV top-20 lists. Do not search, and do not retune."""
    bundle = validate_source_bundle(root)
    full_dir = root / "evaluation" / "runs" / FULL_RUN_DIR
    mismatches = ranking_mismatches(bundle, full_dir)
    output = root / "evaluation" / "runs" / RUN_DIR_NAME
    if mismatches:
        output.mkdir(parents=True, exist_ok=True)
        _dump(
            output / "reproduction_mismatch.json",
            {
                "fusion_scored": False,
                "mismatch_count": len(mismatches),
                "mismatches": mismatches,
            },
        )
        raise UsageError(
            "stored Dense/BM25 top-20 does not match across artifacts: "
            f"{mismatches[0]['query_id']} {mismatches[0]['label']}. "
            "Truncated fusion was not scored."
        )
    full = _load_full_rrf(full_dir)
    questions = sorted(bundle["questions"], key=lambda item: str(item["query_id"]))
    if any(str(item.get("split")) != "dev" for item in questions):
        raise UsageError("truncated RRF received a non-dev question")
    qrels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bundle["qrels"]:
        qrels_by_query[str(row["query_id"])].append(row)

    scored_rows: list[dict[str, Any]] = []
    movement: list[dict[str, Any]] = []
    covered_by_query: dict[str, set[str]] = {}
    formula_checks = 0
    max_rank_used = 0
    for question in questions:
        query_id = str(question["query_id"])
        dense_hits = bundle["dense_by_id"][query_id]["hits"]
        bm25_hits = bundle["bm25_by_id"][query_id]["hits"]
        fused = candidate_truncated_rrf(dense_hits, bm25_hits)
        formula_checks += len(fused)
        for hit in fused:
            for key in ("dense_rank", "bm25_rank"):
                if hit[key] is not None:
                    max_rank_used = max(max_rank_used, int(hit[key]))
        owned = qrels_by_query.get(query_id) or []
        scored = score_question(question, owned, bundle["requirements"], fused)
        covered, _missing = covered_requirement_ids(
            query_id,
            bundle["requirements"],
            owned,
            scored["hits"],
            k=HIT_K,
        )
        covered_by_query[query_id] = set(covered)
        movement.append(
            _movement_record(
                question,
                bundle["query_by_id"][query_id],
                dense_hits,
                bm25_hits,
                full["movement_by_id"][query_id],
                scored,
                fused,
            )
        )
        scored_rows.append(scored)
    if len(scored_rows) != 45 or max_rank_used > CANDIDATE_N:
        raise UsageError("truncated RRF used a rank outside the top-20 window")
    movement.sort(key=lambda item: item["query_id"])
    scored_rows.sort(key=lambda item: item["query_id"])
    truncated_headline = aggregate_headline(scored_rows)
    transitions = _requirement_transitions(bundle["requirement_rows"], covered_by_query)
    retention = _retention(movement)
    recovery = _recovery(movement)
    _assert_named_cases(movement, full["movement_by_id"])
    representatives = _representatives(movement, full["movement_by_id"])
    verdict = judge_truncated_hypothesis(
        truncated_hit10=int(truncated_headline["at_k"]["10"]["hit"]["numerator"]),
        full_hit10=int(full["headline"]["at_k"]["10"]["hit"]["numerator"]),
        recovered_of_eight=int(recovery["recovered_hit10"]),
        dense_only_retained=bool(retention["dense_only"]["retained_hit10"]),
        both_hit_lost=int(retention["both_hit"]["lost"]),
        prior_bm25_only_kept=int(recovery["previously_kept"]),
        prior_bm25_only_kept_still=int(recovery["previously_kept_still_hit10"]),
        bm25_mrr=float(bundle["bm25_headline"]["mrr"]["value"]),
        truncated_mrr=float(truncated_headline["mrr"]["value"]),
        bm25_ndcg=float(bundle["bm25_headline"]["at_k"]["10"]["ndcg_condensed"]["value"]),
        truncated_ndcg=float(truncated_headline["at_k"]["10"]["ndcg_condensed"]["value"]),
        bm25_recall5=float(bundle["bm25_headline"]["recall@5_macro"]["value"]),
        truncated_recall5=float(truncated_headline["recall@5_macro"]["value"]),
        truncated_requirement_recall=int(
            truncated_headline["requirement_recall@10"]["numerator"]
        ),
        truncated_all_requirements=int(
            truncated_headline["all_requirements_covered@10"]["numerator"]
        ),
    )
    coverage = _coverage_audit(
        bundle["dense_headline"],
        bundle["bm25_headline"],
        full["headline"],
        truncated_headline,
    )
    overlap = _overlap_summary(movement)
    revision = git_revision(root)
    audit = {
        "implementation": implementation_audit(),
        "reproduction": {
            "retrieval_reexecuted": False,
            "dense_top20_identity_questions": 45,
            "bm25_top20_identity_questions": 45,
            "identity_pairs": "complementarity artifact vs full-RRF copy",
            "mismatches": [],
            "fusion_scored": True,
            "formula_checks": formula_checks,
            "max_contributing_rank": max_rank_used,
            "deep_tail_ranks_used": 0,
        },
        "input_artifacts": {
            "dense_dev_sha256": bundle["dense_sha256"],
            "bm25_dev_sha256": bundle["bm25_sha256"],
            "full_rrf_run": FULL_RUN_DIR,
            "full_rrf_dev_sha256": _sha_file(full_dir / "rrf_dev.json"),
        },
        "test_questions_searched": 0,
        "git": revision,
        "intended_base_commit": INTENDED_BASE,
        "base_commit_confirmed": revision.get("git_commit") == INTENDED_BASE,
        "head_subject": _head_subject(root),
    }
    payload: dict[str, Any] = {
        "experiment_id": "v21_dev_top20_truncated_rrf",
        "dataset_id": DATASET_ID,
        "dataset_sha256": bundle["frozen"]["dataset_sha256"],
        "population": "dev fully_answerable",
        "question_count": 45,
        "test_questions_searched": 0,
        "dense_headline": bundle["dense_headline"],
        "bm25_headline": bundle["bm25_headline"],
        "full_rrf_headline": full["headline"],
        "truncated_headline": truncated_headline,
        "retention": retention,
        "recovery": recovery,
        "movement_summary": _movement_summary(movement),
        "requirement_transitions": {
            "cells": transitions["cells"],
            "full_rrf_cells": full["requirement_cells"],
            "truncated_requirement_recall@10": truncated_headline["requirement_recall@10"],
            "truncated_all_requirements_covered@10": truncated_headline[
                "all_requirements_covered@10"
            ],
        },
        "judgment_coverage": coverage,
        "overlap": overlap,
        "representatives": representatives,
        "verdict": verdict,
        "audit": audit,
        "named_case_ids": [item["query_id"] for item in NAMED_CASES]
        + [DENSE_ONLY_CASE["query_id"], *BOTH_MISS],
    }
    payload["next_hypothesis"] = _next_hypothesis(payload)
    payload["requirement_rows"] = transitions["requirements"]
    payload["movement"] = movement
    payload["truncated_questions"] = scored_rows
    _write_run(root, payload, bundle, full_dir)
    payload["run_dir"] = str(output)
    return payload


def ranking_mismatches(bundle: dict[str, Any], full_dir: Path) -> list[dict[str, Any]]:
    """Compare stored top-20 lists. A difference blocks fusion."""
    if not full_dir.is_dir():
        return [{"query_id": "", "label": "full_rrf_run", "reason": "missing run directory"}]
    found: list[dict[str, Any]] = []
    config = _read(full_dir / "experiment_config.json")
    if config.get("dataset_sha256") != bundle["frozen"]["dataset_sha256"]:
        found.append(
            {
                "query_id": "",
                "label": "dataset",
                "reason": "full-RRF dataset hash differs from v2.1",
            }
        )
    if config.get("input_dense_sha256") != bundle["dense_sha256"]:
        found.append(
            {
                "query_id": "",
                "label": "dense_sha256",
                "reason": "full-RRF input dense hash differs from complementarity",
            }
        )
    if config.get("input_bm25_sha256") != bundle["bm25_sha256"]:
        found.append(
            {
                "query_id": "",
                "label": "bm25_sha256",
                "reason": "full-RRF input BM25 hash differs from complementarity",
            }
        )
    copied = {
        "dense": _read(full_dir / "dense_dev.json"),
        "bm25": _read(full_dir / "bm25_dev.json"),
    }
    if _sha_file(full_dir / "dense_dev.json") != bundle["dense_sha256"]:
        found.append(
            {
                "query_id": "",
                "label": "dense_copy",
                "reason": "full-RRF dense_dev.json is not the complementarity file",
            }
        )
    if _sha_file(full_dir / "bm25_dev.json") != bundle["bm25_sha256"]:
        found.append(
            {
                "query_id": "",
                "label": "bm25_copy",
                "reason": "full-RRF bm25_dev.json is not the complementarity file",
            }
        )
    for label, source_key in (("dense", "dense_by_id"), ("bm25", "bm25_by_id")):
        copy_by_id = {str(row["query_id"]): row for row in copied[label]["questions"]}
        if set(copy_by_id) != set(bundle[source_key]):
            found.append(
                {
                    "query_id": "",
                    "label": label,
                    "reason": "question ids differ",
                }
            )
            continue
        for query_id, stored in bundle[source_key].items():
            mismatch = _prefix_diff(query_id, label, stored["hits"], copy_by_id[query_id]["hits"])
            if mismatch is not None:
                found.append(mismatch)
    return found


def _load_full_rrf(full_dir: Path) -> dict[str, Any]:
    headline_doc = _read(full_dir / "rrf_dev.json")
    headline = headline_doc["headline"]
    for key, expected in PINNED_FULL_HITS.items():
        got = int(headline["at_k"][key]["hit"]["numerator"])
        if got != expected:
            raise UsageError(f"stored full-RRF Hit@{key} is {got}, expected {expected}")
    requirement = headline["requirement_recall@10"]
    if (int(requirement["numerator"]), int(requirement["denominator"])) != PINNED_FULL_REQUIREMENT:
        raise UsageError("stored full-RRF requirement recall is not 46/58")
    covered = headline["all_requirements_covered@10"]
    if (
        int(covered["numerator"]),
        int(covered["denominator"]),
    ) != PINNED_FULL_ALL_REQUIREMENTS:
        raise UsageError("stored full-RRF all-requirements coverage is not 35/45")
    movement = _read(full_dir / "per_query_rank_movement.json")["queries"]
    by_id = {str(row["query_id"]): row for row in movement}
    if len(by_id) != 45:
        raise UsageError("full-RRF movement is not the DEV 45")
    lost = [
        query_id
        for query_id, row in by_id.items()
        if row["prior_cell"] == CELL_BM25 and not row["rrf_hit10"]
    ]
    if sorted(lost) != sorted(LOST_EIGHT):
        raise UsageError("full-RRF BM25-only losses are not the pinned 8 questions")
    both_lost = [
        query_id
        for query_id, row in by_id.items()
        if row["prior_cell"] == CELL_BOTH and not row["rrf_hit10"]
    ]
    if both_lost:
        raise UsageError("full-RRF both-hit losses are not empty")
    requirements = _read(full_dir / "requirement_analysis.json")
    cells = requirements["transitions"]["cells"]
    if int(cells["bm25_only"]["transitions"].get("lost", 0)) != 8:
        raise UsageError("full-RRF BM25-only requirement losses are not 8")
    return {
        "headline": headline,
        "questions": headline_doc["questions"],
        "movement_by_id": by_id,
        "requirement_cells": cells,
    }


def _movement_record(
    question: dict[str, Any],
    prior: dict[str, Any],
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    full_row: dict[str, Any],
    scored: dict[str, Any],
    fused: list[dict[str, Any]],
) -> dict[str, Any]:
    by_unit = {str(hit["retrieval_unit_id"]): hit for hit in fused}
    dense_ids = {str(hit["retrieval_unit_id"]) for hit in dense_hits[:CANDIDATE_N]}
    bm25_ids = {str(hit["retrieval_unit_id"]) for hit in bm25_hits[:CANDIDATE_N]}
    bm25_unit = _first_relevant_unit(bm25_hits)
    dense_unit = _first_relevant_unit(dense_hits)
    truncated_rank = first_relevant_rank(scored)
    full_rank = full_row["rrf_first_relevant_rank"]
    bm25_rank = prior["bm25_first_relevant_rank"]
    dense_rank = prior["dense_first_relevant_rank"]
    in_dense = None if bm25_unit is None else bm25_unit in dense_ids
    fused_bm25 = None if bm25_unit is None else by_unit.get(bm25_unit)
    dual_ahead = 0
    if fused_bm25 is not None:
        position = int(fused_bm25["rank"])
        dual_ahead = sum(
            1
            for hit in fused
            if int(hit["rank"]) < position and hit["in_dense_top_n"] and hit["in_bm25_top_n"]
        )
    full_hit = bool(full_row["rrf_hit10"])
    truncated_hit = truncated_rank is not None and int(truncated_rank) <= HIT_K
    top3 = []
    for hit in scored["hits"][:3]:
        unit_id = str(hit["retrieval_unit_id"])
        fused_hit = by_unit[unit_id]
        top3.append(
            {
                "rank": int(hit["rank"]),
                "retrieval_unit_id": unit_id,
                "relevance": hit["relevance"],
                "dense_rank": fused_hit["dense_rank"],
                "bm25_rank": fused_hit["bm25_rank"],
                "dense_contribution": fused_hit["dense_contribution"],
                "bm25_contribution": fused_hit["bm25_contribution"],
                "score": fused_hit["score"],
            }
        )
    return {
        "query_id": question["query_id"],
        "company": question.get("company_name"),
        "question": question.get("question"),
        "prior_cell": prior["cell"],
        "dense_first_relevant_rank": dense_rank,
        "bm25_first_relevant_rank": bm25_rank,
        "dense_first_unit_id": dense_unit,
        "bm25_first_unit_id": bm25_unit,
        "bm25_unit_in_dense_top20": in_dense,
        "dense_unit_in_bm25_top20": None if dense_unit is None else dense_unit in bm25_ids,
        "full_rrf_first_relevant_rank": full_rank,
        "truncated_first_relevant_rank": truncated_rank,
        "full_rrf_hit10": full_hit,
        "truncated_hit10": truncated_hit,
        "hit10_recovered_vs_full_rrf": bool(truncated_hit and not full_hit),
        "rank_delta_vs_bm25": rank_delta(bm25_rank, truncated_rank),
        "rank_delta_vs_full_rrf": rank_delta(full_rank, truncated_rank),
        "movement_vs_bm25": movement_label(bm25_rank, truncated_rank),
        "truncated_relevant_in_top5": len(relevant_ids_at(scored, 5)),
        "truncated_relevant_in_top10": len(relevant_ids_at(scored, HIT_K)),
        "intersection_count": len(dense_ids & bm25_ids),
        "union_count": len(fused),
        "dual_listed_ahead_of_bm25_unit": None if bm25_unit is None else dual_ahead,
        "bm25_unit_truncated_rank": None if fused_bm25 is None else int(fused_bm25["rank"]),
        "bm25_unit_dense_contribution": None
        if fused_bm25 is None
        else float(fused_bm25["dense_contribution"]),
        "bm25_unit_bm25_contribution": None
        if fused_bm25 is None
        else float(fused_bm25["bm25_contribution"]),
        "bm25_unit_score": None if fused_bm25 is None else float(fused_bm25["score"]),
        "bm25_unit_dense_rank_used": None if fused_bm25 is None else fused_bm25["dense_rank"],
        "excluded_dense_full_rank": full_row.get("rrf_first_unit_dense_rank"),
        "excluded_bm25_full_rank": full_row.get("rrf_first_unit_bm25_rank"),
        "full_rrf_top1": (full_row.get("rrf_top3") or [None])[0],
        "truncated_top3": top3,
        "unjudged_at_10": int(scored["at_k"]["10"]["unjudged_count"]),
    }


def _retention(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prior_cell"])].append(row)
    both_lost = [row["query_id"] for row in grouped[CELL_BOTH] if not row["truncated_hit10"]]
    bm25_lost = [row["query_id"] for row in grouped[CELL_BM25] if not row["truncated_hit10"]]
    dense_kept = [row["query_id"] for row in grouped[CELL_DENSE] if row["truncated_hit10"]]
    miss_hit = [row["query_id"] for row in grouped[CELL_MISS] if row["truncated_hit10"]]
    return {
        "both_hit": {
            "prior": len(grouped[CELL_BOTH]),
            "truncated_hit10": len(grouped[CELL_BOTH]) - len(both_lost),
            "lost": len(both_lost),
            "lost_query_ids": both_lost,
        },
        "bm25_only": {
            "prior": len(grouped[CELL_BM25]),
            "retained": len(grouped[CELL_BM25]) - len(bm25_lost),
            "lost": len(bm25_lost),
            "lost_query_ids": bm25_lost,
        },
        "dense_only": {
            "prior": len(grouped[CELL_DENSE]),
            "retained_hit10": len(dense_kept) == len(grouped[CELL_DENSE]) and bool(dense_kept),
            "hit10": len(dense_kept),
            "query_ids": dense_kept,
        },
        "both_miss": {
            "prior": len(grouped[CELL_MISS]),
            "truncated_hit10": len(miss_hit),
            "query_ids": miss_hit,
            "interpretation": (
                "이수페타시스 cross-filing과 현금흐름 period-scope 문항은 "
                "truncated RRF의 성공 조건이 아니다. Hit@10에 들어와도 "
                "period, scope, cross-filing 문제가 해결된 것이 아니다."
            ),
        },
    }


def _recovery(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["query_id"]: row for row in rows}
    lost_rows = [by_id[query_id] for query_id in LOST_EIGHT]
    recovered = [row["query_id"] for row in lost_rows if row["truncated_hit10"]]
    still = [row["query_id"] for row in lost_rows if not row["truncated_hit10"]]
    kept_ids = [
        row["query_id"]
        for row in rows
        if row["prior_cell"] == CELL_BM25 and row["query_id"] not in set(LOST_EIGHT)
    ]
    kept_still = [query_id for query_id in kept_ids if by_id[query_id]["truncated_hit10"]]
    buried = [
        row["query_id"]
        for row in lost_rows
        if not row["truncated_hit10"]
        and row["bm25_unit_truncated_rank"] is not None
        and int(row["dual_listed_ahead_of_bm25_unit"]) >= HIT_K
    ]
    return {
        "population": "full-RRF BM25-only Hit@10 losses",
        "prior_lost": len(LOST_EIGHT),
        "recovered_hit10": len(recovered),
        "still_lost": len(still),
        "recovered_query_ids": recovered,
        "still_lost_query_ids": still,
        "buried_by_intersection": buried,
        "previously_kept": len(kept_ids),
        "previously_kept_query_ids": kept_ids,
        "previously_kept_still_hit10": len(kept_still),
        "queries": [_recovery_view(row) for row in lost_rows],
        "previously_kept_queries": [_recovery_view(by_id[query_id]) for query_id in kept_ids],
    }


def _recovery_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": row["query_id"],
        "company": row["company"],
        "question": row["question"],
        "bm25_first_relevant_rank": row["bm25_first_relevant_rank"],
        "bm25_unit_in_dense_top20": row["bm25_unit_in_dense_top20"],
        "excluded_dense_full_rank": row["excluded_dense_full_rank"],
        "dense_contribution": row["bm25_unit_dense_contribution"],
        "bm25_contribution": row["bm25_unit_bm25_contribution"],
        "truncated_score": row["bm25_unit_score"],
        "full_rrf_rank": row["full_rrf_first_relevant_rank"],
        "truncated_rank": row["bm25_unit_truncated_rank"],
        "truncated_first_relevant_rank": row["truncated_first_relevant_rank"],
        "hit10_recovered": row["hit10_recovered_vs_full_rrf"],
        "truncated_hit10": row["truncated_hit10"],
        "intersection_count": row["intersection_count"],
        "dual_listed_ahead": row["dual_listed_ahead_of_bm25_unit"],
        "deep_tail_rank_used_in_score": False,
    }


def _assert_named_cases(
    rows: list[dict[str, Any]],
    full_by_id: dict[str, dict[str, Any]],
) -> None:
    by_id = {row["query_id"]: row for row in rows}
    for spec in NAMED_CASES:
        row = by_id[spec["query_id"]]
        full = full_by_id[spec["query_id"]]
        if int(row["bm25_first_relevant_rank"]) != int(spec["bm25_rank"]):
            raise UsageError(f"{spec['query_id']} BM25 rank is not the pinned value")
        if int(full["rrf_first_relevant_rank"]) != int(spec["full_rrf_rank"]):
            raise UsageError(f"{spec['query_id']} full-RRF rank is not the pinned value")
        if int(full["rrf_first_unit_dense_rank"]) != int(spec["excluded_dense_rank"]):
            raise UsageError(f"{spec['query_id']} excluded Dense rank is not the pinned value")
        if row["bm25_unit_in_dense_top20"]:
            raise UsageError(f"{spec['query_id']} BM25 unit is inside Dense top-20")
        if row["bm25_unit_dense_contribution"] != 0:
            raise UsageError(f"{spec['query_id']} received a Dense deep-tail contribution")
        if int(spec["excluded_dense_rank"]) <= CANDIDATE_N:
            raise UsageError(f"{spec['query_id']} pinned Dense rank is inside top-20")
    semantic = by_id[DENSE_ONLY_CASE["query_id"]]
    full_semantic = full_by_id[DENSE_ONLY_CASE["query_id"]]
    if int(semantic["dense_first_relevant_rank"]) != int(DENSE_ONLY_CASE["dense_rank"]):
        raise UsageError("Dense-only case is not Dense rank 1")
    if int(full_semantic["rrf_first_unit_bm25_rank"]) != int(DENSE_ONLY_CASE["excluded_bm25_rank"]):
        raise UsageError("Dense-only excluded BM25 rank is not 75")
    if semantic["dense_unit_in_bm25_top20"]:
        raise UsageError("Dense-only unit is inside BM25 top-20")
    if int(full_semantic["rrf_first_relevant_rank"]) != int(DENSE_ONLY_CASE["full_rrf_rank"]):
        raise UsageError("Dense-only full-RRF rank is not 6")


def _representatives(
    rows: list[dict[str, Any]],
    full_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_id = {row["query_id"]: row for row in rows}
    improvements = [row for row in rows if row["movement_vs_bm25"] == "improvement"]
    improvements.sort(
        key=lambda row: (-(row["rank_delta_vs_bm25"] or 0), row["query_id"])
    )
    regressions = [row for row in rows if row["movement_vs_bm25"] == "regression"]
    regressions.sort(
        key=lambda row: (
            0 if row["bm25_first_relevant_rank"] and not row["truncated_hit10"] else 1,
            row["rank_delta_vs_bm25"] if row["rank_delta_vs_bm25"] is not None else 0,
            row["query_id"],
        )
    )
    return {
        "selection_rule": (
            "Improvements are the largest positive first-relevant rank deltas "
            "against BM25. Regressions prefer a lost BM25 Hit@10, then the most "
            "negative delta. Named cases are always listed separately."
        ),
        "improvements": [_case_view(row, full_by_id[row["query_id"]]) for row in improvements[:3]],
        "regressions": [_case_view(row, full_by_id[row["query_id"]]) for row in regressions[:3]],
        "named_bm25_regressions": [
            _case_view(by_id[spec["query_id"]], full_by_id[spec["query_id"]])
            for spec in NAMED_CASES
        ],
        "dense_only_semantic": _case_view(
            by_id[DENSE_ONLY_CASE["query_id"]],
            full_by_id[DENSE_ONLY_CASE["query_id"]],
        ),
        "both_miss": [
            _case_view(by_id[query_id], full_by_id[query_id]) for query_id in BOTH_MISS
        ],
    }


def _case_view(row: dict[str, Any], full_row: dict[str, Any]) -> dict[str, Any]:
    top1 = row["full_rrf_top1"]
    return {
        "query_id": row["query_id"],
        "company": row["company"],
        "question": row["question"],
        "prior_cell": row["prior_cell"],
        "bm25_first_relevant_rank": row["bm25_first_relevant_rank"],
        "dense_first_relevant_rank": row["dense_first_relevant_rank"],
        "bm25_unit_in_dense_top20": row["bm25_unit_in_dense_top20"],
        "dense_unit_in_bm25_top20": row["dense_unit_in_bm25_top20"],
        "excluded_dense_full_rank": row["excluded_dense_full_rank"],
        "excluded_bm25_full_rank": row["excluded_bm25_full_rank"],
        "dense_contribution_on_bm25_unit": row["bm25_unit_dense_contribution"],
        "bm25_contribution_on_bm25_unit": row["bm25_unit_bm25_contribution"],
        "truncated_score_on_bm25_unit": row["bm25_unit_score"],
        "full_rrf_rank": row["full_rrf_first_relevant_rank"],
        "truncated_rank": row["truncated_first_relevant_rank"],
        "bm25_unit_truncated_rank": row["bm25_unit_truncated_rank"],
        "truncated_hit10": row["truncated_hit10"],
        "hit10_recovered": row["hit10_recovered_vs_full_rrf"],
        "intersection_count": row["intersection_count"],
        "dual_listed_ahead_of_bm25_unit": row["dual_listed_ahead_of_bm25_unit"],
        "movement_vs_bm25": row["movement_vs_bm25"],
        "truncated_top3": row["truncated_top3"],
        "full_rrf_top1": top1,
        "full_rrf_top1_preview": None if not top1 else top1.get("preview"),
        "relevant_preview": full_row.get("rrf_first_relevant_preview"),
        "deep_tail_rank_used_in_truncated_score": False,
    }


def _requirement_transitions(
    prior_rows: list[dict[str, Any]],
    covered_by_query: dict[str, set[str]],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prior_rows:
        covered = row["requirement_id"] in covered_by_query.get(row["query_id"], set())
        cell = str(row["cell"])
        groups[cell].append(
            {
                "query_id": row["query_id"],
                "requirement_id": row["requirement_id"],
                "requirement_text": row.get("requirement_text") or "",
                "prior_cell": cell,
                "truncated_covered@10": covered,
                "transition": _requirement_transition(cell, covered),
            }
        )
    summary = {}
    for cell in ("both", "dense_only", "bm25_only", "neither"):
        items = groups.get(cell, [])
        counts = Counter(item["transition"] for item in items)
        summary[cell] = {
            "prior": len(items),
            "transitions": {name: int(counts.get(name, 0)) for name in sorted(counts)},
            "lost_or_still_missing": [
                f"{item['query_id']}/{item['requirement_id']}"
                for item in items
                if item["transition"] in {"lost", "still_missing"}
            ],
        }
    rows = [item for cell in groups.values() for item in cell]
    rows.sort(key=lambda item: (item["query_id"], item["requirement_id"]))
    return {"cells": summary, "requirements": rows}


def _requirement_transition(cell: str, covered: bool) -> str:
    if cell == "neither":
        return "recovered" if covered else "still_missing"
    if cell == "dense_only":
        return "recovered" if covered else "lost"
    if cell in {"both", "bm25_only"}:
        return "retained" if covered else "lost"
    raise UsageError(f"unknown requirement cell {cell}")


def _coverage_audit(
    dense: dict[str, Any],
    bm25: dict[str, Any],
    full: dict[str, Any],
    truncated: dict[str, Any],
) -> dict[str, Any]:
    return {
        "unjudged_is_not_nonrelevant": True,
        "ndcg_zero_filled_is_sensitivity_only": True,
        "systems": {
            "dense": _coverage_view(dense),
            "bm25": _coverage_view(bm25),
            "full_rrf": _coverage_view(full),
            "truncated_rrf": _coverage_view(truncated),
        },
    }


def _coverage_view(headline: dict[str, Any]) -> dict[str, Any]:
    return {
        "judged_coverage@10": headline["at_k"]["10"]["judged_coverage"]["value"],
        "judged_coverage@20": headline["judged_coverage@20"]["value"],
        "mean_unjudged@10": headline["at_k"]["10"]["unjudged_count_mean"]["value"],
        "ndcg@10_condensed": headline["at_k"]["10"]["ndcg_condensed"]["value"],
        "ndcg@10_zero_filled": headline["at_k"]["10"]["ndcg_unjudged_as_zero"]["value"],
    }


def _overlap_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [int(row["intersection_count"]) for row in rows]
    return {
        "definition": "retrieval_unit_id intersection of Dense top-20 and BM25 top-20",
        "min": min(counts),
        "median": _median(counts),
        "max": max(counts),
        "mean": sum(counts) / len(counts),
        "questions_with_intersection_at_least_10": sum(1 for value in counts if value >= 10),
    }


def _movement_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["movement_vs_bm25"] for row in rows)
    deltas = [
        int(row["rank_delta_vs_bm25"])
        for row in rows
        if row["rank_delta_vs_bm25"] is not None
    ]
    return {
        "counts": {name: int(counts.get(name, 0)) for name in sorted(counts)},
        "rank_delta_vs_bm25_defined": len(deltas),
        "rank_delta_vs_bm25_mean": None if not deltas else sum(deltas) / len(deltas),
        "rank_delta_vs_bm25_median": _median(deltas),
        "positive_delta_means": "truncated first-relevant rank is earlier than BM25",
    }


def _next_hypothesis(payload: dict[str, Any]) -> str:
    if payload["verdict"]["label"] == "A":
        return (
            "이 Top-20 equal-weight k=60 RRF를 v2.1 TEST에서 한 번 확인할 가설이 다음이고, "
            "both-miss 2건은 period, scope, cross-filing 가설로 따로 남긴다."
        )
    still = [row for row in payload["recovery"]["queries"] if not row["truncated_hit10"]]
    if not still:
        return "Hit@10 잔여 손실의 질의별 순위 구성을 다음 가설로 적는다."
    detail = ", ".join(
        f"BM25 {row['bm25_first_relevant_rank']}위가 truncated {row['truncated_rank']}위"
        for row in still
    )
    return (
        "미회복 문항은 BM25 단독 순위가 교집합과 Dense 단독의 더 앞선 등위 뒤에 놓인 "
        f"{detail}이다. 이 등위 결합이 Hit@10 잔여 손실의 원인인지가 다음 가설이다."
    )


def _write_run(
    root: Path,
    payload: dict[str, Any],
    bundle: dict[str, Any],
    full_dir: Path,
) -> None:
    output = root / "evaluation" / "runs" / RUN_DIR_NAME
    if dataset_dir(root) in output.parents:
        raise UsageError("refusing to write the truncated run into the dataset")
    output.mkdir(parents=True, exist_ok=True)
    mismatch = output / "reproduction_mismatch.json"
    if mismatch.exists():
        mismatch.unlink()
    shutil.copyfile(bundle["source_dir"] / "dense_dev.json", output / "dense_dev.json")
    shutil.copyfile(bundle["source_dir"] / "bm25_dev.json", output / "bm25_dev.json")
    shutil.copyfile(full_dir / "rrf_dev.json", output / "full_rrf_dev.json")
    _dump(output / "experiment_config.json", _config(payload))
    _dump(
        output / "truncated_rrf_dev.json",
        {
            "headline": payload["truncated_headline"],
            "questions": payload["truncated_questions"],
        },
    )
    _dump(output / "headline_comparison.json", _headline_document(payload))
    _dump(output / "per_query_rank_movement.json", {"queries": payload["movement"]})
    _write_csv(output / "per_query_rank_movement.csv", payload["movement"])
    _dump(output / "regression_recovery.json", payload["recovery"])
    _dump(
        output / "requirement_transition.json",
        {
            "full_rrf": payload["requirement_transitions"]["full_rrf_cells"],
            "truncated_rrf": payload["requirement_transitions"]["cells"],
            "requirements": payload["requirement_rows"],
        },
    )
    _dump(output / "judgment_coverage_audit.json", payload["judgment_coverage"])
    _dump(output / "representative_cases.json", payload["representatives"])
    report = render_report(payload)
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "portfolio_summary.md").write_text(render_portfolio(payload), encoding="utf-8")


def render_report(payload: dict[str, Any]) -> str:
    """Write the DEV report for this one configuration."""
    audit = payload["audit"]
    verdict = payload["verdict"]
    sections = [
        "# v2.1 DEV candidate-truncated RRF",
        "",
        "## 1. Executive summary",
        "",
        _executive(payload),
        "",
        "## 2. Experiment commit / base",
        "",
        (
            f"기준 commit은 `{audit['intended_base_commit']}` "
            f"({audit['base_commit_confirmed']})이다. "
            f"실행 시점 HEAD는 `{audit['git']['git_commit']}` "
            f"({audit['head_subject']}), dirty={audit['git']['git_dirty']}이다. "
            "이 실행은 그 기준 위에서 stored ranking만 읽었다."
        ),
        "",
        "## 3. Hypothesis",
        "",
        (
            "각 retriever의 deep-tail 전체 순위를 fusion signal로 쓴 것이 "
            "full-ranking RRF의 문제였다. 실제 반환 Top-20에만 RRF contribution을 주면 "
            "BM25의 lexical evidence를 덜 훼손하고 Dense의 상위 semantic signal을 "
            "보존할 수 있다."
        ),
        "",
        "## 4. Experiment invariants",
        "",
        (
            f"dataset `{payload['dataset_id']}` sha256 `{payload['dataset_sha256']}`. "
            "모집단은 DEV fully_answerable 45문항이다. TEST 검색 수는 0이다. "
            "N=20, k=60, Dense weight 1, BM25 weight 1이다. "
            "N sweep, k sweep, weight, alpha, query routing, reranker, "
            "query expansion, embedding, chunking, v2, v2.1 질문, qrel, split, "
            "judgment는 바꾸지 않았다. 바꾼 변수는 full-ranking RRF에서 "
            "candidate-truncated RRF로의 후보 집합이다."
        ),
        "",
        "## 5. Truncated-RRF semantics",
        "",
        _semantics_prose(),
        "",
        "## 6. Implementation audit",
        "",
        _audit_prose(audit),
        "",
        "## 7. Dense / BM25 / Full-RRF / Truncated-RRF",
        "",
        (
            "A. Dense-D2a는 저장된 제품 경로 순위다. "
            f"모델 `{DENSE_MODEL_ID}`, 질의 원문, 지시문 없음. "
            "B. BM25는 저장된 `Bm25Index.search` top-20이다. "
            f"k1={BM25_PARAMS['k1']} b={BM25_PARAMS['b']} tokenizer `{TOKENIZER_VERSION}`. "
            "C. full-ranking RRF는 직전 `rrf_dev.json`이다. "
            "k=60이고 eligible unit 전체 순위를 썼다. "
            "D. truncated RRF는 A와 B의 top-20 union만 다시 점수한 것이다. "
            "retrieval은 재실행하지 않았다. complementarity 파일과 full-RRF 사본의 "
            "top-20은 45/45 같았다."
        ),
        "",
        "## 8. Headline metrics",
        "",
        _markdown_table(payload),
        "",
        "## 9. BM25-only regression recovery",
        "",
        _recovery_prose(payload),
        "",
        _recovery_table(payload["recovery"]["queries"]),
        "",
        _named_prose(payload["representatives"]["named_bm25_regressions"]),
        "",
        "## 10. Dense-only semantic retention",
        "",
        _semantic_prose(payload["representatives"]["dense_only_semantic"]),
        "",
        "## 11. Both-hit retention",
        "",
        _both_hit_prose(payload),
        "",
        "## 12. Both-miss",
        "",
        payload["retention"]["both_miss"]["interpretation"],
        "",
        _both_miss_prose(payload["representatives"]["both_miss"]),
        "",
        "## 13. Rank movement",
        "",
        _movement_prose(payload),
        "",
        "## 14. Requirement transition",
        "",
        _requirement_prose(payload),
        "",
        "## 15. Judgment coverage",
        "",
        _coverage_prose(payload),
        "",
        "## 16. Representative improvements",
        "",
        _case_lines(payload["representatives"]["improvements"]),
        "",
        "## 17. Representative regressions",
        "",
        _case_lines(payload["representatives"]["regressions"]),
        "",
        "## 18. FACT",
        "",
        _fact(payload),
        "",
        "## 19. INTERPRETATION",
        "",
        _interpretation(payload),
        "",
        "## 20. Limitations",
        "",
        _limitations(),
        "",
        "## 21. Hypothesis verdict",
        "",
        f"판정 {verdict['label']} ({verdict['decision']}). {verdict['rule']}",
        "",
        _verdict_prose(verdict),
        "",
        "## 22. Recommended next hypothesis",
        "",
        payload["next_hypothesis"],
        "",
        "## 23. Portfolio Summary",
        "",
        render_portfolio(payload).split("## Portfolio Summary\n", 1)[-1],
        "",
    ]
    return "\n".join(sections)


def render_portfolio(payload: dict[str, Any]) -> str:
    verdict = payload["verdict"]
    truncated = payload["truncated_headline"]
    recovery = payload["recovery"]
    dense_only = payload["retention"]["dense_only"]
    current = {
        "experiment": "Candidate-truncated RRF",
        "hypothesis": (
            "Top-20 밖 deep-tail 순위를 빼면 BM25 lexical Hit@10을 지키면서 "
            "Dense의 상위 semantic signal을 더할 수 있다."
        ),
        "success": (
            "BM25 Hit@10 41/45 이상, Dense-only 유지, full-RRF가 잃은 "
            "BM25-only 8건의 대부분 회복, MRR/NDCG/Recall@5 하락 0.05 이내, "
            "requirement가 BM25 근처."
        ),
        "result": (
            f"Hit@10 {_count(truncated['at_k']['10']['hit'])}, "
            f"BM25-only 8건 중 회복 {recovery['recovered_hit10']}, "
            f"Dense-only Hit@10 {dense_only['hit10']}/{dense_only['prior']}, "
            f"MRR {_num(truncated['mrr']['value'])}."
        ),
        "verdict": verdict["decision"],
        "next": payload["next_hypothesis"],
    }
    rows = [_portfolio_history(), current]
    lines = [
        "## Portfolio Summary",
        "",
        "### Hypothesis",
        "",
        current["hypothesis"],
        "",
        "### Success Condition",
        "",
        "- BM25 Hit@10 42/45를 유지하거나 41/45 이상.",
        "- 삼성전자 구매/매입 Dense-only 문항이 Hit@10에 남는다.",
        "- full-RRF가 잃은 BM25-only 8건의 대부분을 Hit@10으로 되돌린다.",
        "- MRR, NDCG@10, Recall@5 macro가 BM25보다 0.05를 넘겨 떨어지지 않는다.",
        "- requirement recall과 all-requirements covered가 BM25 근처에 남는다.",
        "",
        "### Result",
        "",
        current["result"],
        "",
        "### Why It Worked / Failed",
        "",
        _why(payload),
        "",
        "### Decision",
        "",
        verdict["decision"],
        "",
        "### Next Hypothesis",
        "",
        payload["next_hypothesis"],
        "",
        "| Experiment | Hypothesis | Success condition | Result | Verdict | Next |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if isinstance(row, list):
            lines.extend(row)
        else:
            lines.append(_portfolio_line(row))
    lines.append("")
    return "\n".join(lines)


def _portfolio_history() -> list[str]:
    history = (
        (
            "Dense-D2a baseline",
            "v2.1 DEV에서 Dense-D2a 단독 순위를 측정한다.",
            "DEV fully_answerable 45의 Hit, Recall, MRR, NDCG를 고정 산출물로 남긴다.",
            "Hit@10 33/45, MRR 0.525, NDCG@10 0.478, requirement recall 41/58.",
            "기준선 확정",
            "BM25와 실패가 겹치는지 본다.",
        ),
        (
            "Dense vs BM25 complementarity",
            "Dense와 BM25의 Hit@10 실패가 서로 다른가.",
            "BM25가 Dense 실패를 복구하고 both-miss가 비대칭 셀보다 작다.",
            "both 32, Dense only 1, BM25 only 10, both miss 2. BM25 Hit@10 42/45.",
            "상보성 관측",
            "전체 순위 RRF가 그 차이를 순위로 옮기는지 본다.",
        ),
        (
            "Full-ranking RRF",
            "k=60 전체 순위 RRF가 BM25 coverage를 유지한 채 Dense-only를 복구하는가.",
            "Hit@10 감소 1 이하, BM25-only 손실 1 이하, Dense-only 복구 또는 순위 품질 개선.",
            "Hit@10 36/45, BM25-only 2/10 유지, Dense-only 1/1, MRR 0.686.",
            "기각",
            "Top-20 밖 순위를 fusion에서 뺀다.",
        ),
    )
    return [
        "| " + " | ".join(cell.replace("|", "/") for cell in item) + " |"
        for item in history
    ]


def _portfolio_line(row: dict[str, str]) -> str:
    cells = (
        row["experiment"],
        row["hypothesis"],
        row["success"],
        row["result"],
        row["verdict"],
        row["next"],
    )
    return "| " + " | ".join(item.replace("|", "/").replace("\n", " ") for item in cells) + " |"


def _semantics_prose() -> str:
    low = intersection_min_score()
    high = single_list_max_score()
    return " ".join(
        [
            "후보 집합은 Dense top-20과 BM25 top-20의 union이다.",
            "같은 retrieval_unit_id는 한 행이다.",
            "양쪽에 있으면 contribution을 더한다.",
            "한쪽에만 있으면 그 retriever의 contribution만 가진다.",
            "양쪽 top-20 밖은 후보가 아니다.",
            f"score(d) = I_dense(d)/({RRF_K}+dense_rank(d)) + "
            f"I_bm25(d)/({RRF_K}+bm25_rank(d)).",
            "I는 그 retriever의 top-20에 있으면 1이고 없으면 0이다.",
            "75, 283, 705 같은 full rank는 이 식의 입력이 아니다.",
            f"이 설정에서 교집합 최소 점수는 {low:.5f}이고 "
            f"단일 리스트 최대 점수는 {high:.5f}이다. 교집합 점수가 더 크다.",
            "그래서 양쪽 top-20에 있는 unit은 한쪽에만 있는 unit보다 항상 앞에 선다.",
            "동점 정렬은 score 내림차순, retrieval_unit_id 오름차순이다.",
        ]
    )


def _audit_prose(audit: dict[str, Any]) -> str:
    impl = audit["implementation"]
    repro = audit["reproduction"]
    return " ".join(
        [
            f"함수 `{impl['function']}`.",
            f"`{impl['full_ranking_function_not_used']}`는 호출하지 않았다.",
            f"N={impl['candidate_n']} k={impl['k']} weights {impl['weights']}.",
            f"식 `{impl['formula']}`.",
            f"tie-break: {', '.join(impl['tie_break'])}.",
            impl["tie_break_reason"],
            f"공식 대조 {repro['formula_checks']}건.",
            f"점수에 들어간 최대 rank {repro['max_contributing_rank']}.",
            f"deep-tail rank 사용 {repro['deep_tail_ranks_used']}.",
            "Dense top-20 일치 45/45, BM25 top-20 일치 45/45.",
            "불일치가 있으면 fusion 점수를 쓰지 않는다.",
        ]
    )


def _executive(payload: dict[str, Any]) -> str:
    truncated = payload["truncated_headline"]
    bm25 = payload["bm25_headline"]
    full = payload["full_rrf_headline"]
    verdict = payload["verdict"]
    recovery = payload["recovery"]
    dense_only = payload["retention"]["dense_only"]
    return (
        f"판정은 {verdict['label']} ({verdict['decision']})이다. "
        f"truncated RRF Hit@10은 {_count(truncated['at_k']['10']['hit'])}, "
        f"full-RRF는 {_count(full['at_k']['10']['hit'])}, "
        f"BM25는 {_count(bm25['at_k']['10']['hit'])}이다. "
        f"full-RRF가 잃은 BM25-only 8건 중 Hit@10 회복은 {recovery['recovered_hit10']}건이다. "
        f"Dense-only Hit@10은 {dense_only['hit10']}/{dense_only['prior']}이다. "
        f"MRR {_num(bm25['mrr']['value'])} → {_num(truncated['mrr']['value'])}, "
        f"NDCG@10 {_num(bm25['at_k']['10']['ndcg_condensed']['value'])} → "
        f"{_num(truncated['at_k']['10']['ndcg_condensed']['value'])}, "
        f"Recall@5 macro {_num(bm25['recall@5_macro']['value'])} → "
        f"{_num(truncated['recall@5_macro']['value'])}."
    )


def _recovery_prose(payload: dict[str, Any]) -> str:
    recovery = payload["recovery"]
    return (
        f"full-RRF BM25-only 손실 {recovery['prior_lost']}건 중 "
        f"truncated Hit@10 회복은 {recovery['recovered_hit10']}건, "
        f"잔존은 {recovery['still_lost']}건이다. "
        f"교집합이 10개 이상 앞에 있어 Hit@10 밖으로 남은 질의는 "
        f"{len(recovery['buried_by_intersection'])}건이다. "
        f"full-RRF가 이미 유지하던 BM25-only "
        f"{recovery['previously_kept']}건 중 truncated Hit@10은 "
        f"{recovery['previously_kept_still_hit10']}건이다. "
        "각 행의 Dense contribution은 top-20 밖 full rank를 쓰지 않은 값이다."
    )


def _recovery_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| query | BM25 rank | Dense top-20 | full RRF | truncated | recovered | intersection |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['query_id']}`",
                    str(row["bm25_first_relevant_rank"]),
                    str(row["bm25_unit_in_dense_top20"]),
                    str(row["full_rrf_rank"]),
                    str(row["truncated_rank"]),
                    str(row["hit10_recovered"]),
                    str(row["intersection_count"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _named_prose(rows: list[dict[str, Any]]) -> str:
    blocks = []
    for row in rows:
        blocks.append(
            "\n".join(
                [
                    f"### {row['query_id']}",
                    "",
                    f"{row['company']}: {row['question']}",
                    "",
                    (
                        f"BM25 relevant rank {row['bm25_first_relevant_rank']}. "
                        "그 unit은 Dense top-20에 "
                        f"{'있다' if row['bm25_unit_in_dense_top20'] else '없다'}. "
                        f"이전 full ranking의 Dense rank {row['excluded_dense_full_rank']}는 "
                        "truncated 점수에 넣지 않았다. "
                        f"Dense contribution {row['dense_contribution_on_bm25_unit']}, "
                        f"BM25 contribution {_fmt(row['bm25_contribution_on_bm25_unit'])}, "
                        f"truncated score {_fmt(row['truncated_score_on_bm25_unit'])}. "
                        f"그 unit의 truncated rank는 {row['bm25_unit_truncated_rank']}이고 "
                        f"full-RRF rank는 {row['full_rrf_rank']}이다. "
                        f"첫 양성 truncated rank는 {row['truncated_rank']}. "
                        f"교집합 {row['intersection_count']}개, "
                        f"그 앞에 있는 교집합 unit "
                        f"{row['dual_listed_ahead_of_bm25_unit']}개. "
                        f"Hit@10 회복 {row['hit10_recovered']}."
                    ),
                    "",
                    _old_winner_sentence(row),
                ]
            )
        )
    return "\n\n".join(blocks)


def _old_winner_sentence(row: dict[str, Any]) -> str:
    top = row.get("full_rrf_top1")
    if not top:
        return "이전 full-RRF 1위 기록이 없다."
    moved = next(
        (
            hit
            for hit in row.get("truncated_top3") or []
            if hit["retrieval_unit_id"] == top.get("retrieval_unit_id")
        ),
        None,
    )
    moved_text = ""
    if moved is not None:
        moved_text = (
            f" truncated top3 안에서는 rank {moved['rank']}, "
            f"Dense rank {moved['dense_rank']}, BM25 rank {moved['bm25_rank']}, "
            f"score {float(moved['score']):.6f}이다."
        )
    return (
        f"이전 full-RRF 1위 relevance {top.get('relevance')}는 "
        f"BM25 full rank {top.get('bm25_rank')}, Dense full rank {top.get('dense_rank')}였다. "
        "그 full rank 쌍은 truncated 점수의 입력이 아니다."
        f"{moved_text} "
        f"발췌: {top.get('preview') or ''}"
    )


def _semantic_prose(row: dict[str, Any]) -> str:
    kept = "Hit@10에 남았다" if row["truncated_hit10"] else "Hit@10 밖으로 나갔다"
    return " ".join(
        [
            f"`{row['query_id']}` {row['company']}: {row['question']}",
            f"Dense 첫 양성 rank {row['dense_first_relevant_rank']}.",
            "BM25 top-20에는 그 unit이 없다."
            if row["dense_unit_in_bm25_top20"] is False
            else "BM25 top-20에 그 unit이 있다.",
            f"이전 full ranking BM25 rank {row['excluded_bm25_full_rank']}는 점수에 넣지 않았다.",
            f"full-RRF rank {row['full_rrf_rank']}, truncated rank {row['truncated_rank']}.",
            f"이 문항은 {kept}.",
            f"교집합은 {row['intersection_count']}개다.",
            "BM25 lexical coverage를 지키는 동안 Dense rank 1 signal이 남는지가 이 사례다.",
            "이 한 문항의 성패로 Dense 전체를 지우거나 hybrid 전체를 확정하지 않는다.",
        ]
    )


def _both_hit_prose(payload: dict[str, Any]) -> str:
    both = payload["retention"]["both_hit"]
    return (
        f"이전 both-hit {both['prior']} 중 truncated Hit@10은 {both['truncated_hit10']}이다. "
        f"새로 잃은 질의는 {both['lost']}건{_id_suffix(both['lost_query_ids'])}. "
        "full-RRF에서는 32/32가 유지됐다."
    )


def _both_miss_prose(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        lines.append(
            f"- `{row['query_id']}` {row['company']}: "
            f"Dense {row['dense_first_relevant_rank']}, "
            f"BM25 {row['bm25_first_relevant_rank']}, "
            f"full-RRF {row['full_rrf_rank']}, "
            f"truncated {row['truncated_rank']}, "
            f"truncated Hit@10 {row['truncated_hit10']}. {row['question']}"
        )
    return "\n".join(lines)


def _movement_prose(payload: dict[str, Any]) -> str:
    summary = payload["movement_summary"]
    overlap = payload["overlap"]
    return (
        f"BM25 대비 movement `{json.dumps(summary['counts'], ensure_ascii=False)}`. "
        f"first-relevant delta 평균 {_num(summary['rank_delta_vs_bm25_mean'])}, "
        f"중앙값 {_num(summary['rank_delta_vs_bm25_median'])}. "
        f"top-20 교집합 크기는 최소 {overlap['min']}, 중앙값 {_num(overlap['median'])}, "
        f"최대 {overlap['max']}, 평균 {_num(overlap['mean'])}이다. "
        f"교집합이 10 이상인 질의는 {overlap['questions_with_intersection_at_least_10']}개다."
    )


def _requirement_prose(payload: dict[str, Any]) -> str:
    dense = payload["dense_headline"]
    bm25 = payload["bm25_headline"]
    full = payload["full_rrf_headline"]
    truncated = payload["truncated_headline"]
    cells = payload["requirement_transitions"]["cells"]
    full_cells = payload["requirement_transitions"]["full_rrf_cells"]
    return "\n".join(
        [
            (
                "requirement recall@10 "
                f"Dense {_count(dense['requirement_recall@10'])}, "
                f"BM25 {_count(bm25['requirement_recall@10'])}, "
                f"full-RRF {_count(full['requirement_recall@10'])}, "
                f"truncated {_count(truncated['requirement_recall@10'])}."
            ),
            (
                "all requirements covered@10 "
                f"Dense {_count(dense['all_requirements_covered@10'])}, "
                f"BM25 {_count(bm25['all_requirements_covered@10'])}, "
                f"full-RRF {_count(full['all_requirements_covered@10'])}, "
                f"truncated {_count(truncated['all_requirements_covered@10'])}."
            ),
            "",
            "full-RRF transition:",
            _transition_lines(full_cells),
            "",
            "truncated RRF transition:",
            _transition_lines(cells),
        ]
    )


def _coverage_prose(payload: dict[str, Any]) -> str:
    systems = payload["judgment_coverage"]["systems"]
    lines = [
        "unjudged는 nonrelevant가 아니다. zero-filled NDCG는 민감도 지표다.",
        "",
        "| system | judged@10 | judged@20 | mean unjudged@10 | NDCG condensed | NDCG zero-filled |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("dense", "bm25", "full_rrf", "truncated_rrf"):
        block = systems[name]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _num(block["judged_coverage@10"]),
                    _num(block["judged_coverage@20"]),
                    _num(block["mean_unjudged@10"]),
                    _num(block["ndcg@10_condensed"]),
                    _num(block["ndcg@10_zero_filled"]),
                ]
            )
            + " |"
        )
    truncated = systems["truncated_rrf"]
    full = systems["full_rrf"]
    lines.extend(
        [
            "",
            (
                f"truncated judged_coverage@10은 {_num(truncated['judged_coverage@10'])}이고 "
                f"mean unjudged@10은 {_num(truncated['mean_unjudged@10'])}이다. "
                f"full-RRF는 {_num(full['judged_coverage@10'])}, "
                f"{_num(full['mean_unjudged@10'])}이다. "
                "후보가 top-20 union으로 줄면 full-RRF가 끌어올리던 deep-tail unjudged가 "
                "상위 10에 덜 섞인다. 그 변화는 평가 확실성을 올린다. "
                "남은 unjudged를 오답으로 세지 않는다."
            ),
        ]
    )
    return "\n".join(lines)


def _case_lines(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "해당 사례가 없다."
    blocks = []
    for row in rows:
        top = row["truncated_top3"][0] if row["truncated_top3"] else None
        blocks.append(
            "\n".join(
                [
                    f"### {row['query_id']}",
                    "",
                    row["question"] or "",
                    "",
                    (
                        f"회사 {row['company']}. 이전 셀 {row['prior_cell']}. "
                        f"movement {row['movement_vs_bm25']}. "
                        f"Dense 첫 양성 {row['dense_first_relevant_rank']}, "
                        f"BM25 첫 양성 {row['bm25_first_relevant_rank']}, "
                        f"full-RRF {row['full_rrf_rank']}, "
                        f"truncated {row['truncated_rank']}. "
                        f"교집합 {row['intersection_count']}."
                    ),
                    "",
                    (
                        "truncated 1위 "
                        f"relevance {None if top is None else top.get('relevance')}, "
                        f"Dense rank {None if top is None else top.get('dense_rank')}, "
                        f"BM25 rank {None if top is None else top.get('bm25_rank')}, "
                        f"score {_fmt(None if top is None else top.get('score'))}."
                    ),
                ]
            )
        )
    return "\n\n".join(blocks)


def _fact(payload: dict[str, Any]) -> str:
    truncated = payload["truncated_headline"]
    retention = payload["retention"]
    recovery = payload["recovery"]
    return " ".join(
        [
            f"truncated Hit@1 {_count(truncated['at_k']['1']['hit'])}, "
            f"Hit@5 {_count(truncated['at_k']['5']['hit'])}, "
            f"Hit@10 {_count(truncated['at_k']['10']['hit'])}.",
            f"BM25-only 8건 회복 {recovery['recovered_hit10']}/8.",
            "both-hit "
            f"{retention['both_hit']['truncated_hit10']}/{retention['both_hit']['prior']}.",
            f"Dense-only {retention['dense_only']['hit10']}/{retention['dense_only']['prior']}.",
            f"both-miss Hit@10 {retention['both_miss']['truncated_hit10']}/"
            f"{retention['both_miss']['prior']}.",
            f"requirement recall {_count(truncated['requirement_recall@10'])}, "
            f"all requirements {_count(truncated['all_requirements_covered@10'])}.",
            f"judged_coverage@10 {_num(truncated['at_k']['10']['judged_coverage']['value'])}.",
        ]
    )


def _interpretation(payload: dict[str, Any]) -> str:
    label = payload["verdict"]["label"]
    overlap = payload["overlap"]
    recovery = payload["recovery"]
    semantic = payload["representatives"]["dense_only_semantic"]
    shared = (
        "이 문장은 Top-20 candidate-truncated, equal-weight, k=60 RRF가 "
        "현재 v2.1 DEV에서 보인 결과다. "
        "Hybrid Search 일반으로 확장하지 않는다. "
        "Dense-only semantic case는 이미 관측된 사실이고, "
        "이번 성패로 Dense가 필요 없다는 결론을 만들지 않는다. "
        f"교집합 중앙값은 {_num(overlap['median'])}이고, "
        f"10 이상인 질의는 {overlap['questions_with_intersection_at_least_10']}개다. "
        "양쪽 top-20에 있는 unit은 한쪽에만 있는 unit보다 항상 앞에 선다."
    )
    if label == "A":
        body = (
            "truncated RRF는 BM25 Hit@10을 41/45 이상으로 유지했고, "
            "full-RRF가 잃은 lexical 문항의 대부분을 되돌렸으며, "
            "Dense-only 문항도 Hit@10에 남았다."
        )
    elif label == "B":
        body = (
            f"full-RRF Hit@10 36/45보다 truncated는 분명 높고, "
            f"BM25-only 8건 중 {recovery['recovered_hit10']}건을 되돌렸다. "
            "BM25보다는 아직 열세이고 lexical 보존과 semantic 추가 사이의 "
            "trade-off가 남아 있다. "
            f"삼성전자 구매/매입 truncated rank는 {semantic['truncated_rank']}이다."
        )
    else:
        body = (
            f"BM25-only 8건 중 회복은 {recovery['recovered_hit10']}건이다. "
            "한쪽에만 있는 BM25 상위 unit은 교집합 전체의 뒤로 밀린다. "
            f"삼성전자 구매/매입 truncated rank는 {semantic['truncated_rank']}이다. "
            "Dense-only 한 건이 전체 순위 하락을 상쇄하지 못한다."
        )
    return body + " " + shared


def _why(payload: dict[str, Any]) -> str:
    recovery = payload["recovery"]
    semantic = payload["representatives"]["dense_only_semantic"]
    return (
        "deep-tail rank를 빼자 "
        f"BM25-only 8건 중 {recovery['recovered_hit10']}건이 Hit@10으로 돌아왔고 "
        f"Dense-only는 truncated rank {semantic['truncated_rank']}에 남았다. "
        "남은 손실은 BM25 단독 rank 4, 7, 8이 교집합과 "
        "상대 리스트의 더 높은 등위 뒤에 선 것이다. "
        f"k=60, N=20에서 교집합 최소 점수는 {intersection_min_score():.5f}이고 "
        f"단일 리스트 최대 점수는 {single_list_max_score():.5f}이다."
    )


def _limitations() -> str:
    return (
        "DEV 45문항만 본다. TEST로 N, k, weight를 고르지 않았다. "
        "저장된 top-20을 재사용했고 retrieval은 다시 돌리지 않았다. "
        "동점은 retrieval_unit_id로 가른다. ordinal은 저장 순위에 없다. "
        "unjudged를 nonrelevant로 두지 않는다. "
        "both-miss 2건으로 period, scope, cross-filing 해결을 말하지 않는다. "
        "다음 실험으로 weighted RRF, score fusion, routing, reranker, "
        "period-aware ranking을 구현하지 않는다."
    )


def _verdict_prose(verdict: dict[str, Any]) -> str:
    drops = ", ".join(
        f"{name} drop {value:.3f}" for name, value in verdict["ranking_drops"].items()
    )
    return (
        f"Hit@10 {verdict['truncated_hit10']}/45, "
        f"full-RRF {verdict['full_hit10']}/45, "
        f"BM25-only 8건 회복 {verdict['recovered_of_eight']}, "
        f"Dense-only retained {verdict['dense_only_retained']}, "
        f"both-hit lost {verdict['both_hit_lost']}, "
        f"{drops}."
    )


def _markdown_table(payload: dict[str, Any]) -> str:
    specs = (
        ("Hit@1", ("at_k", "1", "hit"), "count"),
        ("Hit@5", ("at_k", "5", "hit"), "count"),
        ("Hit@10", ("at_k", "10", "hit"), "count"),
        ("Recall@5 macro", ("recall@5_macro",), "value"),
        ("Recall@5 micro", ("recall@5_micro",), "count"),
        ("Recall@10 macro", ("recall@10_macro",), "value"),
        ("Recall@10 micro", ("recall@10_micro",), "count"),
        ("MRR", ("mrr",), "value"),
        ("NDCG@10 condensed", ("at_k", "10", "ndcg_condensed"), "value"),
        ("NDCG@10 zero-filled", ("at_k", "10", "ndcg_unjudged_as_zero"), "value"),
        ("preferred_hit@10", ("at_k", "10", "preferred_hit"), "count"),
        ("requirement_recall@10", ("requirement_recall@10",), "count"),
        ("all_requirements_covered@10", ("all_requirements_covered@10",), "count"),
        ("judged_coverage@10", ("at_k", "10", "judged_coverage"), "value"),
        ("judged_coverage@20", ("judged_coverage@20",), "value"),
    )
    names = (
        ("dense", "dense_headline"),
        ("bm25", "bm25_headline"),
        ("full_rrf", "full_rrf_headline"),
        ("truncated", "truncated_headline"),
    )
    lines = [
        "| metric | Dense | BM25 | Full RRF | Truncated RRF |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, path, kind in specs:
        cells = [label]
        for _name, key in names:
            cells.append(_metric_cell(_dig(payload[key], path), kind))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _headline_document(payload: dict[str, Any]) -> dict[str, Any]:
    specs = (
        ("Hit@1", ("at_k", "1", "hit")),
        ("Hit@5", ("at_k", "5", "hit")),
        ("Hit@10", ("at_k", "10", "hit")),
        ("Recall@5 macro", ("recall@5_macro",)),
        ("Recall@5 micro", ("recall@5_micro",)),
        ("Recall@10 macro", ("recall@10_macro",)),
        ("Recall@10 micro", ("recall@10_micro",)),
        ("MRR", ("mrr",)),
        ("NDCG@10 condensed", ("at_k", "10", "ndcg_condensed")),
        ("NDCG@10 zero-filled", ("at_k", "10", "ndcg_unjudged_as_zero")),
        ("preferred_hit@10", ("at_k", "10", "preferred_hit")),
        ("requirement_recall@10", ("requirement_recall@10",)),
        ("all_requirements_covered@10", ("all_requirements_covered@10",)),
        ("judged_coverage@10", ("at_k", "10", "judged_coverage")),
        ("judged_coverage@20", ("judged_coverage@20",)),
    )
    keys = ("dense", "bm25", "full_rrf", "truncated")
    rows = []
    for label, path in specs:
        item: dict[str, Any] = {"metric": label}
        for name in keys:
            block = _dig(payload[f"{name}_headline"], path)
            item[f"{name}_numerator"] = block.get("numerator")
            item[f"{name}_denominator"] = block.get("denominator")
            item[f"{name}_value"] = block.get("value")
        rows.append(item)
    return {
        "table": rows,
        "verdict": payload["verdict"],
        "overlap": payload["overlap"],
        "ndcg_note": (
            "headline NDCG is condensed. zero-filled is sensitivity. "
            "preferred_hit@10 is diagnostic. unjudged is not nonrelevant."
        ),
    }


def _config(payload: dict[str, Any]) -> dict[str, Any]:
    audit = payload["audit"]
    return {
        "experiment_id": payload["experiment_id"],
        "dataset_id": payload["dataset_id"],
        "dataset_sha256": payload["dataset_sha256"],
        "population": payload["population"],
        "question_count": payload["question_count"],
        "test_questions_searched": 0,
        "test_used_for_tuning": False,
        "retrieval_reexecuted": False,
        "intended_base_commit": audit["intended_base_commit"],
        "base_commit_confirmed": audit["base_commit_confirmed"],
        "source_runs": {
            "complementarity_dense_sha256": audit["input_artifacts"]["dense_dev_sha256"],
            "complementarity_bm25_sha256": audit["input_artifacts"]["bm25_dev_sha256"],
            "full_rrf_run": audit["input_artifacts"]["full_rrf_run"],
            "full_rrf_dev_sha256": audit["input_artifacts"]["full_rrf_dev_sha256"],
        },
        "truncated_rrf": audit["implementation"],
        "dense": {
            "implementation": "stored Dense-D2a top-20",
            "model": DENSE_MODEL_ID,
            "query_encoding_policy": "raw_question_text_no_instruction",
        },
        "bm25": {
            "implementation": "stored Bm25Index.search top-20",
            "tokenizer_version": TOKENIZER_VERSION,
            "k1": BM25_PARAMS["k1"],
            "b": BM25_PARAMS["b"],
        },
        "forbidden": {
            "n_sweep": False,
            "k_sweep": False,
            "weight_tuning": False,
            "alpha_tuning": False,
            "query_routing": False,
            "reranker": False,
            "query_expansion": False,
            "embedding_change": False,
            "chunking_change": False,
        },
        "git_commit": audit["git"]["git_commit"],
        "git_dirty": audit["git"]["git_dirty"],
        "head_subject": audit["head_subject"],
        "verdict_label": payload["verdict"]["label"],
    }


def _top_index(
    hits: list[dict[str, Any]],
    label: str,
    n: int,
) -> dict[str, dict[str, Any]]:
    """Index ranks 1..n. Do not read the tail."""
    if len(hits) < n:
        raise UsageError(f"{label} top list is shorter than {n}")
    prefix = list(hits[:n])
    indexed: dict[str, dict[str, Any]] = {}
    for index, hit in enumerate(prefix):
        rank = int(hit["rank"])
        if rank != index + 1:
            raise UsageError(f"{label} rank {rank} is not contiguous at position {index + 1}")
        if rank > n:
            raise UsageError(f"{label} rank {rank} is outside the candidate window")
        unit_id = str(hit.get("retrieval_unit_id") or hit.get("chunk_id") or "")
        if not unit_id:
            raise UsageError(f"{label} hit has no retrieval_unit_id")
        if unit_id in indexed:
            raise UsageError(f"duplicate {label} retrieval_unit_id {unit_id}")
        indexed[unit_id] = {
            "rank": rank,
            "source_element_ids": [str(item) for item in hit.get("source_element_ids") or []],
        }
    return indexed


def _contribution(rank: int | None, k: int, label: str) -> float:
    if rank is None:
        return 0.0
    if rank < 1 or rank > CANDIDATE_N:
        raise UsageError(f"{label} contribution rank {rank} is outside 1..{CANDIDATE_N}")
    return 1.0 / (k + rank)


def _elements(
    source: dict[str, Any],
    other: dict[str, Any] | None,
) -> list[str]:
    left = list(source.get("source_element_ids") or [])
    right = [] if other is None else list(other.get("source_element_ids") or [])
    if left and right and left != right:
        raise UsageError("Dense and BM25 source_element_ids disagree for one unit")
    return left or right


def _assert_intersection_precedes(scored: list[dict[str, Any]]) -> None:
    both = [
        float(item["score"])
        for item in scored
        if item["in_dense_top_n"] and item["in_bm25_top_n"]
    ]
    single = [
        float(item["score"])
        for item in scored
        if not (item["in_dense_top_n"] and item["in_bm25_top_n"])
    ]
    if both and single and min(both) <= max(single):
        raise UsageError("an intersection unit did not outrank every single-list unit")


def _first_relevant_unit(hits: list[dict[str, Any]]) -> str | None:
    for hit in hits:
        relevance = hit.get("relevance")
        if relevance is not None and int(relevance) >= 1:
            return str(hit["retrieval_unit_id"])
    return None


def _prefix_diff(
    query_id: str,
    label: str,
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(left) < CANDIDATE_N or len(right) < CANDIDATE_N:
        return {
            "query_id": query_id,
            "label": label,
            "reason": "top-20 list is shorter than 20",
        }
    for index in range(CANDIDATE_N):
        left_id = str(left[index].get("retrieval_unit_id") or left[index].get("chunk_id"))
        right_id = str(right[index].get("retrieval_unit_id") or right[index].get("chunk_id"))
        if int(left[index]["rank"]) != index + 1 or int(right[index]["rank"]) != index + 1:
            return {
                "query_id": query_id,
                "label": label,
                "reason": "rank is not contiguous",
                "rank": index + 1,
            }
        if left_id != right_id:
            return {
                "query_id": query_id,
                "label": label,
                "reason": "retrieval_unit_id differs",
                "rank": index + 1,
                "left_retrieval_unit_id": left_id,
                "right_retrieval_unit_id": right_id,
            }
    return None


def _transition_lines(cells: dict[str, Any]) -> str:
    lines = []
    for name in ("both", "dense_only", "bm25_only", "neither"):
        block = cells[name]
        lines.append(
            f"- {name} prior {block['prior']}: "
            f"`{json.dumps(block['transitions'], ensure_ascii=False)}`"
        )
    return "\n".join(lines)


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def _dig(block: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    current: Any = block
    for key in path:
        current = current[key]
    return current


def _metric_cell(block: dict[str, Any], kind: str) -> str:
    if kind == "count":
        return _count(block)
    return _num(block.get("value"))


def _count(block: dict[str, Any]) -> str:
    return f"{block['numerator']}/{block['denominator']}"


def _num(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _id_suffix(ids: list[str]) -> str:
    if not ids:
        return ""
    return ": " + ", ".join(ids)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MOVEMENT_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in MOVEMENT_COLUMNS})


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _head_subject(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
