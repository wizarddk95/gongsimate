"""Fixed-k RRF of Dense-D2a and BM25 on v2.1 DEV.

The combination rule is the repository's ``reciprocal_rank_fusion``. k stays 60.
Weights, routing, rerankers, query expansion, embeddings, and chunking stay
untouched. v2.1 TEST is not searched and is not used to choose a parameter.

The locked function fuses two complete rankings of the same candidate ids.
A document missing from one list is an error, not a zero contribution. This
run therefore ranks every eligible D2a unit in the complementarity filing
scope. It aborts before fusion metrics if the recomputed top-20 prefix does
not match the stored DEV artifacts.
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

from disclosure_agent.agent.search import _eligible_rows
from disclosure_agent.agent.search_catalog import CatalogDenseSearch, load_search_catalog
from disclosure_agent.bm25 import BM25_PARAMS, TOKENIZER_VERSION
from disclosure_agent.dense import DENSE_MODEL_ID, exact_normalized_search
from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2_1 import DATASET_ID
from disclosure_agent.evalset_v2_1.complementarity import (
    CELL_BM25,
    CELL_BOTH,
    CELL_DENSE,
    CELL_MISS,
    NULL_FILTERS,
    _index_scope,
    assert_dev_population,
    covered_requirement_ids,
    first_relevant_rank,
    metric_hits,
    relevant_ids_at,
)
from disclosure_agent.evalset_v2_1.complementarity import (
    RUN_DIR_NAME as SOURCE_RUN_DIR,
)
from disclosure_agent.evalset_v2_1.freeze import dataset_dir
from disclosure_agent.evalset_v2_1.metrics import aggregate_headline, score_question
from disclosure_agent.evaluation import git_revision
from disclosure_agent.hybrid import RRF_K, reciprocal_rank_fusion
from disclosure_agent.retrieval_baseline import TOP_K, _search_question

RUN_DIR_NAME = "multi_company_retrieval_v2_1_dev_rrf"
HIT_K = 10
DATASET_SHA = "2b1c372b7adaca29af831986bf5e6807c2ae2bd7835a52735690e227c1b03137"
PINNED_CELLS = {
    CELL_BOTH: 32,
    CELL_DENSE: 1,
    CELL_BM25: 10,
    CELL_MISS: 2,
}
PINNED_REQUIREMENT_CELLS = {
    "both": 40,
    "dense_only": 1,
    "bm25_only": 11,
    "neither": 6,
}
PINNED_HITS = {
    "dense": {"1": 20, "5": 28, "10": 33},
    "bm25": {"1": 29, "5": 40, "10": 42},
}
PINNED_MICRO = {
    "dense": {"5": (78, 231), "10": (129, 231)},
    "bm25": {"5": (113, 231), "10": (174, 231)},
}
PINNED_REQUIREMENT_RECALL = {"dense": (41, 58), "bm25": (51, 58)}
PINNED_ALL_REQUIREMENTS = {"dense": (32, 45), "bm25": (41, 45)}
TRACKED_QUERIES = {
    "dense_only": "V2-00126380-finance_table-02",
    "both_miss": (
        "V2-00107613-cross_period-02",
        "V2-00107613-finance_table-03",
    ),
    "samyang_statements": "V2-00126955-cross_period-01",
    "samyang_debt": "V2-00126955-cross_period-02",
}
MOVEMENT_COLUMNS = (
    "query_id",
    "company",
    "prior_cell",
    "dense_first_relevant_rank",
    "bm25_first_relevant_rank",
    "rrf_first_relevant_rank",
    "rank_delta_vs_bm25",
    "rank_delta_vs_dense",
    "rrf_best_relevant_rank",
    "rrf_relevant_in_top5",
    "rrf_relevant_in_top10",
    "movement",
    "bm25_hit10",
    "rrf_hit10",
    "bm25_first_unit_rrf_rank",
    "same_unit_rank_delta",
)


def implementation_audit() -> dict[str, Any]:
    """State the locked function's contract. Do not describe a different RRF."""
    return {
        "function": "disclosure_agent.hybrid.reciprocal_rank_fusion",
        "k": RRF_K,
        "k_swept": False,
        "rank_base": 1,
        "formula": "score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_dense(d))",
        "weights": "1:1",
        "weighted_score_fusion": False,
        "score_mixing": False,
        "candidate_cutoff": None,
        "full_ranking_fusion": True,
        "absent_contribution_is_zero": False,
        "absent_document_behavior": (
            "reciprocal_rank_fusion raises UsageError when the candidate id sets "
            "differ. A document returned by only one retriever is not given "
            "contribution 0. This experiment ranks every eligible D2a unit in the "
            "filing scope with both retrievers, so every fused document has both terms."
        ),
        "bm25_zero_score_contribution": "1/(k + rank), not 0",
        "zero_tie_variant_used": False,
        "tie_break": ["score descending", "ordinal", "unit_index", "chunk_id"],
        "duplicate_chunk_id": "rejected",
        "retrieval_unit_id_policy": (
            "chunk_id is the retrieval_unit_id. Fusion emits each id once."
        ),
    }


def fuse_complete_rankings(
    bm25_hits: list[dict[str, Any]],
    dense_hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Call the locked RRF. Reject a partial list and a repeated unit id."""
    _assert_unit_identity(bm25_hits, "bm25")
    _assert_unit_identity(dense_hits, "dense")
    fused = reciprocal_rank_fusion(
        bm25_hits,
        dense_hits,
        k_rrf=RRF_K,
        expected_count=len(bm25_hits),
    )
    ids = [str(hit.get("retrieval_unit_id") or hit["chunk_id"]) for hit in fused]
    if len(ids) != len(set(ids)):
        raise UsageError("fused retrieval_unit_id is not unique")
    if len(ids) != len(bm25_hits):
        raise UsageError("fusion dropped or added a candidate")
    for hit in fused:
        if hit.get("bm25_rank") is None or hit.get("dense_rank") is None:
            raise UsageError("fused hit is missing a retriever rank")
        expected = (1.0 / (RRF_K + int(hit["bm25_rank"]))) + (
            1.0 / (RRF_K + int(hit["dense_rank"]))
        )
        if abs(float(hit["score"]) - expected) > 1e-12:
            raise UsageError("RRF score is not the sum of the two reciprocal ranks")
        hit["retrieval_unit_id"] = str(hit.get("retrieval_unit_id") or hit["chunk_id"])
    return fused


def ranking_ids(hits: list[dict[str, Any]]) -> list[str]:
    return [str(hit.get("retrieval_unit_id") or hit.get("chunk_id")) for hit in hits]


def prefix_mismatch(
    query_id: str,
    label: str,
    stored: list[dict[str, Any]],
    fresh: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the first top-20 disagreement, or None when the prefix matches."""
    stored_ids = ranking_ids(stored)
    fresh_ids = ranking_ids(fresh)
    if len(fresh_ids) < len(stored_ids):
        return {
            "query_id": query_id,
            "label": label,
            "reason": "fresh ranking is shorter than the stored ranking",
            "stored_length": len(stored_ids),
            "fresh_length": len(fresh_ids),
        }
    for index, stored_id in enumerate(stored_ids):
        if int(stored[index]["rank"]) != index + 1:
            return {
                "query_id": query_id,
                "label": label,
                "reason": "stored rank is not 1-based and contiguous",
                "rank": int(stored[index]["rank"]),
            }
        if fresh_ids[index] != stored_id:
            return {
                "query_id": query_id,
                "label": label,
                "reason": "retrieval_unit_id prefix differs",
                "rank": index + 1,
                "stored_retrieval_unit_id": stored_id,
                "fresh_retrieval_unit_id": fresh_ids[index],
            }
    return None


def raise_if_mismatches(mismatches: list[dict[str, Any]]) -> None:
    if not mismatches:
        return
    first = mismatches[0]
    raise UsageError(
        "recomputed ranking does not match the stored DEV artifact: "
        f"{first['query_id']} {first['label']} {first['reason']} "
        f"({len(mismatches)} mismatches). Fusion was not scored."
    )


def rank_delta(baseline: int | None, fused: int | None) -> int | None:
    """Positive means the fused first-relevant rank is earlier than the baseline."""
    if baseline is None or fused is None:
        return None
    return int(baseline) - int(fused)


def movement_label(bm25_rank: int | None, rrf_rank: int | None, *, hit_k: int = HIT_K) -> str:
    """Classify the first-relevant rank against the stored BM25 window."""
    if bm25_rank is None:
        if rrf_rank is not None and int(rrf_rank) <= hit_k:
            return "recovered"
        return "still_missing"
    if rrf_rank is None or int(rrf_rank) > int(bm25_rank):
        return "regression"
    if int(rrf_rank) == int(bm25_rank):
        return "neutral"
    return "improvement"


def requirement_transitions(
    prior_rows: list[dict[str, Any]],
    covered_by_query: dict[str, set[str]],
) -> dict[str, Any]:
    """Track the previous requirement cells under the RRF top-10 window."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prior_rows:
        covered = row["requirement_id"] in covered_by_query.get(row["query_id"], set())
        groups[str(row["cell"])].append(
            {
                "query_id": row["query_id"],
                "requirement_id": row["requirement_id"],
                "requirement_text": row.get("requirement_text") or "",
                "prior_cell": row["cell"],
                "rrf_covered@10": covered,
                "transition": _requirement_transition(str(row["cell"]), covered),
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
    return {"cells": summary, "requirements": [item for cell in groups.values() for item in cell]}


def judge_hypothesis(
    *,
    bm25_hit10: int,
    rrf_hit10: int,
    bm25_only_lost: int,
    dense_only_total: int,
    dense_only_recovered: int,
    bm25_mrr: float,
    rrf_mrr: float,
    bm25_ndcg: float,
    rrf_ndcg: float,
    bm25_recall5: float,
    rrf_recall5: float,
) -> dict[str, Any]:
    """Apply the pre-registered order. Hit@10 rising alone is not success.

    C when two or more BM25 Hit@10 questions are lost, or two or more of the
    BM25-only Hit@10 questions are lost. One loss still counts as almost
    retained. A requires that almost-retained coverage and either full
    recovery of the Dense-only cell or a ranking-quality gain with no
    ranking-quality loss among MRR, NDCG@10, and Recall@5 macro. B is the
    remaining almost-retained result.
    """
    hit_drop = int(bm25_hit10) - int(rrf_hit10)
    directions = {
        "mrr": _direction(bm25_mrr, rrf_mrr),
        "ndcg@10": _direction(bm25_ndcg, rrf_ndcg),
        "recall@5_macro": _direction(bm25_recall5, rrf_recall5),
    }
    ups = [name for name, direction in directions.items() if direction == "up"]
    downs = [name for name, direction in directions.items() if direction == "down"]
    significant = hit_drop >= 2 or int(bm25_only_lost) >= 2
    coverage_kept = hit_drop <= 1 and int(bm25_only_lost) <= 1
    recovered = int(dense_only_total) > 0 and int(dense_only_recovered) == int(dense_only_total)
    quality_improved = bool(ups) and not downs
    if significant or not coverage_kept:
        label = "C"
    elif recovered or quality_improved:
        label = "A"
    else:
        label = "B"
    return {
        "label": label,
        "hit10_drop": hit_drop,
        "bm25_only_lost": int(bm25_only_lost),
        "coverage_almost_or_fully_retained": coverage_kept,
        "significant_coverage_regression": significant,
        "dense_only_recovered": recovered,
        "ranking_directions": directions,
        "ranking_quality_improved": quality_improved,
        "rule": (
            "C if Hit@10 drops by 2 or more, or 2 or more BM25-only Hit@10 "
            "questions are lost. Otherwise A if coverage is almost or fully "
            "retained and either the Dense-only question is recovered or MRR, "
            "NDCG@10, and Recall@5 macro improve with none of them falling. "
            "Otherwise B."
        ),
    }


def validate_source_bundle(root: Path) -> dict[str, Any]:
    """Check the committed complementarity artifacts before any retrieval."""
    directory = dataset_dir(root)
    frozen = _read(directory / "freeze.json")
    if frozen.get("dataset_id") != DATASET_ID:
        raise UsageError("RRF dataset is not v2.1")
    if frozen.get("dataset_sha256") != DATASET_SHA:
        raise UsageError("v2.1 dataset hash changed since the complementarity run")
    questions = assert_dev_population(
        _read(directory / "questions.json")["questions"],
        expected_count=int(frozen["dev_size"]),
    )
    if len(questions) != 45:
        raise UsageError("DEV fully_answerable population is not 45")
    source = root / "evaluation" / "runs" / SOURCE_RUN_DIR
    config = _read(source / "experiment_config.json")
    if config.get("dataset_sha256") != DATASET_SHA:
        raise UsageError("complementarity artifact dataset hash does not match v2.1")
    if int(config.get("test_questions_searched") or 0) != 0:
        raise UsageError("complementarity artifact searched test questions")
    dense_doc = _read(source / "dense_dev.json")
    bm25_doc = _read(source / "bm25_dev.json")
    per_query = _read(source / "per_query_complementarity.json")["queries"]
    requirements = _read(source / "requirement_complementarity.json")
    _assert_pinned_headline(dense_doc["headline"], "dense")
    _assert_pinned_headline(bm25_doc["headline"], "bm25")
    dense_by_id = _index_questions(dense_doc["questions"], questions)
    bm25_by_id = _index_questions(bm25_doc["questions"], questions)
    query_by_id = {str(row["query_id"]): row for row in per_query}
    if set(query_by_id) != {str(row["query_id"]) for row in questions}:
        raise UsageError("per-query complementarity ids are not the DEV 45")
    cell_counts = Counter(row["cell"] for row in per_query)
    for name, expected in PINNED_CELLS.items():
        if int(cell_counts.get(name, 0)) != expected:
            raise UsageError(f"stored query cell {name} is not {expected}")
    requirement_counts = Counter(row["cell"] for row in requirements["requirements"])
    for name, expected in PINNED_REQUIREMENT_CELLS.items():
        if int(requirement_counts.get(name, 0)) != expected:
            raise UsageError(f"stored requirement cell {name} is not {expected}")
    for query_id, row in query_by_id.items():
        _assert_stored_rank(query_id, "dense", dense_by_id[query_id]["hits"], row)
        _assert_stored_rank(query_id, "bm25", bm25_by_id[query_id]["hits"], row)
    tracked = {TRACKED_QUERIES["dense_only"], *TRACKED_QUERIES["both_miss"]}
    tracked.update((TRACKED_QUERIES["samyang_statements"], TRACKED_QUERIES["samyang_debt"]))
    missing = sorted(tracked - set(query_by_id))
    if missing:
        raise UsageError(f"tracked queries are absent from DEV: {missing}")
    return {
        "frozen": frozen,
        "questions": questions,
        "requirements": _read(directory / "requirements.json")["requirements"],
        "qrels": _read(directory / "qrels.json")["qrels"],
        "config": config,
        "dense_by_id": dense_by_id,
        "bm25_by_id": bm25_by_id,
        "query_by_id": query_by_id,
        "requirement_rows": requirements["requirements"],
        "dense_headline": dense_doc["headline"],
        "bm25_headline": bm25_doc["headline"],
        "source_dir": source,
        "dense_sha256": _sha_file(source / "dense_dev.json"),
        "bm25_sha256": _sha_file(source / "bm25_dev.json"),
    }


def run_rrf_experiment(root: Path) -> dict[str, Any]:
    """Fuse DEV rankings after the stored top-20 prefix matches."""
    bundle = validate_source_bundle(root)
    questions = bundle["questions"]
    qrels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bundle["qrels"]:
        qrels_by_query[str(row["query_id"])].append(row)
    catalog = load_search_catalog(root)
    search = CatalogDenseSearch(catalog)
    by_company: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        by_company[str(question["corp_code"])].append(question)

    mismatches: list[dict[str, Any]] = []
    movement: list[dict[str, Any]] = []
    rrf_scored: list[dict[str, Any]] = []
    covered_by_query: dict[str, set[str]] = {}
    zero_score_docs = 0
    zero_score_spot_checks = 0
    formula_checks = 0
    companies: list[dict[str, Any]] = []
    scope_cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

    for corp in sorted(by_company):
        corpus = catalog.corpus(corp)
        if corpus is None:
            raise UsageError(f"{corp} is not in the product catalog")
        group = sorted(by_company[corp], key=lambda item: str(item["query_id"]))
        company_units = 0
        for question in group:
            query_id = str(question["query_id"])
            scope, dense_top = _search_question(
                {
                    "question_id": query_id,
                    "corp_code": corp,
                    "text": question["question"],
                    "filters": dict(NULL_FILTERS),
                },
                catalog,
                search,
            )
            receipts = tuple(str(item) for item in scope["effective_receipt_nos"])
            cached = scope_cache.get((corp, receipts))
            if cached is None:
                cached = _index_scope(search, corpus, corp, receipts)
                scope_cache[(corp, receipts)] = cached
            bm25_top = cached["index"].search(question["question"], TOP_K)
            stored_dense = bundle["dense_by_id"][query_id]["hits"]
            stored_bm25 = bundle["bm25_by_id"][query_id]["hits"]
            for label, stored, fresh in (
                ("dense_product_vs_stored", stored_dense, dense_top),
                ("bm25_search_vs_stored", stored_bm25, bm25_top),
            ):
                found = prefix_mismatch(query_id, label, stored, fresh)
                if found is not None:
                    mismatches.append(found)
            if mismatches:
                print(f"rrf mismatch {query_id}", flush=True)
                continue
            units, matrix, scoring = _eligible_rows(
                search._search[corp].index(), corpus.binding, receipts
            )
            fresh_ids = [str(unit["retrieval_unit_id"]) for unit in units]
            if fresh_ids != cached["ids"]:
                raise UsageError(f"{query_id} eligible units diverged from the BM25 index")
            company_units = len(fresh_ids)
            vector = search.encoder.encode(question["question"])
            dense_full = exact_normalized_search(matrix, vector, scoring)
            bm25_full = cached["index"].rank_all(question["question"])
            for label, stored, fresh in (
                ("dense_full_vs_product", dense_top, dense_full),
                ("bm25_rank_all_vs_search", bm25_top, bm25_full),
            ):
                found = prefix_mismatch(query_id, label, stored, fresh)
                if found is not None:
                    mismatches.append(found)
            if mismatches:
                print(f"rrf mismatch {query_id}", flush=True)
                continue
            fused = fuse_complete_rankings(bm25_full, dense_full)
            formula_checks += len(fused)
            zero_score_docs, zero_score_spot_checks = _count_zero_scores(
                bm25_full,
                fused,
                zero_score_docs,
                zero_score_spot_checks,
            )
            owned_qrels = qrels_by_query.get(query_id) or []
            scored = score_question(
                question,
                owned_qrels,
                bundle["requirements"],
                metric_hits(fused, cached["meta"]),
            )
            record = _movement_record(
                question,
                bundle["query_by_id"][query_id],
                bundle["dense_by_id"][query_id]["hits"],
                bundle["bm25_by_id"][query_id]["hits"],
                scored,
                fused,
                cached["meta"],
            )
            covered, _missing = covered_requirement_ids(
                query_id,
                bundle["requirements"],
                owned_qrels,
                scored["hits"],
                k=HIT_K,
            )
            covered_by_query[query_id] = set(covered)
            scored["hits"] = scored["hits"][:TOP_K]
            rrf_scored.append(scored)
            movement.append(record)
            print(f"rrf {query_id} {record['movement']}", flush=True)
        companies.append(
            {
                "corp_code": corp,
                "company_name": corpus.binding.company_name,
                "question_count": len(group),
                "eligible_unit_count": company_units,
            }
        )
        search._search.pop(corp, None)
        scope_cache.clear()

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
        raise_if_mismatches(mismatches)

    if len(movement) != 45 or len(rrf_scored) != 45:
        raise UsageError("RRF did not score every DEV question")
    movement.sort(key=lambda item: item["query_id"])
    rrf_scored.sort(key=lambda item: item["query_id"])
    rrf_headline = aggregate_headline(rrf_scored)
    transitions = requirement_transitions(bundle["requirement_rows"], covered_by_query)
    transitions["requirements"].sort(key=lambda item: (item["query_id"], item["requirement_id"]))
    retention = summarize_retention(movement)
    representatives = select_representatives(movement)
    verdict = judge_hypothesis(
        bm25_hit10=int(bundle["bm25_headline"]["at_k"]["10"]["hit"]["numerator"]),
        rrf_hit10=int(rrf_headline["at_k"]["10"]["hit"]["numerator"]),
        bm25_only_lost=int(retention["bm25_only"]["lost"]),
        dense_only_total=int(retention["dense_only"]["prior"]),
        dense_only_recovered=int(retention["dense_only"]["recovered"]),
        bm25_mrr=float(bundle["bm25_headline"]["mrr"]["value"]),
        rrf_mrr=float(rrf_headline["mrr"]["value"]),
        bm25_ndcg=float(bundle["bm25_headline"]["at_k"]["10"]["ndcg_condensed"]["value"]),
        rrf_ndcg=float(rrf_headline["at_k"]["10"]["ndcg_condensed"]["value"]),
        bm25_recall5=float(bundle["bm25_headline"]["recall@5_macro"]["value"]),
        rrf_recall5=float(rrf_headline["recall@5_macro"]["value"]),
    )
    oracle = _oracle(retention, rrf_headline)
    audit = {
        "implementation": implementation_audit(),
        "reproduction": {
            "dense_top20_matched_questions": 45,
            "bm25_top20_matched_questions": 45,
            "dense_full_prefix_matched_product_top20": 45,
            "bm25_rank_all_prefix_matched_search_top20": 45,
            "mismatches": [],
            "fusion_scored": True,
        },
        "contribution": {
            "formula_checks": formula_checks,
            "documents_missing_one_rank": 0,
            "bm25_zero_score_documents": zero_score_docs,
            "bm25_zero_score_spot_checks_with_positive_contribution": zero_score_spot_checks,
            "units_outside_filing_scope_are_not_candidates": True,
        },
        "companies": companies,
        "input_artifacts": {
            "dense_dev_sha256": bundle["dense_sha256"],
            "bm25_dev_sha256": bundle["bm25_sha256"],
            "source_run": SOURCE_RUN_DIR,
        },
        "test_questions_searched": 0,
        "git": git_revision(root),
        "head_subject": _head_subject(root),
    }
    payload = {
        "experiment_id": "v21_dev_dense_d2a_bm25_rrf",
        "dataset_id": DATASET_ID,
        "dataset_sha256": DATASET_SHA,
        "population": "dev fully_answerable",
        "question_count": 45,
        "test_questions_searched": 0,
        "dense_headline": bundle["dense_headline"],
        "bm25_headline": bundle["bm25_headline"],
        "rrf_headline": rrf_headline,
        "oracle": oracle,
        "retention": retention,
        "movement_summary": _movement_summary(movement),
        "requirement_transitions": {
            "cells": transitions["cells"],
            "rrf_requirement_recall@10": rrf_headline["requirement_recall@10"],
            "rrf_all_requirements_covered@10": rrf_headline["all_requirements_covered@10"],
        },
        "representatives": representatives,
        "verdict": verdict,
        "audit": audit,
        "tracked_requirement_ids": list(TRACKED_QUERIES["both_miss"])
        + [
            TRACKED_QUERIES["dense_only"],
            TRACKED_QUERIES["samyang_statements"],
            TRACKED_QUERIES["samyang_debt"],
        ],
    }
    payload["requirement_rows"] = transitions["requirements"]
    payload["movement"] = movement
    payload["rrf_questions"] = rrf_scored
    _write_run(root, payload, bundle)
    payload["run_dir"] = str(output)
    return payload


def summarize_retention(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prior_cell"])].append(row)
    both_lost = [row["query_id"] for row in grouped[CELL_BOTH] if not row["rrf_hit10"]]
    bm25_lost = [row["query_id"] for row in grouped[CELL_BM25] if not row["rrf_hit10"]]
    dense_recovered = [row["query_id"] for row in grouped[CELL_DENSE] if row["rrf_hit10"]]
    miss_recovered = [row["query_id"] for row in grouped[CELL_MISS] if row["rrf_hit10"]]
    return {
        "both_hit": {
            "prior": len(grouped[CELL_BOTH]),
            "rrf_hit10": len(grouped[CELL_BOTH]) - len(both_lost),
            "lost": len(both_lost),
            "lost_query_ids": both_lost,
        },
        "bm25_only": {
            "prior": len(grouped[CELL_BM25]),
            "retained": len(grouped[CELL_BM25]) - len(bm25_lost),
            "lost": len(bm25_lost),
            "lost_query_ids": bm25_lost,
            "queries": [_retention_view(row) for row in grouped[CELL_BM25]],
        },
        "dense_only": {
            "prior": len(grouped[CELL_DENSE]),
            "recovered": len(dense_recovered),
            "lost": len(grouped[CELL_DENSE]) - len(dense_recovered),
            "recovered_query_ids": dense_recovered,
            "queries": [_retention_view(row) for row in grouped[CELL_DENSE]],
        },
        "both_miss": {
            "prior": len(grouped[CELL_MISS]),
            "recovered": len(miss_recovered),
            "still_missing": len(grouped[CELL_MISS]) - len(miss_recovered),
            "recovered_query_ids": miss_recovered,
            "queries": [_retention_view(row) for row in grouped[CELL_MISS]],
            "interpretation": (
                "이 두 질의는 period, 연결/별도 scope, cross-filing failure family로 둔다. "
                "RRF가 Hit@10에 넣어도 그 family가 해결된 것으로 보지 않고, "
                "못 넣어도 chunking 실패로 단정하지 않는다."
            ),
        },
    }


def select_representatives(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick rank movements without using them to retune k."""
    improvements = [row for row in rows if row["movement"] == "improvement"]
    improvements.sort(
        key=lambda row: (
            -(row["rank_delta_vs_bm25"] or 0),
            0 if row["dense_signal"] else 1,
            row["query_id"],
        )
    )
    regressions = [
        row
        for row in rows
        if row["movement"] == "regression" or (row["same_unit_rank_delta"] or 0) < 0
    ]
    regressions.sort(
        key=lambda row: (
            0 if row["bm25_hit10"] and not row["rrf_hit10"] else 1,
            row["same_unit_rank_delta"] if row["same_unit_rank_delta"] is not None else 0,
            row["query_id"],
        )
    )
    tracked_ids = {
        TRACKED_QUERIES["dense_only"],
        *TRACKED_QUERIES["both_miss"],
        TRACKED_QUERIES["samyang_statements"],
        TRACKED_QUERIES["samyang_debt"],
    }
    by_id = {row["query_id"]: row for row in rows}
    return {
        "selection_rule": (
            "Improvements are the largest positive first-relevant rank deltas. "
            "Dense-signal is only a tie break. Regressions prefer a lost BM25 "
            "Hit@10, then the most negative same-unit rank delta. Tracked "
            "queries are always included."
        ),
        "improvements": improvements[:3],
        "regressions": regressions[:3],
        "tracked": [by_id[query_id] for query_id in sorted(tracked_ids)],
    }


def _movement_record(
    question: dict[str, Any],
    prior: dict[str, Any],
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    scored: dict[str, Any],
    fused: list[dict[str, Any]],
    meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_unit = {str(hit["retrieval_unit_id"]): hit for hit in fused}
    dense_rank = prior["dense_first_relevant_rank"]
    bm25_rank = prior["bm25_first_relevant_rank"]
    rrf_rank = first_relevant_rank(scored)
    dense_unit = _first_relevant_unit(dense_hits)
    bm25_unit = _first_relevant_unit(bm25_hits)
    rrf_unit = None
    if rrf_rank is not None:
        rrf_unit = str(scored["hits"][int(rrf_rank) - 1]["retrieval_unit_id"])
    _assert_known_unit(question["query_id"], "dense", dense_unit, dense_rank, by_unit)
    _assert_known_unit(question["query_id"], "bm25", bm25_unit, bm25_rank, by_unit)
    bm25_unit_rrf = None if bm25_unit is None else int(by_unit[bm25_unit]["rank"])
    rrf_unit_hit = None if rrf_unit is None else by_unit[rrf_unit]
    dense_signal = bool(
        rrf_unit_hit is not None
        and int(rrf_unit_hit["dense_rank"]) < int(rrf_unit_hit["bm25_rank"])
    )
    top3 = []
    for hit in scored["hits"][:3]:
        unit_id = str(hit["retrieval_unit_id"])
        fused_hit = by_unit[unit_id]
        top3.append(
            {
                "rank": int(hit["rank"]),
                "retrieval_unit_id": unit_id,
                "relevance": hit["relevance"],
                "bm25_rank": int(fused_hit["bm25_rank"]),
                "dense_rank": int(fused_hit["dense_rank"]),
                "preview": _preview((meta.get(unit_id) or {}).get("text") or ""),
            }
        )
    first_preview = ""
    if rrf_unit is not None:
        first_preview = _preview((meta.get(rrf_unit) or {}).get("text") or "")
    return {
        "query_id": question["query_id"],
        "company": question.get("company_name"),
        "question": question.get("question"),
        "prior_cell": prior["cell"],
        "dense_first_relevant_rank": dense_rank,
        "bm25_first_relevant_rank": bm25_rank,
        "rrf_first_relevant_rank": rrf_rank,
        "rrf_best_relevant_rank": rrf_rank,
        "best_relevant_definition": (
            "minimum rank whose own judged relevance is >= 1. "
            "This is the same value as rrf_first_relevant_rank."
        ),
        "rank_delta_vs_bm25": rank_delta(bm25_rank, rrf_rank),
        "rank_delta_vs_dense": rank_delta(dense_rank, rrf_rank),
        "rrf_relevant_in_top5": len(relevant_ids_at(scored, 5)),
        "rrf_relevant_in_top10": len(relevant_ids_at(scored, HIT_K)),
        "movement": movement_label(bm25_rank, rrf_rank),
        "bm25_hit10": bm25_rank is not None and int(bm25_rank) <= HIT_K,
        "dense_hit10": dense_rank is not None and int(dense_rank) <= HIT_K,
        "rrf_hit10": rrf_rank is not None and int(rrf_rank) <= HIT_K,
        "bm25_first_unit_id": bm25_unit,
        "bm25_first_unit_rrf_rank": bm25_unit_rrf,
        "same_unit_rank_delta": rank_delta(bm25_rank, bm25_unit_rrf),
        "rrf_first_unit_id": rrf_unit,
        "rrf_first_unit_bm25_rank": None
        if rrf_unit_hit is None
        else int(rrf_unit_hit["bm25_rank"]),
        "rrf_first_unit_dense_rank": None
        if rrf_unit_hit is None
        else int(rrf_unit_hit["dense_rank"]),
        "dense_signal": dense_signal,
        "first_relevant_unit_changed": bool(
            bm25_unit is not None and rrf_unit is not None and bm25_unit != rrf_unit
        ),
        "rrf_first_relevant_preview": first_preview,
        "rrf_top3": top3,
    }


def _movement_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["movement"] for row in rows)
    deltas = [
        int(row["rank_delta_vs_bm25"])
        for row in rows
        if row["rank_delta_vs_bm25"] is not None
    ]
    same = [
        int(row["same_unit_rank_delta"])
        for row in rows
        if row["same_unit_rank_delta"] is not None
    ]
    pushed = [row["query_id"] for row in rows if (row["same_unit_rank_delta"] or 0) < 0]
    return {
        "counts": {name: int(counts.get(name, 0)) for name in sorted(counts)},
        "rank_delta_vs_bm25_defined": len(deltas),
        "rank_delta_vs_bm25_mean": None if not deltas else sum(deltas) / len(deltas),
        "rank_delta_vs_bm25_median": _median(deltas),
        "same_unit_rank_delta_mean": None if not same else sum(same) / len(same),
        "same_unit_pushed_down": len(pushed),
        "same_unit_pushed_down_query_ids": pushed,
        "positive_delta_means": "RRF first-relevant rank is earlier than the baseline",
    }


def _oracle(retention: dict[str, Any], rrf_headline: dict[str, Any]) -> dict[str, Any]:
    both = int(retention["both_hit"]["prior"])
    dense_only = int(retention["dense_only"]["prior"])
    bm25_only = int(retention["bm25_only"]["prior"])
    both_miss = int(retention["both_miss"]["prior"])
    oracle = both + dense_only + bm25_only
    rrf_hit = int(rrf_headline["at_k"]["10"]["hit"]["numerator"])
    bm25_hit = both + bm25_only
    return {
        "definition": (
            "Hit@10 union of the stored Dense and BM25 top-10 lists. "
            "Full-ranking RRF can still surface a relevant unit that was "
            "outside both top-10 lists, so this union is the previous "
            "top-10 ceiling, not a mathematical cap on the fused ranking."
        ),
        "both_hit": both,
        "dense_only": dense_only,
        "bm25_only": bm25_only,
        "both_miss": both_miss,
        "oracle_hit10": oracle,
        "oracle_denominator": 45,
        "bm25_hit10": bm25_hit,
        "rrf_hit10": rrf_hit,
        "theoretical_top10_headroom_over_bm25": oracle - bm25_hit,
        "rrf_minus_bm25_hit10": rrf_hit - bm25_hit,
        "rrf_minus_oracle_hit10": rrf_hit - oracle,
    }


def _write_run(root: Path, payload: dict[str, Any], bundle: dict[str, Any]) -> None:
    output = root / "evaluation" / "runs" / RUN_DIR_NAME
    if dataset_dir(root) in output.parents:
        raise UsageError("refusing to write the RRF run into the dataset")
    output.mkdir(parents=True, exist_ok=True)
    mismatch = output / "reproduction_mismatch.json"
    if mismatch.exists():
        mismatch.unlink()
    shutil.copyfile(bundle["source_dir"] / "dense_dev.json", output / "dense_dev.json")
    shutil.copyfile(bundle["source_dir"] / "bm25_dev.json", output / "bm25_dev.json")
    _dump(output / "experiment_config.json", _config(payload))
    _dump(
        output / "rrf_dev.json",
        {"headline": payload["rrf_headline"], "questions": payload["rrf_questions"]},
    )
    _dump(output / "headline_comparison.json", _headline_document(payload))
    _dump(output / "per_query_rank_movement.json", {"queries": payload["movement"]})
    _write_csv(output / "per_query_rank_movement.csv", payload["movement"])
    _dump(output / "regression_audit.json", payload["retention"] | {
        "movement": payload["movement_summary"],
    })
    _dump(
        output / "requirement_analysis.json",
        {
            "transitions": payload["requirement_transitions"],
            "requirements": payload["requirement_rows"],
            "tracked": [
                row
                for row in payload["requirement_rows"]
                if row["query_id"]
                in {
                    TRACKED_QUERIES["samyang_statements"],
                    TRACKED_QUERIES["samyang_debt"],
                    TRACKED_QUERIES["dense_only"],
                }
            ],
        },
    )
    _dump(output / "representative_cases.json", payload["representatives"])
    _dump(output / "audit.json", payload["audit"])
    (output / "report.md").write_text(render_report(payload), encoding="utf-8")


def render_report(payload: dict[str, Any]) -> str:
    dense = payload["dense_headline"]
    bm25 = payload["bm25_headline"]
    rrf = payload["rrf_headline"]
    verdict = payload["verdict"]
    retention = payload["retention"]
    oracle = payload["oracle"]
    audit = payload["audit"]
    movement = payload["movement_summary"]
    transitions = payload["requirement_transitions"]["cells"]
    lines = [
        "# v2.1 DEV Dense-D2a + BM25 RRF",
        "",
        "## 1. Executive summary",
        "",
        _executive_summary(payload),
        "",
        "## 2. Previous experiment commit",
        "",
        (
            f"이 실행의 HEAD는 `{audit['git']['git_commit']}` "
            f"({audit['head_subject']})이다. "
            "작업 트리의 RRF 코드는 아직 그 커밋에 들어 있지 않다. "
            "입력 순위는 complementarity 산출물의 top-20과 같아야 한다."
        ),
        "",
        "## 3. Experiment invariants",
        "",
        (
            f"dataset `{payload['dataset_id']}` sha256 `{payload['dataset_sha256']}`. "
            "모집단은 DEV fully_answerable 45문항이다. TEST 검색 수는 0이다. "
            "k sweep, weighted fusion, alpha, query routing, reranker, "
            "query expansion, embedding 교체, chunking 변경은 없다."
        ),
        "",
        "## 4. RRF implementation audit",
        "",
        (
            f"`{audit['implementation']['function']}` k={audit['implementation']['k']}. "
            f"식은 `{audit['implementation']['formula']}`이다. "
            "한쪽에만 있는 문서에 contribution 0을 주지 않는다. "
            "후보 집합이 다르면 UsageError다. "
            "이번 실험은 filing scope 안의 eligible D2a unit 전체를 양쪽이 순위화한 뒤 합친다. "
            "BM25 점수가 0인 문서도 `1/(k+rank)`를 받는다. "
            "`reciprocal_rank_fusion_zero_tie`는 쓰지 않았다. "
            f"공식 대조 {audit['contribution']['formula_checks']}건, "
            f"rank가 하나뿐인 문서 {audit['contribution']['documents_missing_one_rank']}건, "
            f"BM25 0점 문서 {audit['contribution']['bm25_zero_score_documents']}건, "
            "0점 spot check에서 contribution이 0보다 큰 횟수 "
            f"{audit['contribution']['bm25_zero_score_spot_checks_with_positive_contribution']}."
        ),
        "",
        "filing scope 밖 unit은 후보가 아니다. 0점으로 합쳐지지 않는다.",
        "",
        "## 5. Dense / BM25 / RRF configuration",
        "",
        (
            "Dense는 제품 경로 `CatalogDenseSearch`와 `exact_normalized_search`다. "
            f"모델은 `{DENSE_MODEL_ID}`이고 질의 원문을 지시문 없이 인코딩한다. "
            f"BM25는 `Bm25Index` k1={BM25_PARAMS['k1']} b={BM25_PARAMS['b']} "
            f"tokenizer `{TOKENIZER_VERSION}`이다. "
            "headline BM25는 양수 점수 top-20이다. RRF 입력 BM25는 같은 순서의 `rank_all`이다. "
            "재계산한 top-20 prefix 45/45가 저장된 Dense, BM25와 같았다."
        ),
        "",
        "## 6. Headline comparison",
        "",
        _markdown_table(dense, bm25, rrf),
        "",
        (
            f"RRF judged_coverage@10은 {_num(rrf['at_k']['10']['judged_coverage']['value'])}이고 "
            "상위 10의 unjudged 평균은 "
            f"{_num(rrf['at_k']['10']['unjudged_count_mean']['value'])}이다. "
            "condensed NDCG@10은 "
            f"{_num(rrf['at_k']['10']['ndcg_condensed']['value'])}, "
            "unjudged-as-zero NDCG@10은 "
            f"{_num(rrf['at_k']['10']['ndcg_unjudged_as_zero']['value'])}이다. "
            "민감도 지표도 BM25 condensed NDCG보다 낮다."
        ),
        "",
        "## 7. Oracle union ceiling",
        "",
        (
            f"이전 top-10 합집합 oracle Hit@10은 {oracle['oracle_hit10']}/45이다. "
            f"BM25는 {oracle['bm25_hit10']}/45, RRF는 {oracle['rrf_hit10']}/45이다. "
            f"BM25 대비 RRF 차이는 {oracle['rrf_minus_bm25_hit10']}이고, "
            f"oracle 대비 차이는 {oracle['rrf_minus_oracle_hit10']}이다. "
            f"top-10 합집합 기준으로 BM25 위에 남은 이론적 여지는 "
            f"{oracle['theoretical_top10_headroom_over_bm25']}문항이다. "
            "전체 순위 RRF는 양쪽 top-10 밖에 있던 양성 unit을 끌어올릴 수 있으므로 "
            "이 oracle은 수학적 상한이 아니다."
        ),
        "",
        "## 8. Both-hit retention",
        "",
        (
            f"이전 both-hit {retention['both_hit']['prior']} 중 "
            f"RRF Hit@10은 {retention['both_hit']['rrf_hit10']}이다. "
            f"잃은 질의는 {retention['both_hit']['lost']}건"
            f"{_id_suffix(retention['both_hit']['lost_query_ids'])}."
        ),
        "",
        "## 9. BM25-only retention",
        "",
        (
            f"이전 BM25-only {retention['bm25_only']['prior']} 중 "
            f"RRF가 유지한 Hit@10은 {retention['bm25_only']['retained']}이다. "
            f"잃은 질의는 {retention['bm25_only']['lost']}건"
            f"{_id_suffix(retention['bm25_only']['lost_query_ids'])}."
        ),
        "",
        _query_lines(retention["bm25_only"]["queries"]),
        "",
        "## 10. Dense-only recovery",
        "",
        _query_lines(retention["dense_only"]["queries"]),
        "",
        "## 11. Both-miss result",
        "",
        retention["both_miss"]["interpretation"],
        "",
        _query_lines(retention["both_miss"]["queries"]),
        "",
        "## 12. Rank movement",
        "",
        (
            f"movement 분포: `{json.dumps(movement['counts'], ensure_ascii=False)}`. "
            f"BM25 대비 first-relevant delta 평균 "
            f"{_num(movement['rank_delta_vs_bm25_mean'])}, "
            f"중앙값 {_num(movement['rank_delta_vs_bm25_median'])}. "
            "양수 delta는 RRF가 첫 양성 unit을 더 앞에 둔 것이다. "
            f"같은 unit이 뒤로 밀린 질의는 {movement['same_unit_pushed_down']}건이다."
        ),
        "",
        "## 13. Requirement-level analysis",
        "",
        (
            "requirement recall@10 "
            f"Dense {_count(dense['requirement_recall@10'])}, "
            f"BM25 {_count(bm25['requirement_recall@10'])}, "
            f"RRF {_count(rrf['requirement_recall@10'])}. "
            "all requirements covered@10 "
            f"Dense {_count(dense['all_requirements_covered@10'])}, "
            f"BM25 {_count(bm25['all_requirements_covered@10'])}, "
            f"RRF {_count(rrf['all_requirements_covered@10'])}."
        ),
        "",
        _transition_lines(transitions),
        "",
        _tracked_requirement_lines(payload["requirement_rows"]),
        "",
        "## 14. Representative improvements",
        "",
        _case_lines(payload["representatives"]["improvements"]),
        "",
        "## 15. Representative regressions",
        "",
        _case_lines(payload["representatives"]["regressions"]),
        "",
        _mechanism_note(payload["representatives"]["regressions"]),
        "",
        "## 16. FACT",
        "",
        _fact_lines(payload),
        "",
        "## 17. INTERPRETATION",
        "",
        _interpretation(payload),
        "",
        "## 18. Limitations",
        "",
        (
            "DEV 45문항만 본다. k=60은 고정이고 sweep하지 않았다. "
            "TEST로 k, weight, routing을 고르지 않았다. "
            "재사용한 함수는 반환 목록에 없다는 이유로 contribution을 0으로 두지 않는다. "
            "이번 기각은 그 전체 순위 RRF에 대한 것이다. "
            "top-20 밖 contribution을 0으로 두는 변형은 실행하지 않았다. "
            "BM25 0점 문서도 깊은 순위의 reciprocal rank를 더한다. "
            "both-miss를 chunking 실패로 바꾸지 않는다. "
            "순위 동점은 ordinal, unit_index, chunk_id로 가른다."
        ),
        "",
        "## 19. RRF hypothesis verdict",
        "",
        f"판정 {verdict['label']}. {verdict['rule']}",
        "",
        _verdict_prose(verdict),
        "",
        "## 20. Recommended next experiment",
        "",
        _next_experiment(verdict["label"]),
        "",
    ]
    return "\n".join(lines)


def _executive_summary(payload: dict[str, Any]) -> str:
    rrf = payload["rrf_headline"]
    bm25 = payload["bm25_headline"]
    verdict = payload["verdict"]
    retention = payload["retention"]
    return (
        f"판정은 {verdict['label']}이다. "
        f"RRF Hit@10은 {_count(rrf['at_k']['10']['hit'])}이고 "
        f"BM25는 {_count(bm25['at_k']['10']['hit'])}이다. "
        f"BM25-only {retention['bm25_only']['prior']}건 중 "
        f"{retention['bm25_only']['retained']}건을 유지했다. "
        f"Dense-only 복구는 {retention['dense_only']['recovered']}/"
        f"{retention['dense_only']['prior']}이다. "
        f"MRR {_num(bm25['mrr']['value'])} → {_num(rrf['mrr']['value'])}, "
        f"NDCG@10 {_num(bm25['at_k']['10']['ndcg_condensed']['value'])} → "
        f"{_num(rrf['at_k']['10']['ndcg_condensed']['value'])}, "
        f"Recall@5 macro {_num(bm25['recall@5_macro']['value'])} → "
        f"{_num(rrf['recall@5_macro']['value'])}."
    )


def _markdown_table(
    dense: dict[str, Any], bm25: dict[str, Any], rrf: dict[str, Any]
) -> str:
    specs = (
        ("Hit@1", ("at_k", "1", "hit"), "count"),
        ("Hit@5", ("at_k", "5", "hit"), "count"),
        ("Hit@10", ("at_k", "10", "hit"), "count"),
        ("Recall@5 macro", ("recall@5_macro",), "value"),
        ("Recall@5 micro", ("recall@5_micro",), "count"),
        ("Recall@10 macro", ("recall@10_macro",), "value"),
        ("Recall@10 micro", ("recall@10_micro",), "count"),
        ("MRR", ("mrr",), "value"),
        ("NDCG@10", ("at_k", "10", "ndcg_condensed"), "value"),
        ("preferred_hit@10", ("at_k", "10", "preferred_hit"), "count"),
        ("requirement_recall@10", ("requirement_recall@10",), "count"),
        ("all_requirements_covered@10", ("all_requirements_covered@10",), "count"),
        ("judged_coverage@10", ("at_k", "10", "judged_coverage"), "value"),
        ("judged_coverage@20", ("judged_coverage@20",), "value"),
    )
    lines = ["| metric | Dense | BM25 | RRF |", "| --- | --- | --- | --- |"]
    for label, path, kind in specs:
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _metric_cell(_dig(dense, path), kind),
                    _metric_cell(_dig(bm25, path), kind),
                    _metric_cell(_dig(rrf, path), kind),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _fact_lines(payload: dict[str, Any]) -> str:
    rrf = payload["rrf_headline"]
    retention = payload["retention"]
    return " ".join(
        [
            f"RRF Hit@1 {_count(rrf['at_k']['1']['hit'])}, "
            f"Hit@5 {_count(rrf['at_k']['5']['hit'])}, "
            f"Hit@10 {_count(rrf['at_k']['10']['hit'])}.",
            f"both-hit 유지 {retention['both_hit']['rrf_hit10']}/"
            f"{retention['both_hit']['prior']}.",
            f"BM25-only 유지 {retention['bm25_only']['retained']}/"
            f"{retention['bm25_only']['prior']}.",
            f"Dense-only 복구 {retention['dense_only']['recovered']}/"
            f"{retention['dense_only']['prior']}.",
            f"both-miss 복구 {retention['both_miss']['recovered']}/"
            f"{retention['both_miss']['prior']}.",
            f"requirement recall {_count(rrf['requirement_recall@10'])}, "
            f"all requirements covered {_count(rrf['all_requirements_covered@10'])}.",
        ]
    )


def _interpretation(payload: dict[str, Any]) -> str:
    label = payload["verdict"]["label"]
    if label == "A":
        return (
            "RRF가 BM25의 Hit@10 coverage를 거의 또는 완전히 유지한 상태에서 "
            "Dense-only 복구 또는 MRR, NDCG@10, Recall@5 macro의 개선이 있다. "
            "관측된 complementarity의 일부가 순위 개선으로 옮겨 갔다고 볼 수 있다. "
            "이 문장은 DEV 45문항에 한정된다."
        )
    if label == "B":
        return (
            "RRF가 BM25 coverage를 크게 깨지는 않았지만, Dense-only 복구와 "
            "MRR, NDCG@10, Recall@5 macro의 깨끗한 개선이 함께 나오지 않았다. "
            "현재 고정 k의 단순 RRF가 BM25 위에 더하는 가치는 낮다. "
            "BM25-only baseline을 유지하는 쪽을 먼저 검토한다."
        )
    retention = payload["retention"]
    return (
        f"both-hit Hit@10은 {retention['both_hit']['rrf_hit10']}/"
        f"{retention['both_hit']['prior']}로 남았다. "
        f"BM25-only는 {retention['bm25_only']['retained']}/"
        f"{retention['bm25_only']['prior']}만 남았다. "
        "손실은 BM25만 맞히던 lexical anchor에 몰린다. "
        f"Dense-only 복구는 {retention['dense_only']['recovered']}/"
        f"{retention['dense_only']['prior']}이고, "
        f"both-miss 복구는 {retention['both_miss']['recovered']}/"
        f"{retention['both_miss']['prior']}이다. "
        "both-miss 유입은 성공 조건이 아니다. "
        "MRR, NDCG@10, Recall@5 macro는 BM25보다 낮다. "
        "전체 순위 RRF는 Dense 순위가 깊어도 0이 아닌 reciprocal rank를 더한다. "
        "BM25 1위 양성 unit의 Dense 순위가 수백이면, Dense 1위 irrelevant unit이 앞선다. "
        "이 고정 k의 단순 RRF는 DEV에서 기각한다. "
        "weighted fusion과 query routing은 구현하지 않는다."
    )


def _verdict_prose(verdict: dict[str, Any]) -> str:
    directions = ", ".join(
        f"{name} {direction}" for name, direction in verdict["ranking_directions"].items()
    )
    return (
        f"Hit@10 drop {verdict['hit10_drop']}, "
        f"BM25-only lost {verdict['bm25_only_lost']}, "
        f"Dense-only recovered {verdict['dense_only_recovered']}, "
        f"ranking {directions}."
    )


def _mechanism_note(rows: list[dict[str, Any]]) -> str:
    if not rows or not rows[0].get("rrf_top3"):
        return ""
    row = rows[0]
    top = row["rrf_top3"][0]
    relevant_bm25 = row.get("rrf_first_unit_bm25_rank")
    relevant_dense = row.get("rrf_first_unit_dense_rank")
    if relevant_bm25 is None or relevant_dense is None:
        return ""
    relevant_score = (1.0 / (RRF_K + int(relevant_bm25))) + (
        1.0 / (RRF_K + int(relevant_dense))
    )
    top_score = (1.0 / (RRF_K + int(top["bm25_rank"]))) + (
        1.0 / (RRF_K + int(top["dense_rank"]))
    )
    return (
        f"점수 대조 `{row['query_id']}`: 첫 양성 unit은 BM25 {relevant_bm25}, "
        f"Dense {relevant_dense}, RRF 점수 {relevant_score:.5f}. "
        f"RRF 1위는 relevance {top['relevance']}, BM25 {top['bm25_rank']}, "
        f"Dense {top['dense_rank']}, RRF 점수 {top_score:.5f}."
    )


def _next_experiment(label: str) -> str:
    shared = (
        "both-miss 2건은 period-aware ranking, 연결/별도 scope, "
        "cross-filing retrieval 후보로 남긴다. chunking 실패로 단정하지 않는다. "
        "다음에도 TEST로 k나 weight를 고르지 않는다."
    )
    if label == "A":
        return (
            "DEV에서 관측된 이 RRF를 제품 경로에 바로 넣지 않는다. "
            "미리 등록한 확인은 같은 k=60을 v2.1 TEST에 한 번 평가하는 것이다. "
            "그 전에 k를 바꾸거나 weight를 맞추지 않는다. " + shared
        )
    if label == "B":
        return (
            "검색 baseline은 BM25를 유지하는 안을 검토한다. "
            "단순 RRF를 제품에 넣지 않는다. "
            "추가 결합 규칙은 새 DEV 가설로 분리한다. " + shared
        )
    return (
        "전체 순위 단순 RRF를 기각한다. "
        "반환 밖 contribution을 0으로 두는 변형은 이번 숫자가 판단하지 않는다. "
        "weighted fusion 또는 query routing은 새 DEV 가설로만 남기고 "
        "이번 결과에서 그 규칙을 만들지 않는다. " + shared
    )


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
        "source_run": SOURCE_RUN_DIR,
        "input_dense_sha256": audit["input_artifacts"]["dense_dev_sha256"],
        "input_bm25_sha256": audit["input_artifacts"]["bm25_dev_sha256"],
        "rrf": audit["implementation"],
        "dense": {
            "implementation": "product_catalog_dense_d2a",
            "model": DENSE_MODEL_ID,
            "query_encoding_policy": "raw_question_text_no_instruction",
            "full_ranking": "exact_normalized_search",
        },
        "bm25": {
            "implementation": "disclosure_agent.bm25.Bm25Index.rank_all",
            "headline_implementation": "Bm25Index.search top-20 positive scores",
            "tokenizer_version": TOKENIZER_VERSION,
            "k1": BM25_PARAMS["k1"],
            "b": BM25_PARAMS["b"],
        },
        "forbidden": {
            "weighted_score_fusion": False,
            "alpha_tuning": False,
            "query_routing": False,
            "reranker": False,
            "query_expansion": False,
            "embedding_change": False,
            "chunking_change": False,
            "k_sweep": False,
        },
        "git_commit": audit["git"]["git_commit"],
        "git_dirty": audit["git"]["git_dirty"],
        "head_subject": audit["head_subject"],
        "verdict_label": payload["verdict"]["label"],
    }


def _headline_document(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "table": _headline_rows(payload),
        "oracle": payload["oracle"],
        "verdict": payload["verdict"],
        "ndcg_note": "headline NDCG is condensed. preferred_hit@10 is diagnostic.",
    }


def _headline_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    specs = (
        ("Hit@1", ("at_k", "1", "hit")),
        ("Hit@5", ("at_k", "5", "hit")),
        ("Hit@10", ("at_k", "10", "hit")),
        ("Recall@5 macro", ("recall@5_macro",)),
        ("Recall@5 micro", ("recall@5_micro",)),
        ("Recall@10 macro", ("recall@10_macro",)),
        ("Recall@10 micro", ("recall@10_micro",)),
        ("MRR", ("mrr",)),
        ("NDCG@10", ("at_k", "10", "ndcg_condensed")),
        ("preferred_hit@10", ("at_k", "10", "preferred_hit")),
        ("requirement_recall@10", ("requirement_recall@10",)),
        ("all_requirements_covered@10", ("all_requirements_covered@10",)),
        ("judged_coverage@10", ("at_k", "10", "judged_coverage")),
        ("judged_coverage@20", ("judged_coverage@20",)),
    )
    rows = []
    for label, path in specs:
        item = {"metric": label}
        for name in ("dense", "bm25", "rrf"):
            block = _dig(payload[f"{name}_headline"], path)
            item[f"{name}_numerator"] = block.get("numerator")
            item[f"{name}_denominator"] = block.get("denominator")
            item[f"{name}_value"] = block.get("value")
        rows.append(item)
    return rows


def _count_zero_scores(
    bm25_full: list[dict[str, Any]],
    fused: list[dict[str, Any]],
    docs: int,
    checks: int,
) -> tuple[int, int]:
    by_id = {str(hit["retrieval_unit_id"]): hit for hit in fused}
    spotted = False
    for hit in bm25_full:
        if float(hit["score"]) != 0:
            continue
        docs += 1
        if spotted:
            continue
        fused_hit = by_id[str(hit["retrieval_unit_id"])]
        contribution = 1.0 / (RRF_K + int(fused_hit["bm25_rank"]))
        if contribution <= 0:
            raise UsageError("zero-score BM25 contribution was 0 under reciprocal_rank_fusion")
        checks += 1
        spotted = True
    return docs, checks


def _assert_unit_identity(hits: list[dict[str, Any]], label: str) -> None:
    for hit in hits:
        unit_id = str(hit.get("retrieval_unit_id") or "")
        if not unit_id or unit_id != str(hit.get("chunk_id")):
            raise UsageError(f"{label} chunk_id is not the retrieval_unit_id")


def _assert_known_unit(
    query_id: str,
    label: str,
    unit_id: str | None,
    rank: int | None,
    by_unit: dict[str, dict[str, Any]],
) -> None:
    if unit_id is None:
        return
    hit = by_unit.get(unit_id)
    if hit is None:
        raise UsageError(f"{query_id} stored {label} relevant unit is outside the fused universe")
    if rank is not None and int(hit[f"{label}_rank"]) != int(rank):
        raise UsageError(
            f"{query_id} stored {label} rank {rank} is not the full-ranking rank "
            f"{hit[f'{label}_rank']}"
        )


def _first_relevant_unit(hits: list[dict[str, Any]]) -> str | None:
    for hit in hits:
        relevance = hit.get("relevance")
        if relevance is not None and int(relevance) >= 1:
            return str(hit["retrieval_unit_id"])
    return None


def _assert_pinned_headline(headline: dict[str, Any], label: str) -> None:
    for k, expected in PINNED_HITS[label].items():
        got = int(headline["at_k"][k]["hit"]["numerator"])
        if got != expected:
            raise UsageError(f"stored {label} Hit@{k} is {got}, expected {expected}")
    for k, (numerator, denominator) in PINNED_MICRO[label].items():
        block = headline[f"recall@{k}_micro"]
        if (int(block["numerator"]), int(block["denominator"])) != (numerator, denominator):
            raise UsageError(f"stored {label} Recall@{k} micro changed")
    requirement = PINNED_REQUIREMENT_RECALL[label]
    block = headline["requirement_recall@10"]
    if (int(block["numerator"]), int(block["denominator"])) != requirement:
        raise UsageError(f"stored {label} requirement recall changed")
    covered = PINNED_ALL_REQUIREMENTS[label]
    block = headline["all_requirements_covered@10"]
    if (int(block["numerator"]), int(block["denominator"])) != covered:
        raise UsageError(f"stored {label} all-requirements coverage changed")


def _assert_stored_rank(
    query_id: str,
    label: str,
    hits: list[dict[str, Any]],
    prior: dict[str, Any],
) -> None:
    found: int | None = None
    for hit in hits:
        relevance = hit.get("relevance")
        if relevance is not None and int(relevance) >= 1:
            found = int(hit["rank"])
            break
    expected = prior.get(f"{label}_first_relevant_rank")
    if found != expected:
        raise UsageError(
            f"{query_id} stored {label} first relevant rank {expected} "
            f"does not match the hits ({found})"
        )


def _index_questions(
    rows: list[dict[str, Any]], questions: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_id = {str(row["query_id"]): row for row in rows}
    expected = {str(row["query_id"]) for row in questions}
    if set(by_id) != expected:
        raise UsageError("stored ranking questions are not the DEV 45")
    return by_id


def _requirement_transition(cell: str, covered: bool) -> str:
    if cell == "neither":
        return "recovered" if covered else "still_missing"
    if cell == "dense_only":
        return "recovered" if covered else "lost"
    if cell in {"both", "bm25_only"}:
        return "retained" if covered else "lost"
    raise UsageError(f"unknown requirement cell {cell}")


def _retention_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": row["query_id"],
        "company": row["company"],
        "question": row["question"],
        "dense_first_relevant_rank": row["dense_first_relevant_rank"],
        "bm25_first_relevant_rank": row["bm25_first_relevant_rank"],
        "rrf_first_relevant_rank": row["rrf_first_relevant_rank"],
        "movement": row["movement"],
        "rrf_hit10": row["rrf_hit10"],
        "rrf_first_unit_bm25_rank": row.get("rrf_first_unit_bm25_rank"),
        "rrf_first_unit_dense_rank": row.get("rrf_first_unit_dense_rank"),
        "rrf_first_relevant_preview": row["rrf_first_relevant_preview"],
    }


def _direction(before: float, after: float) -> str:
    if after > before:
        return "up"
    if after < before:
        return "down"
    return "flat"


def _median(values: list[int]) -> float | None:
    if not values:
        return None
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
    return f"{value:.3f}"


def _id_suffix(ids: list[str]) -> str:
    if not ids:
        return ""
    return ": " + ", ".join(ids)


def _query_lines(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "해당 질의가 없다."
    lines = []
    for row in rows:
        lines.append(
            f"- `{row['query_id']}` {row['company']}: "
            f"Dense {row['dense_first_relevant_rank']}, "
            f"BM25 {row['bm25_first_relevant_rank']}, "
            f"RRF {row['rrf_first_relevant_rank']}, "
            f"{row['movement']}. "
            f"그 unit의 전체 순위는 BM25 {row.get('rrf_first_unit_bm25_rank')}, "
            f"Dense {row.get('rrf_first_unit_dense_rank')}. "
            f"{row['question']}"
        )
    return "\n".join(lines)


def _transition_lines(cells: dict[str, Any]) -> str:
    lines = []
    for name in ("both", "dense_only", "bm25_only", "neither"):
        block = cells[name]
        lines.append(
            f"- {name} prior {block['prior']}: "
            f"`{json.dumps(block['transitions'], ensure_ascii=False)}`"
        )
    return "\n".join(lines)


def _tracked_requirement_lines(rows: list[dict[str, Any]]) -> str:
    tracked = {
        TRACKED_QUERIES["samyang_statements"],
        TRACKED_QUERIES["samyang_debt"],
        TRACKED_QUERIES["dense_only"],
    }
    lines = []
    for row in rows:
        if row["query_id"] not in tracked:
            continue
        lines.append(
            f"- `{row['query_id']}` {row['requirement_id']} "
            f"{row['prior_cell']} → {row['transition']}. {row['requirement_text']}"
        )
    return "\n".join(lines)


def _case_lines(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "해당 사례가 없다."
    blocks = []
    for row in rows:
        top = row["rrf_top3"][0] if row["rrf_top3"] else None
        preview = "" if top is None else top["preview"]
        blocks.append(
            "\n".join(
                [
                    f"### {row['query_id']}",
                    "",
                    row["question"] or "",
                    "",
                    (
                        f"회사 {row['company']}. 이전 셀 {row['prior_cell']}. "
                        f"movement {row['movement']}. "
                        f"Dense 첫 양성 {row['dense_first_relevant_rank']}, "
                        f"BM25 첫 양성 {row['bm25_first_relevant_rank']}, "
                        f"RRF 첫 양성 {row['rrf_first_relevant_rank']}. "
                        f"delta vs BM25 {row['rank_delta_vs_bm25']}, "
                        f"delta vs Dense {row['rank_delta_vs_dense']}. "
                        f"같은 BM25 unit의 RRF 순위 {row['bm25_first_unit_rrf_rank']} "
                        f"(delta {row['same_unit_rank_delta']}). "
                        f"RRF 첫 양성 unit의 Dense 순위 {row['rrf_first_unit_dense_rank']}, "
                        f"BM25 순위 {row['rrf_first_unit_bm25_rank']}. "
                        f"상위 5 양성 {row['rrf_relevant_in_top5']}, "
                        f"상위 10 양성 {row['rrf_relevant_in_top10']}."
                    ),
                    "",
                    f"RRF 1위 relevance {None if top is None else top.get('relevance')}: {preview}",
                    "",
                    f"첫 양성 발췌: {row.get('rrf_first_relevant_preview') or ''}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _preview(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


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
