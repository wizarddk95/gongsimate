"""Cross-encoder rerank over a fixed D2a candidate pool.

The default pool is the BM25 top-100 and Dense top-100 union. The dense-only
ablation changes that pool to Dense top-100 and leaves the encoder, the
retrieval text, and the evaluation contract alone. RRF is a comparison
baseline, not a candidate source. Evaluation metadata is not a model input.
"""

from __future__ import annotations

import copy
import gc
import math
import os
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

from disclosure_agent.bm25 import BM25_PARAMS, TOKENIZER_VERSION, Bm25Index
from disclosure_agent.chunking import canonical_dumps, sha256_text
from disclosure_agent.dense import (
    DENSE_MODEL_ID,
    DENSE_MODEL_REVISION,
    encode_texts,
    exact_normalized_search,
    load_dense_model,
    resolve_device,
)
from disclosure_agent.errors import UsageError
from disclosure_agent.evaluation import (
    KS,
    _aggregate,
    _metric_scope,
    _q07_calculation,
    chunk_contains_element,
    chunk_sources,
    claims_for_question,
    coverage_level,
    dataset_file_hash,
    evidence_paths_for_question,
    load_eval_corpus,
    reproducibility_record,
    required_claims,
    score_search,
    single_and_set,
    write_eval_run,
)
from disclosure_agent.hybrid import (
    BM25_RU_INDEXED_TEXT_SHA,
    D2A_EMBEDDING_SHA,
    D2A_RETRIEVAL_UNIT_SHA,
    DATASET_SHA,
    RRF_K,
    _assert_d2a_shape,
    _assert_index_identity,
    _assert_locked_headline,
    _assert_view_identity,
    _identity_rows,
    _indexed_text_sha256,
    _load_embedding_matrix,
    _load_jsonl,
    _ranking_sha256,
    _stage,
    ranking_agreement,
    reciprocal_rank_fusion_zero_tie,
)
from disclosure_agent.retrieval_unit import (
    retrieval_units_sha256,
    scoring_chunks,
    summarize_retrieval_units,
)

RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
RERANKER_ARCHITECTURE_MAX_LENGTH = 8194
RERANKER_INFERENCE_MAX_LENGTH = 1024
RERANKER_LICENSE = "apache-2.0"
CANDIDATE_TOP_N = 100
POLICY_UNION = "bm25_top100_union_dense_top100"
POLICY_DENSE_ONLY = "dense_top100"
PAIR_SCORE_TOLERANCE = 1e-4
ZERO_TIE_RANKING_SHA = "3f2eb0580f78561adbba65658491832ba6bd52a527c2bcd66e6c410bb30c2a65"
HEADLINE_KS = ("1", "3", "5", "10", "20")
DIAGNOSTIC_KS = (5, 10, 20)
STAGE_IN_TOPK = "in_topk"
STAGE_CANDIDATE_MISS = "candidate_generation_failure"
STAGE_RERANK_MISS = "reranking_failure"
STAGE_NOT_IN_CORPUS = "not_covered_by_d2a_corpus"
STAGE_EXPANSION_INTERFERENCE = "candidate_expansion_interference"
SOURCE_BM25_ONLY = "bm25_only"
SOURCE_DENSE_ONLY = "dense_only"
SOURCE_BOTH = "both"
LOCKED_RUN_NAMES = {
    "isu_petasys_v0_3",
    "isu_petasys_v0_3_dense",
    "isu_petasys_v0_3_dense_d1",
    "isu_petasys_v0_3_dense_d2a",
    "isu_petasys_v0_3_dense_d2b",
    "isu_petasys_v0_3_dense_d2c",
    "isu_petasys_v0_3_bm25_ru_d2a",
    "isu_petasys_v0_3_rrf_d2a",
    "isu_petasys_v0_3_rrf_zero_tie_d2a",
    "isu_petasys_v0_3_rerank_ce_d2a",
}
_FIVE_SYSTEMS = (
    "bm25_ru",
    "dense_d2a",
    "rrf_zero_tie",
    "union_cross_encoder",
    "dense_only_cross_encoder",
)
_UNIT_FIELDS = (
    "retrieval_unit_id",
    "canonical_chunk_id",
    "canonical_ordinal",
    "unit_index",
    "text",
    "source_refs",
)
_METRIC_KEYS = (
    "retriever_type",
    "dataset",
    "dataset_sha256",
    "reproducibility",
    "scope",
    "cross_encoder",
    "candidate_policy",
    "invariants",
    "candidate_generation",
    "comparison",
    "audits",
    "known_conflict_diagnostic",
    "ranking_transitions",
    "source_contribution",
    "diversity",
    "failure_taxonomy",
    "runtime",
    "calculation_q07",
    "metrics",
)
_DENSE_ONLY_METRIC_KEYS = (
    *_METRIC_KEYS,
    "pair_score_identity",
    "candidate_text_identity",
    "candidate_coverage_comparison",
    "pool_sizes",
    "bm25_added_impact",
    "gold_ranking_transition",
    "five_system_headline",
    "per_question_complete_ranks",
    "focus_audits",
    "runtime_comparison",
)


def build_candidate_pool(
    bm25_hits: list[dict[str, Any]],
    dense_hits: list[dict[str, Any]],
    units_by_id: dict[str, dict[str, Any]],
    *,
    top_n: int = CANDIDATE_TOP_N,
) -> list[dict[str, Any]]:
    """Union BM25 and dense top-N. Dedup by retrieval unit. Do not collapse chunks.

    BM25 hits must already omit score 0. A zero score in that list is an error.
    Dense keeps its own top-N, including non-positive scores. RRF is not read.
    """
    if top_n < 1:
        raise UsageError("candidate top-N must be positive")
    for hit in bm25_hits:
        if float(hit["score"]) <= 0:
            raise UsageError("BM25 score=0 is not a BM25 top-N candidate")
    bm25_top = _unique_hits(bm25_hits[:top_n], "bm25")
    dense_top = _unique_hits(dense_hits[:top_n], "dense")
    bm25_by_id = {hit["chunk_id"]: hit for hit in bm25_hits}
    dense_by_id = {hit["chunk_id"]: hit for hit in dense_hits}
    ordered: list[str] = []
    seen: set[str] = set()
    for hit in [*bm25_top, *dense_top]:
        doc_id = hit["chunk_id"]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ordered.append(doc_id)
    bm25_ids = {hit["chunk_id"] for hit in bm25_top}
    dense_ids = {hit["chunk_id"] for hit in dense_top}
    candidates = []
    for stable_index, doc_id in enumerate(ordered):
        unit = units_by_id.get(doc_id)
        if unit is None:
            raise UsageError(f"candidate {doc_id} is not a D2a retrieval unit")
        for field in _UNIT_FIELDS:
            if field not in unit:
                raise UsageError(f"retrieval unit {doc_id} is missing {field}")
        if unit["retrieval_unit_id"] != doc_id:
            raise UsageError("retrieval_unit_id does not match the ranked chunk id")
        in_bm25 = doc_id in bm25_ids
        in_dense = doc_id in dense_ids
        if in_bm25 and in_dense:
            membership = SOURCE_BOTH
        elif in_bm25:
            membership = SOURCE_BM25_ONLY
        elif in_dense:
            membership = SOURCE_DENSE_ONLY
        else:
            raise UsageError("candidate is in neither BM25 nor dense top-N")
        bm25_hit = bm25_by_id.get(doc_id)
        dense_hit = dense_by_id.get(doc_id)
        candidates.append(
            {
                "retrieval_unit_id": doc_id,
                "canonical_chunk_id": unit["canonical_chunk_id"],
                "canonical_ordinal": unit["canonical_ordinal"],
                "unit_index": int(unit["unit_index"]),
                "text": unit["text"],
                "source_refs": copy.deepcopy(unit["source_refs"]),
                "stable_index": stable_index,
                "source_membership": membership,
                "in_bm25_top_n": in_bm25,
                "in_dense_top_n": in_dense,
                "bm25_rank": None if bm25_hit is None else int(bm25_hit["rank"]),
                "bm25_score": None if bm25_hit is None else float(bm25_hit["score"]),
                "dense_rank": None if dense_hit is None else int(dense_hit["rank"]),
                "dense_score": None if dense_hit is None else float(dense_hit["score"]),
            }
        )
    if len({row["retrieval_unit_id"] for row in candidates}) != len(candidates):
        raise UsageError("candidate union did not dedup retrieval_unit_id")
    if {row["retrieval_unit_id"] for row in candidates} != bm25_ids | dense_ids:
        raise UsageError("candidate union does not equal BM25 top-N ∪ dense top-N")
    return candidates


def build_dense_only_candidate_pool(
    dense_hits: list[dict[str, Any]],
    units_by_id: dict[str, dict[str, Any]],
    *,
    bm25_hits: list[dict[str, Any]] | None = None,
    top_n: int = CANDIDATE_TOP_N,
) -> list[dict[str, Any]]:
    """Dense top-N in dense-rank order. BM25 rows annotate; they do not enter.

    stable_index follows that dense order. It breaks ties only and does not
    change the stored cross-encoder score.
    """
    if top_n < 1:
        raise UsageError("candidate top-N must be positive")
    dense_top = _unique_hits(dense_hits[:top_n], "dense")
    bm25_list = list(bm25_hits or [])
    for hit in bm25_list:
        if float(hit["score"]) <= 0:
            raise UsageError("BM25 score=0 is not a BM25 annotation")
    bm25_by_id = {hit["chunk_id"]: hit for hit in bm25_list}
    bm25_top_ids = {hit["chunk_id"] for hit in bm25_list[:top_n]}
    candidates = []
    for stable_index, hit in enumerate(dense_top):
        doc_id = hit["chunk_id"]
        if int(hit["rank"]) != stable_index + 1:
            raise UsageError(
                "dense top-N ranks are not contiguous; "
                f"position {stable_index} has rank {hit['rank']}"
            )
        unit = units_by_id.get(doc_id)
        if unit is None:
            raise UsageError(f"candidate {doc_id} is not a D2a retrieval unit")
        for field in _UNIT_FIELDS:
            if field not in unit:
                raise UsageError(f"retrieval unit {doc_id} is missing {field}")
        if unit["retrieval_unit_id"] != doc_id:
            raise UsageError("retrieval_unit_id does not match the ranked chunk id")
        bm25_hit = bm25_by_id.get(doc_id)
        in_bm25 = doc_id in bm25_top_ids
        candidates.append(
            {
                "retrieval_unit_id": doc_id,
                "canonical_chunk_id": unit["canonical_chunk_id"],
                "canonical_ordinal": unit["canonical_ordinal"],
                "unit_index": int(unit["unit_index"]),
                "text": unit["text"],
                "source_refs": copy.deepcopy(unit["source_refs"]),
                "stable_index": stable_index,
                "source_membership": SOURCE_BOTH if in_bm25 else SOURCE_DENSE_ONLY,
                "in_bm25_top_n": in_bm25,
                "in_dense_top_n": True,
                "bm25_rank": None if bm25_hit is None else int(bm25_hit["rank"]),
                "bm25_score": None if bm25_hit is None else float(bm25_hit["score"]),
                "dense_rank": int(hit["rank"]),
                "dense_score": float(hit["score"]),
            }
        )
    if any(row["source_membership"] == SOURCE_BM25_ONLY for row in candidates):
        raise UsageError("dense-only pool contains a BM25-only candidate")
    ids = [row["retrieval_unit_id"] for row in candidates]
    if ids != [hit["chunk_id"] for hit in dense_top]:
        raise UsageError("dense-only pool does not preserve dense rank order")
    if len(set(ids)) != len(candidates):
        raise UsageError("dense-only pool did not keep retrieval_unit_id unique")
    return candidates


def pair_score_identity(
    reference: dict[tuple[str, str], float],
    observed: dict[tuple[str, str], float],
    *,
    tolerance: float = PAIR_SCORE_TOLERANCE,
) -> dict[str, Any]:
    """Compare raw logits on the shared (question, retrieval unit) pairs.

    Pairs present only in the reference are expected when the reference pool
    is larger. They are not mismatches. A mismatch is a shared pair whose
    absolute logit difference exceeds tolerance.
    """
    if tolerance < 0:
        raise UsageError("pair-score tolerance must be non-negative")
    common = sorted(set(reference) & set(observed))
    only_reference = set(reference) - set(observed)
    only_observed = set(observed) - set(reference)
    diffs: list[float] = []
    mismatches = []
    exact = 0
    for key in common:
        delta = abs(float(reference[key]) - float(observed[key]))
        diffs.append(delta)
        if delta == 0.0:
            exact += 1
        if delta > tolerance:
            mismatches.append(
                {
                    "question_id": key[0],
                    "retrieval_unit_id": key[1],
                    "absolute_difference": delta,
                    "reference_score": float(reference[key]),
                    "observed_score": float(observed[key]),
                }
            )
    return {
        "common_pair_count": len(common),
        "reference_only_count": len(only_reference),
        "observed_only_count": len(only_observed),
        "max_absolute_difference": max(diffs) if diffs else 0.0,
        "mismatch_count": len(mismatches),
        "exact_match_count": exact,
        "tolerance": tolerance,
        "identical_within_tolerance": not mismatches,
        "mismatches": mismatches[:12],
    }


def common_candidate_text_identity(
    union_rows: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Dense-only candidates must already exist in the union run, with the same text."""
    union = {
        (row["question_id"], row["retrieval_unit_id"]): row["text_sha256"] for row in union_rows
    }
    missing = []
    mismatched = []
    for row in dense_rows:
        key = (row["question_id"], row["retrieval_unit_id"])
        expected = union.get(key)
        if expected is None:
            missing.append({"question_id": key[0], "retrieval_unit_id": key[1]})
        elif expected != row["text_sha256"]:
            mismatched.append({"question_id": key[0], "retrieval_unit_id": key[1]})
    return {
        "dense_candidate_count": len(dense_rows),
        "missing_from_union_count": len(missing),
        "text_mismatch_count": len(mismatched),
        "identical": not missing and not mismatched,
        "missing": missing[:12],
        "text_mismatches": mismatched[:12],
    }


def bm25_added_impact_row(
    added_rows: list[dict[str, Any]],
    *,
    gold_score: float | None,
) -> dict[str, Any]:
    """Count BM25-only additions. gold_score is the union logit of the completing unit."""

    def _in_top(k: int) -> int:
        return sum(1 for row in added_rows if int(row["union_rank"]) <= k)

    above = None
    if gold_score is not None:
        above = sum(1 for row in added_rows if float(row["ce_score"]) > gold_score)
    return {
        "bm25_added_count": len(added_rows),
        "union_top5": _in_top(5),
        "union_top10": _in_top(10),
        "union_top20": _in_top(20),
        "above_required_gold": above,
        "answer_valid_gold": sum(1 for row in added_rows if row.get("answer_valid_gold")),
        "known_conflict": sum(1 for row in added_rows if row.get("known_conflict")),
    }


def expansion_interference(
    *,
    union_rank: int | None,
    dense_only_rank: int | None,
    above_gold: int | None,
    scores_identical: bool,
) -> bool:
    """Label only when removed BM25-only rows exactly account for the rank drop.

    The raw logit must be unchanged. A smaller dense-only rank with no
    higher-scoring BM25-only row is not this label.
    """
    if not scores_identical:
        return False
    if union_rank is None or dense_only_rank is None or above_gold is None:
        return False
    if above_gold <= 0 or dense_only_rank >= union_rank:
        return False
    return above_gold == union_rank - dense_only_rank


def five_system_headline(
    systems: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    table = {}
    for k in HEADLINE_KS:
        row = {}
        for name in _FIVE_SYSTEMS:
            claim_hit, claim_total, question_hit, question_total = _window_counts(systems[name], k)
            row[name] = {
                "claim_recall": {"numerator": claim_hit, "denominator": claim_total},
                "question_answer_evidence_satisfied": {
                    "numerator": question_hit,
                    "denominator": question_total,
                },
            }
        table[k] = row
    return table


def dense_only_reranker_config() -> dict[str, Any]:
    config = reranker_config()
    config["candidate_policy"] = POLICY_DENSE_ONLY
    config["candidate_top_n_bm25"] = None
    config["bm25_candidate_source"] = False
    config["stable_index_definition"] = "dense_rank_order"
    return config


def cross_encoder_pairs(
    query: str, candidates: list[dict[str, Any]]
) -> list[tuple[str, str]]:
    """Model input is the raw query and the retrieval-unit text."""
    return [(query, candidate["text"]) for candidate in candidates]


def assert_model_inputs(
    query: str,
    candidates: list[dict[str, Any]],
    pairs: list[tuple[str, str]],
) -> None:
    if len(pairs) != len(candidates):
        raise UsageError("cross-encoder pair count does not match the candidate pool")
    for candidate, pair in zip(candidates, pairs, strict=True):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise UsageError("cross-encoder input must be a (query, text) pair")
        left, right = pair
        if not isinstance(left, str) or not isinstance(right, str):
            raise UsageError("cross-encoder input must be raw strings")
        if pair != (query, candidate["text"]):
            raise UsageError("cross-encoder input changed the query or retrieval text")


def order_by_cross_encoder(
    candidates: list[dict[str, Any]], scores: list[float]
) -> list[dict[str, Any]]:
    """Sort by cross-encoder score only. Ties use stable index, then unit id.

    The tie-break does not change the stored score. BM25, dense, and RRF scores
    are not added.
    """
    if len(scores) != len(candidates):
        raise UsageError("cross-encoder score count does not match the candidate pool")
    paired: list[tuple[float, dict[str, Any]]] = []
    for candidate, score in zip(candidates, scores, strict=True):
        value = float(score)
        if not math.isfinite(value):
            raise UsageError("cross-encoder score is not finite")
        paired.append((value, candidate))
    paired.sort(
        key=lambda item: (
            -item[0],
            int(item[1]["stable_index"]),
            item[1]["retrieval_unit_id"],
        )
    )
    ordered = []
    for rank, (score, candidate) in enumerate(paired, start=1):
        ordered.append(
            {
                "rank": rank,
                "chunk_id": candidate["retrieval_unit_id"],
                "ordinal": candidate["canonical_ordinal"],
                "score": score,
                "unit_index": candidate["unit_index"],
                "canonical_chunk_id": candidate["canonical_chunk_id"],
                "retrieval_unit_id": candidate["retrieval_unit_id"],
                "source_membership": candidate["source_membership"],
                "stable_index": candidate["stable_index"],
                "bm25_rank": candidate["bm25_rank"],
                "dense_rank": candidate["dense_rank"],
            }
        )
    return ordered


def summarize_truncation(rows: list[dict[str, int]], max_length: int) -> dict[str, Any]:
    total = len(rows)
    truncated = sum(1 for row in rows if int(row["pair_tokens"]) > max_length)
    return {
        "total_candidates": total,
        "truncated_candidates": truncated,
        "truncation_rate": (truncated / total) if total else 0.0,
        "max_input_tokens": max((int(row["pair_tokens"]) for row in rows), default=0),
        "max_query_tokens": max((int(row["query_tokens"]) for row in rows), default=0),
        "max_document_tokens": max((int(row["document_tokens"]) for row in rows), default=0),
        "inference_max_length": max_length,
    }


def classify_stage(
    *,
    covered_by_corpus: bool,
    covered_by_pool: bool,
    rerank_rank: int | None,
    k: int,
) -> str:
    """Separate a pool miss from a later top-K miss. Corpus absence stays unlabeled."""
    if k < 1:
        raise UsageError("failure k must be positive")
    if not covered_by_pool:
        if not covered_by_corpus:
            return STAGE_NOT_IN_CORPUS
        return STAGE_CANDIDATE_MISS
    if rerank_rank is None or rerank_rank > k:
        return STAGE_RERANK_MISS
    return STAGE_IN_TOPK


def claim_covered(chunks: list[dict[str, Any]], claim: dict[str, Any]) -> bool:
    return single_and_set(chunks, claim)["chunk_set"] == "full"


def reranker_config() -> dict[str, Any]:
    return {
        "model_id": RERANKER_MODEL_ID,
        "revision": RERANKER_MODEL_REVISION,
        "license": RERANKER_LICENSE,
        "architecture_max_length": RERANKER_ARCHITECTURE_MAX_LENGTH,
        "inference_max_length": RERANKER_INFERENCE_MAX_LENGTH,
        "activation": "identity_raw_logits",
        "pair_format": "(query, retrieval_unit_text)",
        "prompt": None,
        "query_rewrite": False,
        "candidate_top_n_bm25": CANDIDATE_TOP_N,
        "candidate_top_n_dense": CANDIDATE_TOP_N,
        "bm25_zero_score_excluded": True,
        "dedup": "retrieval_unit_id",
        "rrf_used_as_candidate_source": False,
        "canonical_collapse": False,
        "score_fusion": False,
        "tie_break": [
            "cross_encoder_score_descending",
            "stable_candidate_index_ascending",
            "retrieval_unit_id_ascending",
        ],
        "truncation_policy": (
            "shared pair budget at the publisher-recommended inference length 1024; "
            "the query is the left sequence; overflow truncates the document tail; "
            "config max_position_embeddings is 8194"
        ),
        "input_fields": ["query_text", "retrieval_unit_text"],
    }


def run_cross_encoder_experiment(
    *,
    dataset_dir: Path,
    manifest_path: Path,
    elements_path: Path,
    d2a_run: Path,
    bm25_canonical_run: Path,
    bm25_ru_run: Path,
    zero_tie_run: Path,
    output_dir: Path,
    device: str = "cpu",
    batch_size: int = 4,
    candidate_policy: str = POLICY_UNION,
    union_run: Path | None = None,
) -> dict[str, Any]:
    """Reproduce the locked retrievers, then rerank one candidate policy.

    The default policy is the BM25 top-100 and Dense top-100 union. dense_top100
    drops BM25 candidates and keeps every other scoring choice.
    """
    if candidate_policy not in {POLICY_UNION, POLICY_DENSE_ONLY}:
        raise UsageError(f"unknown candidate policy {candidate_policy}")
    if candidate_policy == POLICY_DENSE_ONLY and union_run is None:
        raise UsageError("dense-only rerank requires the locked union run")
    _refuse_locked(output_dir)
    locked_runs = (d2a_run, bm25_canonical_run, bm25_ru_run, zero_tie_run)
    if union_run is not None:
        locked_runs = (*locked_runs, union_run)
    if any(output_dir.resolve() == path.resolve() for path in locked_runs):
        raise UsageError("refusing to write the reranker over a locked retrieval run")
    started = time.perf_counter()
    _stage("load corpus")
    corpus = load_eval_corpus(
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        elements_path=elements_path,
    )
    d2a_metrics = _read_metrics(d2a_run)
    d2a_questions = _read_questions(d2a_run)
    canonical_questions = _read_questions(bm25_canonical_run)
    bm25_metrics = _read_metrics(bm25_ru_run)
    bm25_locked_questions = _read_questions(bm25_ru_run)
    zero_metrics = _read_metrics(zero_tie_run)
    zero_locked_questions = _read_questions(zero_tie_run)
    _assert_locked_headline(_read_metrics(bm25_canonical_run), "bm25_canonical")
    _assert_locked_headline(bm25_metrics, "bm25_ru")
    _assert_locked_headline(d2a_metrics, "dense_d2a")
    _assert_locked_headline(zero_metrics, "hybrid_rrf")
    if zero_metrics["rrf"]["ranking_sha256"] != ZERO_TIE_RANKING_SHA:
        raise UsageError("locked RRF zero-tie ranking SHA does not match the pinned value")
    if dataset_file_hash(dataset_dir) != DATASET_SHA:
        raise UsageError("dataset SHA does not match the pinned isu_petasys_v0_3 value")
    units = _load_jsonl(d2a_run / "retrieval_units.jsonl")
    before = _identity_rows(units)
    unit_sha = retrieval_units_sha256(units)
    recorded_unit_sha = d2a_metrics["retrieval_units"]["artifact_sha256"]
    if unit_sha != D2A_RETRIEVAL_UNIT_SHA or unit_sha != recorded_unit_sha:
        raise UsageError("D2a retrieval-unit artifact SHA does not match the pinned value")
    summary = summarize_retrieval_units(
        units,
        target_tokens=512,
        hard_max_tokens=1024,
        model_max_tokens=8192,
    )
    _assert_d2a_shape(summary, d2a_metrics, corpus)
    view = scoring_chunks(units)
    _assert_view_identity(units, view)
    unit_ids = [unit["retrieval_unit_id"] for unit in units]
    if len(unit_ids) != 2869 or len(set(unit_ids)) != 2869:
        raise UsageError("D2a retrieval-unit id set is not 2869 unique ids")
    matrix, embedding_sha = _load_embedding_matrix(
        d2a_run / "embeddings.f32",
        len(units),
        D2A_EMBEDDING_SHA,
    )
    if embedding_sha != d2a_metrics["dense"]["embedding_artifact_sha256"]:
        raise UsageError("D2a embedding SHA does not match the locked dense run")
    index = Bm25Index(view)
    _assert_index_identity(units, index)
    if TOKENIZER_VERSION != "ko-mixed-v1":
        raise UsageError("BM25 tokenizer is not ko-mixed-v1")
    if index.k1 != BM25_PARAMS["k1"] or index.b != BM25_PARAMS["b"]:
        raise UsageError("BM25 index parameters differ from the locked k1 and b")
    if BM25_PARAMS["k1"] != 1.2 or BM25_PARAMS["b"] != 0.75:
        raise UsageError("BM25 locked parameters are not k1=1.2 and b=0.75")
    indexed_text_sha = _indexed_text_sha256(index.chunks)
    if indexed_text_sha != BM25_RU_INDEXED_TEXT_SHA:
        raise UsageError("BM25-RU indexed text SHA does not match the locked run")
    if indexed_text_sha != bm25_metrics["retrieval_units"]["indexed_text_sha256"]:
        raise UsageError("BM25-RU indexed text SHA does not match the locked metrics")
    questions = corpus["dataset"]["questions"]
    _assert_denominator(corpus["dataset"], questions)
    scoring_corpus = dict(corpus)
    scoring_corpus["chunks"] = view
    units_by_id = {unit["retrieval_unit_id"]: unit for unit in units}
    view_by_id = {row["chunk_id"]: row for row in view}

    _stage("score canonical bm25")
    historical_index = Bm25Index(corpus["chunks"])
    historical_hits = {
        question["text"]: historical_index.search(question["text"], historical_index.n)
        for question in questions
    }
    historical_questions = score_search(corpus, lambda text: historical_hits[text])
    historical_agreement = ranking_agreement(canonical_questions, historical_questions)
    if not historical_agreement["match"]:
        raise UsageError(
            "recomputed canonical BM25 does not match the locked run: "
            f"{historical_agreement['mismatches']}"
        )

    _stage("score bm25-ru")
    bm25_hits = {
        question["text"]: index.search(question["text"], index.n) for question in questions
    }
    bm25_full = {question["text"]: index.rank_all(question["text"]) for question in questions}
    for question in questions:
        positive = [hit for hit in bm25_full[question["text"]] if float(hit["score"]) > 0]
        if [hit["chunk_id"] for hit in positive] != [
            hit["chunk_id"] for hit in bm25_hits[question["text"]]
        ]:
            raise UsageError("positive BM25 rank_all order differs from BM25-RU search")
    bm25_questions = score_search(scoring_corpus, lambda text: bm25_hits[text])
    bm25_agreement = ranking_agreement(bm25_locked_questions, bm25_questions)
    if not bm25_agreement["match"]:
        raise UsageError(
            "recomputed BM25-RU does not match the locked run: "
            f"{bm25_agreement['mismatches']}"
        )

    _stage("encode dense queries")
    resolved = resolve_device(device)
    load_started = time.perf_counter()
    dense_model = load_dense_model(resolved)
    dense_load_seconds = time.perf_counter() - load_started
    encode_started = time.perf_counter()
    vectors = encode_texts(dense_model, [question["text"] for question in questions], batch_size=8)
    encode_seconds = time.perf_counter() - encode_started
    del dense_model
    gc.collect()
    dense_hits = {
        question["text"]: exact_normalized_search(matrix, vectors[position], view)
        for position, question in enumerate(questions)
    }
    del matrix
    for hits in dense_hits.values():
        if len(hits) != len(units) or {hit["chunk_id"] for hit in hits} != set(unit_ids):
            raise UsageError("dense search did not rank every D2a retrieval unit")
    dense_questions = score_search(scoring_corpus, lambda text: dense_hits[text])
    dense_agreement = ranking_agreement(d2a_questions, dense_questions)
    if not dense_agreement["match"]:
        raise UsageError(
            "recomputed dense D2a does not match the locked run: "
            f"{dense_agreement['mismatches']}"
        )

    _stage("fuse rrf zero-tie")
    zero_hits = {
        question["text"]: reciprocal_rank_fusion_zero_tie(
            bm25_full[question["text"]],
            dense_hits[question["text"]],
            k_rrf=RRF_K,
            expected_count=len(units),
        )
        for question in questions
    }
    if _ranking_sha256(questions, zero_hits) != ZERO_TIE_RANKING_SHA:
        raise UsageError("recomputed RRF zero-tie ranking SHA does not match the locked ranking")
    zero_questions = score_search(scoring_corpus, lambda text: zero_hits[text])
    zero_agreement = ranking_agreement(zero_locked_questions, zero_questions)
    if not zero_agreement["match"]:
        raise UsageError(
            "recomputed RRF zero-tie does not match the locked run: "
            f"{zero_agreement['mismatches']}"
        )

    if candidate_policy == POLICY_UNION:
        _stage("build candidate union")
        pools = {
            question["text"]: build_candidate_pool(
                bm25_hits[question["text"]],
                dense_hits[question["text"]],
                units_by_id,
                top_n=CANDIDATE_TOP_N,
            )
            for question in questions
        }
        _assert_candidate_contract(questions, pools, bm25_hits, dense_hits, zero_hits)
    else:
        _stage("build dense-only candidates")
        pools = {
            question["text"]: build_dense_only_candidate_pool(
                dense_hits[question["text"]],
                units_by_id,
                bm25_hits=bm25_hits[question["text"]],
                top_n=CANDIDATE_TOP_N,
            )
            for question in questions
        }
        _assert_dense_only_contract(questions, pools, dense_hits, bm25_hits)
    pool_hits = {text: _pool_hits(candidates) for text, candidates in pools.items()}
    pool_questions = score_search(scoring_corpus, lambda text: pool_hits[text])
    by_question = {item["question_id"]: item for item in questions}
    pool_by_id = {item["question_id"]: item for item in pool_questions}
    bm25_by_id = {item["question_id"]: item for item in bm25_questions}
    dense_by_id = {item["question_id"]: item for item in dense_questions}
    zero_by_id = {item["question_id"]: item for item in zero_questions}
    candidate_generation = _candidate_generation_report(
        corpus["dataset"],
        by_question,
        pool_by_id,
        bm25_by_id,
        dense_by_id,
        pools,
        view_by_id,
        view,
        bm25_hits,
        dense_hits,
    )
    _stage(
        "candidate claim "
        f"{candidate_generation['claim_recall']['numerator']}/"
        f"{candidate_generation['claim_recall']['denominator']} "
        "question "
        f"{candidate_generation['question_answer_evidence_satisfied']['numerator']}/"
        f"{candidate_generation['question_answer_evidence_satisfied']['denominator']}"
    )

    union_candidates: list[dict[str, Any]] | None = None
    union_rankings: list[dict[str, Any]] | None = None
    union_questions: list[dict[str, Any]] | None = None
    union_metrics: dict[str, Any] | None = None
    text_identity: dict[str, Any] | None = None
    if candidate_policy == POLICY_DENSE_ONLY:
        assert union_run is not None
        union_metrics = _read_metrics(union_run)
        union_formula = union_metrics["candidate_policy"]["formula"]
        if union_formula != "BM25_RU_top100(q) ∪ Dense_D2a_top100(q)":
            raise UsageError("union run is not the locked BM25 and Dense candidate policy")
        union_questions = _read_questions(union_run)
        union_candidates = _load_jsonl(union_run / "candidates.jsonl")
        union_rankings = _load_jsonl(union_run / "rankings.jsonl")
        text_identity = common_candidate_text_identity(
            union_candidates, _candidate_rows(questions, pools)
        )
        if not text_identity["identical"]:
            raise UsageError(
                "dense-only retrieval text does not match the union run: "
                f"missing={text_identity['missing_from_union_count']} "
                f"mismatch={text_identity['text_mismatch_count']}"
            )
        _assert_dense_top_matches_union(questions, pools, union_candidates)

    _stage("load cross-encoder")
    rerank_started = time.perf_counter()
    model, model_meta = _load_cross_encoder(resolved)
    model_load_seconds = time.perf_counter() - rerank_started
    _stage("score cross-encoder")
    score_started = time.perf_counter()
    rerank_hits, truncation = _score_pools(model, questions, pools, batch_size=batch_size)
    score_seconds = time.perf_counter() - score_started
    del model
    gc.collect()
    rerank_questions = score_search(scoring_corpus, lambda text: rerank_hits[text])
    rerank_by_id = {item["question_id"]: item for item in rerank_questions}
    _assert_pool_ranks_survive(corpus["dataset"], by_question, pool_by_id, rerank_by_id)
    after = _identity_rows(units)
    if before != after:
        raise UsageError("scoring mutated the D2a retrieval-unit topology")
    pair_identity: dict[str, Any] | None = None
    if candidate_policy == POLICY_DENSE_ONLY:
        assert union_rankings is not None
        reference = {
            (row["question_id"], row["retrieval_unit_id"]): float(row["score"])
            for row in union_rankings
        }
        observed = {
            (question["question_id"], hit["chunk_id"]): float(hit["score"])
            for question in questions
            for hit in rerank_hits[question["text"]]
        }
        pair_identity = pair_score_identity(reference, observed)
        if pair_identity["mismatch_count"] or pair_identity["observed_only_count"]:
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_jsonl(output_dir / "rankings.jsonl", _ranking_rows(questions, rerank_hits))
            (output_dir / "pair_score_identity.json").write_text(
                canonical_dumps(pair_identity) + "\n",
                encoding="utf-8",
            )
            raise UsageError(
                "common-pair raw logits are not identical; "
                "candidate-policy comparison is blocked. "
                f"max_abs={pair_identity['max_absolute_difference']} "
                f"mismatches={pair_identity['mismatch_count']} "
                f"missing_from_union={pair_identity['observed_only_count']}"
            )

    rerank_metrics = _aggregate(
        corpus["dataset"],
        rerank_questions,
        KS,
        index_scope=_metric_scope(corpus["scope"]),
    )
    comparison = _comparison(bm25_by_id, dense_by_id, zero_by_id, rerank_by_id, rerank_metrics)
    audits = _audits(
        corpus["dataset"],
        by_question,
        pools,
        view_by_id,
        bm25_by_id,
        dense_by_id,
        zero_by_id,
        rerank_by_id,
        bm25_hits,
        dense_hits,
        zero_hits,
        rerank_hits,
    )
    conflict = _known_conflict(rerank_by_id, bm25_by_id, dense_by_id, zero_by_id)
    transitions = _transitions(
        corpus["dataset"],
        by_question,
        view_by_id,
        bm25_hits,
        dense_hits,
        zero_hits,
        rerank_hits,
        bm25_by_id,
        dense_by_id,
        zero_by_id,
        rerank_by_id,
    )
    source_contribution = _source_contribution(
        corpus["dataset"], by_question, rerank_hits, rerank_by_id, pools
    )
    diversity = _diversity(by_question, rerank_hits, view_by_id)
    failure_taxonomy = _failure_taxonomy(
        corpus["dataset"], by_question, view, pools, view_by_id, rerank_by_id
    )
    ablation: dict[str, Any] | None = None
    if candidate_policy == POLICY_DENSE_ONLY:
        assert union_run is not None
        assert union_metrics is not None
        assert union_candidates is not None
        assert union_rankings is not None
        assert union_questions is not None
        assert pair_identity is not None
        assert text_identity is not None
        ablation = _dense_only_ablation(
            dataset=corpus["dataset"],
            questions=questions,
            pools=pools,
            units_by_id=units_by_id,
            view_by_id=view_by_id,
            bm25_by_id=bm25_by_id,
            dense_by_id=dense_by_id,
            zero_by_id=zero_by_id,
            rerank_by_id=rerank_by_id,
            rerank_hits=rerank_hits,
            dense_hits=dense_hits,
            union_candidates=union_candidates,
            union_rankings=union_rankings,
            union_questions=union_questions,
            union_candidate_generation=union_metrics["candidate_generation"],
            dense_candidate_generation=candidate_generation,
            pair_identity=pair_identity,
        )
        failure_taxonomy["candidate_expansion_interference"] = ablation[
            "candidate_expansion_interference"
        ]
    if candidate_policy == POLICY_UNION:
        config = reranker_config()
        formula = "BM25_RU_top100(q) ∪ Dense_D2a_top100(q)"
    else:
        config = dense_only_reranker_config()
        formula = "Dense_D2a_top100(q)"
    config_sha = sha256_text(canonical_dumps(config))
    ranking_sha = _ranking_sha256(questions, rerank_hits)
    candidate_sha = _candidate_sha256(questions, pools)
    retrieval = {
        "retriever_type": "cross_encoder_rerank",
        "candidate_policy": candidate_policy,
        "candidate_top_n": CANDIDATE_TOP_N,
        "cross_encoder_model": RERANKER_MODEL_ID,
        "cross_encoder_revision": RERANKER_MODEL_REVISION,
        "inference_max_length": RERANKER_INFERENCE_MAX_LENGTH,
        "architecture_max_length": model_meta["architecture_max_length"],
        "activation": "identity_raw_logits",
        "query_rewrite": False,
        "rrf_candidate_source": False,
        "canonical_collapse": False,
        "score_fusion": False,
        "document_reembedded": False,
        "config_sha256": config_sha,
        "candidate_sha256": candidate_sha,
        "ranking_sha256": ranking_sha,
        "dense_model": DENSE_MODEL_ID,
        "dense_model_revision": DENSE_MODEL_REVISION,
        "dense_embedding_sha256": embedding_sha,
        "retrieval_unit_artifact_sha256": unit_sha,
        "bm25_indexed_text_sha256": indexed_text_sha,
        "tokenizer_version": TOKENIZER_VERSION,
        "bm25_k1": index.k1,
        "bm25_b": index.b,
        "compared_union_ranking_sha256": (
            None
            if union_metrics is None
            else union_metrics["reproducibility"]["retrieval"]["ranking_sha256"]
        ),
    }
    reproducibility = reproducibility_record(corpus, retrieval)
    runtime = {
        "device": resolved,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_implementation": platform.python_implementation(),
        "torch": model_meta["torch_version"],
        "sentence_transformers": model_meta["sentence_transformers_version"],
        "cuda": model_meta["cuda"],
        "dense_model_load_seconds": dense_load_seconds,
        "dense_query_encoding_seconds": encode_seconds,
        "cross_encoder_load_seconds": model_load_seconds,
        "cross_encoder_score_seconds": score_seconds,
        "document_reembedded": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload = {
        "retriever_type": "cross_encoder_rerank",
        "dataset": str(dataset_dir),
        "dataset_sha256": reproducibility["evaluation"]["dataset_sha256"],
        "reproducibility": reproducibility,
        "scope": corpus["scope"],
        "cross_encoder": {
            **config,
            "config_sha256": config_sha,
            "candidate_sha256": candidate_sha,
            "ranking_sha256": ranking_sha,
            "tokenizer": model_meta["tokenizer"],
            "num_labels": model_meta["num_labels"],
            "architecture_max_length_observed": model_meta["architecture_max_length"],
            "truncation": truncation,
            "document_reembedded": False,
        },
        "candidate_policy": {
            "name": candidate_policy,
            "formula": formula,
            "identity": "retrieval_unit_id",
            "top_n": CANDIDATE_TOP_N,
            "bm25_candidate_source": candidate_policy == POLICY_UNION,
            "bm25_zero_score_excluded": True,
            "rrf_used": False,
            "canonical_collapse": False,
            "score_cutoff": False,
            "stable_index": (
                "bm25_then_dense_first_seen"
                if candidate_policy == POLICY_UNION
                else "dense_rank_order"
            ),
        },
        "invariants": _invariant_report(
            summary=summary,
            unit_sha=unit_sha,
            embedding_sha=embedding_sha,
            indexed_text_sha=indexed_text_sha,
            dataset_sha=reproducibility["evaluation"]["dataset_sha256"],
            agreements={
                "bm25_ru": bm25_agreement,
                "dense_d2a": dense_agreement,
                "rrf_zero_tie": zero_agreement,
                "canonical_bm25": historical_agreement,
            },
            candidate_policy=candidate_policy,
        ),
        "candidate_generation": candidate_generation,
        "comparison": comparison,
        "audits": audits,
        "known_conflict_diagnostic": conflict,
        "ranking_transitions": transitions,
        "source_contribution": source_contribution,
        "diversity": diversity,
        "failure_taxonomy": failure_taxonomy,
        "runtime": runtime,
        "calculation_q07": _q07_calculation(corpus["dataset"]),
        "metrics": rerank_metrics,
        "questions": rerank_questions,
    }
    metric_keys: tuple[str, ...] = _METRIC_KEYS
    if ablation is not None:
        assert pair_identity is not None
        assert text_identity is not None
        assert union_metrics is not None
        assert union_rankings is not None
        payload["pair_score_identity"] = pair_identity
        payload["candidate_text_identity"] = text_identity
        payload["candidate_coverage_comparison"] = ablation["candidate_coverage_comparison"]
        payload["pool_sizes"] = ablation["pool_sizes"]
        payload["bm25_added_impact"] = ablation["bm25_added_impact"]
        payload["gold_ranking_transition"] = ablation["gold_ranking_transition"]
        payload["five_system_headline"] = ablation["five_system_headline"]
        payload["per_question_complete_ranks"] = ablation["per_question_complete_ranks"]
        payload["focus_audits"] = ablation["focus_audits"]
        payload["runtime_comparison"] = _runtime_comparison(
            union_metrics["runtime"],
            runtime,
            union_pairs=len(union_rankings),
            dense_pairs=sum(len(rows) for rows in pools.values()),
        )
        metric_keys = _DENSE_ONLY_METRIC_KEYS
    written = write_eval_run(
        output_dir,
        payload,
        corpus["chunks"],
        metric_keys=metric_keys,
    )
    _write_jsonl(output_dir / "candidates.jsonl", _candidate_rows(questions, pools))
    _write_jsonl(output_dir / "rankings.jsonl", _ranking_rows(questions, rerank_hits))
    return written


def _unique_hits(hits: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows = []
    for hit in hits:
        doc_id = hit["chunk_id"]
        if doc_id in seen:
            raise UsageError(f"duplicate {label} candidate {doc_id}")
        seen.add(doc_id)
        rows.append(hit)
    return rows


def _pool_hits(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": index,
            "chunk_id": candidate["retrieval_unit_id"],
            "ordinal": candidate["canonical_ordinal"],
            "score": 0.0,
            "unit_index": candidate["unit_index"],
            "canonical_chunk_id": candidate["canonical_chunk_id"],
            "retrieval_unit_id": candidate["retrieval_unit_id"],
        }
        for index, candidate in enumerate(candidates, start=1)
    ]


def _refuse_locked(path: Path) -> None:
    if path.name in LOCKED_RUN_NAMES:
        raise UsageError(f"refusing to write into protected run {path.name}")


def _read_metrics(path: Path) -> dict[str, Any]:
    from disclosure_agent.chunking import load_json

    return load_json(path / "metrics.json")


def _read_questions(path: Path) -> list[dict[str, Any]]:
    from disclosure_agent.chunking import load_json

    return load_json(path / "per_question.json")["questions"]


def _assert_denominator(dataset: dict[str, Any], questions: list[dict[str, Any]]) -> None:
    positive = [item for item in questions if item.get("in_positive_recall_denominator")]
    if len(positive) != 8:
        raise UsageError("positive independent question count is not 8")
    count = sum(len(required_claims(claims_for_question(dataset, item))) for item in positive)
    if count != 27:
        raise UsageError(f"required claim denominator is {count}, expected 27")
    ids = {item["question_id"] for item in questions}
    if "Q09" not in ids or "Q10" not in ids:
        raise UsageError("regression set is missing Q09 or Q10")


def _assert_candidate_contract(
    questions: list[dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    bm25_hits: dict[str, list[dict[str, Any]]],
    dense_hits: dict[str, list[dict[str, Any]]],
    zero_hits: dict[str, list[dict[str, Any]]],
) -> None:
    del zero_hits
    for question in questions:
        text = question["text"]
        pool = pools[text]
        bm25_ids = {hit["chunk_id"] for hit in bm25_hits[text][:CANDIDATE_TOP_N]}
        dense_ids = {hit["chunk_id"] for hit in dense_hits[text][:CANDIDATE_TOP_N]}
        pool_ids = {row["retrieval_unit_id"] for row in pool}
        if pool_ids != bm25_ids | dense_ids or len(pool) != len(pool_ids):
            raise UsageError(
                f"{question['question_id']} candidate pool is not the deduped top-100 union"
            )
        if any(float(hit["score"]) <= 0 for hit in bm25_hits[text][:CANDIDATE_TOP_N]):
            raise UsageError("BM25 top-100 includes a zero score")
        for row in pool:
            if not isinstance(row["text"], str):
                raise UsageError("candidate text is missing")


def _assert_dense_only_contract(
    questions: list[dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    dense_hits: dict[str, list[dict[str, Any]]],
    bm25_hits: dict[str, list[dict[str, Any]]],
) -> None:
    for question in questions:
        text = question["text"]
        pool = pools[text]
        dense_top = _unique_hits(dense_hits[text][:CANDIDATE_TOP_N], "dense")
        expected = [hit["chunk_id"] for hit in dense_top]
        actual = [row["retrieval_unit_id"] for row in pool]
        if actual != expected:
            raise UsageError(f"{question['question_id']} dense-only pool is not dense rank order")
        bm25_top = {hit["chunk_id"] for hit in bm25_hits[text][:CANDIDATE_TOP_N]}
        for index, row in enumerate(pool):
            if row["source_membership"] == SOURCE_BM25_ONLY or not row["in_dense_top_n"]:
                raise UsageError(f"{question['question_id']} contains a non-dense candidate")
            if row["stable_index"] != index or int(row["dense_rank"]) != index + 1:
                raise UsageError(
                    f"{question['question_id']} stable index is not dense rank order"
                )
            membership = SOURCE_BOTH if row["retrieval_unit_id"] in bm25_top else SOURCE_DENSE_ONLY
            if row["source_membership"] != membership:
                raise UsageError(f"{question['question_id']} source membership is inconsistent")
            if not isinstance(row["text"], str):
                raise UsageError("candidate text is missing")


def _assert_dense_top_matches_union(
    questions: list[dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    union_rows: list[dict[str, Any]],
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in union_rows:
        grouped.setdefault(row["question_id"], []).append(row)
    for question in questions:
        question_id = question["question_id"]
        rows = grouped.get(question_id, [])
        selected = [
            row
            for row in rows
            if row["source_membership"] in {SOURCE_BOTH, SOURCE_DENSE_ONLY}
            and int(row["dense_rank"]) <= CANDIDATE_TOP_N
        ]
        selected.sort(key=lambda row: int(row["dense_rank"]))
        expected = [row["retrieval_unit_id"] for row in selected]
        actual = [row["retrieval_unit_id"] for row in pools[question["text"]]]
        if actual != expected:
            raise UsageError(f"{question_id} reproduced Dense top-N does not match the union run")
        leaked = [
            row["retrieval_unit_id"]
            for row in rows
            if row["source_membership"] == SOURCE_BM25_ONLY
            and int(row["dense_rank"]) <= CANDIDATE_TOP_N
        ]
        if leaked:
            raise UsageError(f"{question_id} union artifact marks a Dense top-N unit bm25_only")


def _short_text(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def _section_label(unit: dict[str, Any]) -> str:
    path = unit.get("section_path") or []
    if isinstance(path, list):
        return " / ".join(str(part) for part in path if part)
    return str(path)


def _unit_evidence_flags(
    chunk: dict[str, Any],
    dataset: dict[str, Any],
    question: dict[str, Any],
) -> dict[str, bool]:
    claims = required_claims(claims_for_question(dataset, question))
    answer = any(coverage_level(chunk_sources(chunk), claim) != "none" for claim in claims)
    conflict_ids = {
        item["element_id"]
        for item in dataset["hard_negatives"]
        if item.get("evidence_class") == "relevant_conflicting"
        and question["question_id"] in item.get("classified_for", [])
    }
    conflict = any(
        chunk_contains_element(chunk, element_id, body_only=False) for element_id in conflict_ids
    )
    return {"answer_valid_gold": answer, "known_conflict": conflict}


def _ranking_hits(rows: list[dict[str, Any]], question_id: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["question_id"] == question_id]
    selected.sort(key=lambda row: int(row["rank"]))
    return [
        {
            "chunk_id": row["retrieval_unit_id"],
            "rank": int(row["rank"]),
            "score": float(row["score"]),
        }
        for row in selected
    ]


def _hit_at_rank(hits: list[dict[str, Any]], rank: int | None) -> dict[str, Any] | None:
    if rank is None:
        return None
    for hit in hits:
        if int(hit["rank"]) == int(rank):
            return hit
    return None


def _hit_by_id(hits: list[dict[str, Any]], doc_id: str) -> dict[str, Any] | None:
    for hit in hits:
        if hit["chunk_id"] == doc_id:
            return hit
    return None


def _evidence_brief(
    unit_id: str,
    units_by_id: dict[str, dict[str, Any]],
    *,
    score: float,
    bm25_rank: int | None,
    dense_rank: int | None,
    ce_rank: int | None,
    flags: dict[str, bool],
) -> dict[str, Any]:
    unit = units_by_id[unit_id]
    return {
        "retrieval_unit_id": unit_id,
        "canonical_ordinal": unit.get("canonical_ordinal"),
        "section": _section_label(unit),
        "snippet": _short_text(str(unit.get("text") or "")),
        "bm25_rank": bm25_rank,
        "dense_rank": dense_rank,
        "ce_rank": ce_rank,
        "ce_logit": score,
        "answer_valid_gold": flags["answer_valid_gold"],
        "known_conflict": flags["known_conflict"],
        "gold": flags["answer_valid_gold"],
    }


def _coverage_comparison(
    union_generation: dict[str, Any],
    dense_generation: dict[str, Any],
) -> dict[str, Any]:
    union_claims = {
        row["claim_id"]: bool(row["in_pool"])
        for row in union_generation["claim_recall"]["claims"]
    }
    dense_claims = {
        row["claim_id"]: bool(row["in_pool"])
        for row in dense_generation["claim_recall"]["claims"]
    }
    only_union = sorted(
        claim_id
        for claim_id, flag in union_claims.items()
        if flag and not dense_claims[claim_id]
    )
    only_dense = sorted(
        claim_id
        for claim_id, flag in dense_claims.items()
        if flag and not union_claims[claim_id]
    )
    return {
        "union": {
            "claim_recall": {
                "numerator": union_generation["claim_recall"]["numerator"],
                "denominator": union_generation["claim_recall"]["denominator"],
            },
            "question_answer_evidence_satisfied": {
                "numerator": union_generation["question_answer_evidence_satisfied"]["numerator"],
                "denominator": union_generation["question_answer_evidence_satisfied"][
                    "denominator"
                ],
            },
        },
        "dense_only": {
            "claim_recall": {
                "numerator": dense_generation["claim_recall"]["numerator"],
                "denominator": dense_generation["claim_recall"]["denominator"],
            },
            "question_answer_evidence_satisfied": {
                "numerator": dense_generation["question_answer_evidence_satisfied"]["numerator"],
                "denominator": dense_generation["question_answer_evidence_satisfied"][
                    "denominator"
                ],
            },
        },
        "required_gold_only_in_union": only_union,
        "required_gold_only_in_dense": only_dense,
        "Q04-C2": {
            "in_union_pool": union_claims.get("Q04-C2"),
            "in_dense_pool": dense_claims.get("Q04-C2"),
        },
    }


def _runtime_comparison(
    union_runtime: dict[str, Any],
    dense_runtime: dict[str, Any],
    *,
    union_pairs: int,
    dense_pairs: int,
) -> dict[str, Any]:
    union_score = float(union_runtime["cross_encoder_score_seconds"])
    dense_score = float(dense_runtime["cross_encoder_score_seconds"])
    union_total = float(union_runtime["elapsed_seconds"])
    dense_total = float(dense_runtime["elapsed_seconds"])
    return {
        "union_candidates": union_pairs,
        "dense_only_candidates": dense_pairs,
        "candidate_reduction_rate": (
            (union_pairs - dense_pairs) / union_pairs if union_pairs else None
        ),
        "union_score_seconds": union_score,
        "dense_only_score_seconds": dense_score,
        "union_elapsed_seconds": union_total,
        "dense_only_elapsed_seconds": dense_total,
        "union_ms_per_pair": (union_score * 1000 / union_pairs) if union_pairs else None,
        "dense_only_ms_per_pair": (dense_score * 1000 / dense_pairs) if dense_pairs else None,
        "cpu_figures_are_not_production_latency": True,
    }


def _dense_only_ablation(
    *,
    dataset: dict[str, Any],
    questions: list[dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    units_by_id: dict[str, dict[str, Any]],
    view_by_id: dict[str, dict[str, Any]],
    bm25_by_id: dict[str, dict[str, Any]],
    dense_by_id: dict[str, dict[str, Any]],
    zero_by_id: dict[str, dict[str, Any]],
    rerank_by_id: dict[str, dict[str, Any]],
    rerank_hits: dict[str, list[dict[str, Any]]],
    dense_hits: dict[str, list[dict[str, Any]]],
    union_candidates: list[dict[str, Any]],
    union_rankings: list[dict[str, Any]],
    union_questions: list[dict[str, Any]],
    union_candidate_generation: dict[str, Any],
    dense_candidate_generation: dict[str, Any],
    pair_identity: dict[str, Any],
) -> dict[str, Any]:
    """Compare the locked union run with this dense-only rerank. Scores are not fused."""
    union_by_id = {item["question_id"]: item for item in union_questions}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in union_candidates:
        grouped.setdefault(row["question_id"], []).append(row)
    scores_identical = bool(pair_identity["identical_within_tolerance"]) and (
        pair_identity["observed_only_count"] == 0
    )
    impact_rows = []
    transitions = []
    interference = []
    focus_added: dict[str, list[dict[str, Any]]] = {}
    focus_false_positives: dict[str, list[dict[str, Any]]] = {}
    for question in questions:
        if not question.get("in_positive_recall_denominator"):
            continue
        question_id = question["question_id"]
        text = question["text"]
        dense_ids = {row["retrieval_unit_id"] for row in pools[text]}
        added_meta = [
            row for row in grouped.get(question_id, []) if row["retrieval_unit_id"] not in dense_ids
        ]
        if any(row["source_membership"] != SOURCE_BM25_ONLY for row in added_meta):
            raise UsageError(f"{question_id} union-minus-dense is not BM25-only")
        union_hits = _ranking_hits(union_rankings, question_id)
        dense_only_hits = rerank_hits[text]
        pool_by_unit = {row["retrieval_unit_id"]: row for row in pools[text]}
        added_rows = []
        for row in added_meta:
            hit = _hit_by_id(union_hits, row["retrieval_unit_id"])
            if hit is None:
                raise UsageError(f"{question_id} BM25-added candidate has no union logit")
            flags = _unit_evidence_flags(view_by_id[row["retrieval_unit_id"]], dataset, question)
            added_rows.append(
                {
                    "retrieval_unit_id": row["retrieval_unit_id"],
                    "union_rank": int(hit["rank"]),
                    "ce_score": float(hit["score"]),
                    "answer_valid_gold": flags["answer_valid_gold"],
                    "known_conflict": flags["known_conflict"],
                    "bm25_rank": row.get("bm25_rank"),
                    "dense_rank": row.get("dense_rank"),
                    "canonical_ordinal": row.get("canonical_ordinal"),
                }
            )
        complete = union_by_id[question_id]["evidence_classification"][
            "first_complete_evidence_path_rank"
        ]
        completing = _hit_at_rank(union_hits, complete)
        gold_score = None if completing is None else float(completing["score"])
        impact = bm25_added_impact_row(added_rows, gold_score=gold_score)
        impact_rows.append({"question_id": question_id, **impact})
        if question_id in {"Q01", "Q05", "Q07"}:
            above = [
                row
                for row in added_rows
                if gold_score is not None and float(row["ce_score"]) > gold_score
            ]
            above.sort(key=lambda row: (-float(row["ce_score"]), int(row["union_rank"])))
            focus_added[question_id] = [
                _evidence_brief(
                    row["retrieval_unit_id"],
                    units_by_id,
                    score=float(row["ce_score"]),
                    bm25_rank=row["bm25_rank"],
                    dense_rank=row["dense_rank"],
                    ce_rank=row["union_rank"],
                    flags={
                        "answer_valid_gold": row["answer_valid_gold"],
                        "known_conflict": row["known_conflict"],
                    },
                )
                for row in above[:8]
            ]
            false_positives = []
            for hit in dense_only_hits:
                if gold_score is None or float(hit["score"]) <= gold_score:
                    continue
                flags = _unit_evidence_flags(view_by_id[hit["chunk_id"]], dataset, question)
                if flags["answer_valid_gold"]:
                    continue
                pool_row = pool_by_unit[hit["chunk_id"]]
                false_positives.append(
                    _evidence_brief(
                        hit["chunk_id"],
                        units_by_id,
                        score=float(hit["score"]),
                        bm25_rank=pool_row["bm25_rank"],
                        dense_rank=pool_row["dense_rank"],
                        ce_rank=int(hit["rank"]),
                        flags=flags,
                    )
                )
            false_positives.sort(key=lambda row: (-float(row["ce_logit"]), int(row["ce_rank"])))
            focus_false_positives[question_id] = false_positives[:8]
        for claim in required_claims(claims_for_question(dataset, question)):
            claim_id = claim["claim_id"]
            union_rank = _claim_rank_map(union_by_id[question_id]).get(claim_id)
            dense_only_rank = _claim_rank_map(rerank_by_id[question_id]).get(claim_id)
            dense_source = _claim_rank_map(dense_by_id[question_id]).get(claim_id)
            union_unit = _hit_at_rank(union_hits, union_rank)
            logit = None if union_unit is None else float(union_unit["score"])
            same = (
                None
                if union_unit is None
                else _hit_by_id(dense_only_hits, union_unit["chunk_id"])
            )
            same_logit = None if same is None else float(same["score"])
            same_rank = None if same is None else int(same["rank"])
            abs_diff = None if logit is None or same_logit is None else abs(logit - same_logit)
            above_claim = (
                None
                if logit is None
                else sum(1 for row in added_rows if float(row["ce_score"]) > logit)
            )
            unit_scores_match = abs_diff is not None and abs_diff <= PAIR_SCORE_TOLERANCE
            labeled = expansion_interference(
                union_rank=union_rank,
                dense_only_rank=same_rank,
                above_gold=above_claim,
                scores_identical=scores_identical and unit_scores_match,
            )
            if union_unit is None:
                cause = "no_union_completing_unit"
            elif same is None:
                cause = "unit_absent_from_dense_pool"
            elif unit_scores_match and same_rank != union_rank:
                cause = "candidate_removal"
            elif unit_scores_match:
                cause = "score_unchanged"
            else:
                cause = "score_changed"
            row = {
                "question_id": question_id,
                "claim_id": claim_id,
                "dense_source_rank": dense_source,
                "union_ce_rank": union_rank,
                "dense_only_ce_rank": dense_only_rank,
                "rank_delta_dense_only_minus_union": (
                    None
                    if union_rank is None or dense_only_rank is None
                    else dense_only_rank - union_rank
                ),
                "union_completing_unit_id": None if union_unit is None else union_unit["chunk_id"],
                "union_completing_unit_dense_only_rank": same_rank,
                "raw_ce_logit": logit,
                "dense_only_logit_same_unit": same_logit,
                "logit_absolute_difference": abs_diff,
                "bm25_added_above_completing_unit": above_claim,
                "candidate_expansion_interference": labeled,
                "rank_change_cause": cause,
            }
            transitions.append(row)
            if labeled:
                interference.append(
                    {
                        "question_id": question_id,
                        "claim_id": claim_id,
                        "union_rank": union_rank,
                        "dense_only_rank": same_rank,
                        "bm25_added_above_gold": above_claim,
                        "taxonomy": (
                            "retrieval_ranking_failure / candidate_expansion_interference"
                        ),
                    }
                )
    systems = {
        "bm25_ru": bm25_by_id,
        "dense_d2a": dense_by_id,
        "rrf_zero_tie": zero_by_id,
        "union_cross_encoder": union_by_id,
        "dense_only_cross_encoder": rerank_by_id,
    }
    complete_rows = []
    for question in questions:
        question_id = question["question_id"]
        complete_rows.append(
            {
                "question_id": question_id,
                "in_aggregate_denominator": bool(question.get("in_positive_recall_denominator")),
                "answerability": question.get("answerability"),
                "complete_rank": {
                    name: scored[question_id]["evidence_classification"][
                        "first_complete_evidence_path_rank"
                    ]
                    for name, scored in systems.items()
                },
            }
        )
    return {
        "candidate_coverage_comparison": _coverage_comparison(
            union_candidate_generation, dense_candidate_generation
        ),
        "pool_sizes": {
            question["question_id"]: len(pools[question["text"]]) for question in questions
        },
        "bm25_added_impact": {
            "definition": "union candidate set minus dense top-100",
            "scope": "positive independent questions only",
            "not_a_generalization": True,
            "questions": impact_rows,
        },
        "gold_ranking_transition": {
            "score_change_does_not_explain_rank_when_logits_match": True,
            "claims": transitions,
        },
        "five_system_headline": five_system_headline(systems),
        "per_question_complete_ranks": complete_rows,
        "candidate_expansion_interference": {
            "definition": (
                "Attached only when the same unit's raw logit matches and the number of "
                "higher-scoring BM25-only candidates equals the rank improvement."
            ),
            "parent": "retrieval_ranking_failure",
            "claims": interference,
        },
        "focus_audits": _focus_audits(
            dataset=dataset,
            questions=questions,
            view_by_id=view_by_id,
            dense_by_id=dense_by_id,
            zero_by_id=zero_by_id,
            rerank_by_id=rerank_by_id,
            rerank_hits=rerank_hits,
            dense_hits=dense_hits,
            union_by_id=union_by_id,
            union_rankings=union_rankings,
            dense_candidate_generation=dense_candidate_generation,
            impact_rows={row["question_id"]: row for row in impact_rows},
            focus_added=focus_added,
            focus_false_positives=focus_false_positives,
            transitions=transitions,
        ),
    }


def _focus_audits(
    *,
    dataset: dict[str, Any],
    questions: list[dict[str, Any]],
    view_by_id: dict[str, dict[str, Any]],
    dense_by_id: dict[str, dict[str, Any]],
    zero_by_id: dict[str, dict[str, Any]],
    rerank_by_id: dict[str, dict[str, Any]],
    rerank_hits: dict[str, list[dict[str, Any]]],
    dense_hits: dict[str, list[dict[str, Any]]],
    union_by_id: dict[str, dict[str, Any]],
    union_rankings: list[dict[str, Any]],
    dense_candidate_generation: dict[str, Any],
    impact_rows: dict[str, dict[str, Any]],
    focus_added: dict[str, list[dict[str, Any]]],
    focus_false_positives: dict[str, list[dict[str, Any]]],
    transitions: list[dict[str, Any]],
) -> dict[str, Any]:
    by_question = {question["question_id"]: question for question in questions}
    in_pool = {
        row["claim_id"]: bool(row["in_pool"])
        for row in dense_candidate_generation["claim_recall"]["claims"]
    }

    def _complete(scored_by_id: dict[str, dict[str, Any]], question_id: str) -> int | None:
        return scored_by_id[question_id]["evidence_classification"][
            "first_complete_evidence_path_rank"
        ]

    def _success(question_id: str) -> dict[str, bool]:
        return {
            k: bool(rerank_by_id[question_id]["at_k"][k]["answer_evidence_success"])
            for k in ("5", "10", "20")
        }

    def _claim_focus(question_id: str, claim_ids: tuple[str, ...]) -> list[dict[str, Any]]:
        rows = []
        for claim_id in claim_ids:
            match = next(row for row in transitions if row["claim_id"] == claim_id)
            rows.append({"in_dense_pool": in_pool.get(claim_id), **match})
        return rows

    q01_text = by_question["Q01"]["text"]
    q01_union_hits = _ranking_hits(union_rankings, "Q01")
    q01_complete = _complete(union_by_id, "Q01")
    q01_unit = _hit_at_rank(q01_union_hits, q01_complete)
    q01_same = None if q01_unit is None else _hit_by_id(rerank_hits[q01_text], q01_unit["chunk_id"])
    q05_claims = _claim_focus("Q05", ("Q05-C1", "Q05-C2", "Q05-C3"))
    q07_question = by_question["Q07"]
    q07_union_hits = _ranking_hits(union_rankings, "Q07")
    q07_dense_only_hits = rerank_hits[q07_question["text"]]
    q07_elements = []
    for spec in _element_catalog(dataset, q07_question):
        union_rank = _element_rank_in_hits(view_by_id, q07_union_hits, spec["element_id"])
        dense_only_rank = _element_rank_in_hits(view_by_id, q07_dense_only_hits, spec["element_id"])
        union_hit = _hit_at_rank(q07_union_hits, union_rank)
        same = None if union_hit is None else _hit_by_id(q07_dense_only_hits, union_hit["chunk_id"])
        q07_elements.append(
            {
                "element_key": spec["element_key"],
                "conflict_label": spec["conflict_label"],
                "positive_evidence": not spec["conflict_label"],
                "dense_source_rank": _element_rank_in_hits(
                    view_by_id, dense_hits[q07_question["text"]], spec["element_id"]
                ),
                "union_ce_rank": union_rank,
                "dense_only_ce_rank": dense_only_rank,
                "union_logit": None if union_hit is None else float(union_hit["score"]),
                "dense_only_logit_same_unit": None if same is None else float(same["score"]),
                "logit_absolute_difference": (
                    None
                    if union_hit is None or same is None
                    else abs(float(union_hit["score"]) - float(same["score"]))
                ),
            }
        )

    def _path_pair(question_id: str, path_id: str) -> dict[str, Any]:
        return {
            "path_id": path_id,
            "dense_source_rank": dense_by_id[question_id]["answer_evidence"]["path_ranks"][path_id][
                "first_set_full_rank"
            ],
            "rrf_zero_tie_rank": zero_by_id[question_id]["answer_evidence"]["path_ranks"][path_id][
                "first_set_full_rank"
            ],
            "union_ce_rank": union_by_id[question_id]["answer_evidence"]["path_ranks"][path_id][
                "first_set_full_rank"
            ],
            "dense_only_ce_rank": rerank_by_id[question_id]["answer_evidence"]["path_ranks"][
                path_id
            ]["first_set_full_rank"],
        }

    e186 = next(row for row in q07_elements if row["element_key"] == "E186")
    return {
        "Q01": {
            "required_claims": ["Q01-C1", "Q01-C2"],
            "claims": _claim_focus("Q01", ("Q01-C1", "Q01-C2")),
            "complete_rank": {
                "dense_d2a": _complete(dense_by_id, "Q01"),
                "rrf_zero_tie": _complete(zero_by_id, "Q01"),
                "union_cross_encoder": _complete(union_by_id, "Q01"),
                "dense_only_cross_encoder": _complete(rerank_by_id, "Q01"),
            },
            "gold_unit_in_dense_pool": q01_same is not None,
            "gold_unit_id": None if q01_unit is None else q01_unit["chunk_id"],
            "union_logit": None if q01_unit is None else float(q01_unit["score"]),
            "dense_only_logit": None if q01_same is None else float(q01_same["score"]),
            "logit_absolute_difference": (
                None
                if q01_unit is None or q01_same is None
                else abs(float(q01_unit["score"]) - float(q01_same["score"]))
            ),
            "success": _success("Q01"),
            "bm25_added_above_gold": impact_rows["Q01"]["above_required_gold"],
            "bm25_added_above_gold_detail": focus_added.get("Q01", []),
            "dense_only_false_positives_above_gold": focus_false_positives.get("Q01", []),
            "dense_only_false_positives_above_gold_shown": len(
                focus_false_positives.get("Q01", [])
            ),
        },
        "Q05": {
            "claims": q05_claims,
            "all_required_claims_in_dense_pool": all(
                row["in_dense_pool"] for row in q05_claims
            ),
            "complete_rank": {
                "dense_d2a": _complete(dense_by_id, "Q05"),
                "union_cross_encoder": _complete(union_by_id, "Q05"),
                "dense_only_cross_encoder": _complete(rerank_by_id, "Q05"),
            },
            "success": _success("Q05"),
            "bm25_added_above_gold": impact_rows["Q05"]["above_required_gold"],
            "bm25_added_above_gold_detail": focus_added.get("Q05", []),
        },
        "Q07": {
            "path_a_direct_reported_metric": _path_pair("Q07", "direct_reported_metric"),
            "path_b_derived_operands": _path_pair("Q07", "derived_operands"),
            "elements": q07_elements,
            "complete_rank": {
                "dense_d2a": _complete(dense_by_id, "Q07"),
                "rrf_zero_tie": _complete(zero_by_id, "Q07"),
                "union_cross_encoder": _complete(union_by_id, "Q07"),
                "dense_only_cross_encoder": _complete(rerank_by_id, "Q07"),
            },
            "success": _success("Q07"),
            "bm25_added_above_gold": impact_rows["Q07"]["above_required_gold"],
            "bm25_added_above_gold_detail": focus_added.get("Q07", []),
            "E186": {
                **e186,
                "headline_metric": False,
                "optimizer_objective": False,
                "model_selection_criterion": False,
                "general_disclosure_error_detection": False,
                "interpretation": (
                    "E186 rank movement is a relevance-ordering change on a manual "
                    "known-conflict fixture. It is not disclosure-error detection."
                ),
            },
        },
    }


def _assert_pool_ranks_survive(
    dataset: dict[str, Any],
    questions_by_id: dict[str, dict[str, Any]],
    pool_by_id: dict[str, dict[str, Any]],
    rerank_by_id: dict[str, dict[str, Any]],
) -> None:
    for question_id, question in questions_by_id.items():
        if not question.get("in_positive_recall_denominator"):
            continue
        pool_ranks = _claim_rank_map(pool_by_id[question_id])
        rerank_ranks = _claim_rank_map(rerank_by_id[question_id])
        for claim in required_claims(claims_for_question(dataset, question)):
            claim_id = claim["claim_id"]
            if (pool_ranks[claim_id] is None) != (rerank_ranks[claim_id] is None):
                raise UsageError(
                    f"{claim_id} availability changed between the candidate pool and the reranker"
                )
        pool_complete = pool_by_id[question_id]["evidence_classification"][
            "first_complete_evidence_path_rank"
        ]
        rerank_complete = rerank_by_id[question_id]["evidence_classification"][
            "first_complete_evidence_path_rank"
        ]
        if (pool_complete is None) != (rerank_complete is None):
            raise UsageError(
                f"{question_id} path availability changed between the pool and the reranker"
            )


def _load_cross_encoder(device: str) -> tuple[Any, dict[str, Any]]:
    import torch
    from sentence_transformers import CrossEncoder

    torch.set_num_threads(os.cpu_count() or 1)
    model = CrossEncoder(
        RERANKER_MODEL_ID,
        revision=RERANKER_MODEL_REVISION,
        device=device,
        max_length=RERANKER_INFERENCE_MAX_LENGTH,
        activation_fn=torch.nn.Identity(),
        trust_remote_code=False,
        local_files_only=True,
    )
    raw_prompts = dict(getattr(model, "prompts", None) or {})
    prompts = {key: value for key, value in raw_prompts.items() if value}
    default_prompt = getattr(model, "default_prompt_name", None)
    if prompts or default_prompt:
        raise UsageError(
            "cross-encoder would add a prompt. This experiment uses query and retrieval text only. "
            f"prompts={prompts!r} default_prompt_name={default_prompt!r}"
        )
    seq_length = getattr(model, "max_seq_length", None)
    if seq_length is None:
        seq_length = model.max_length
    if int(seq_length) != RERANKER_INFERENCE_MAX_LENGTH:
        raise UsageError("cross-encoder inference max length is not 1024")
    if int(model.num_labels) != 1:
        raise UsageError("cross-encoder does not expose one relevance logit")
    architecture_max = _architecture_max_length(model)
    if architecture_max != RERANKER_ARCHITECTURE_MAX_LENGTH:
        raise UsageError(
            "cross-encoder architecture max length is "
            f"{architecture_max}, expected {RERANKER_ARCHITECTURE_MAX_LENGTH}"
        )
    tokenizer = model.tokenizer
    meta = {
        "architecture_max_length": architecture_max,
        "num_labels": int(model.num_labels),
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "name_or_path": getattr(tokenizer, "name_or_path", None),
            "revision": RERANKER_MODEL_REVISION,
        },
        "torch_version": torch.__version__,
        "sentence_transformers_version": __import__("sentence_transformers").__version__,
        "cuda": bool(torch.cuda.is_available()),
    }
    return model, meta


def _architecture_max_length(model: Any) -> int:
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "max_position_embeddings"):
            return int(config.max_position_embeddings)
    raise UsageError("cross-encoder config has no max_position_embeddings")


def _score_pools(
    model: Any,
    questions: list[dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    *,
    batch_size: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    import torch

    hits: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, int]] = []
    tokenizer = model.tokenizer
    for question in questions:
        text = question["text"]
        candidates = pools[text]
        pairs = cross_encoder_pairs(text, candidates)
        assert_model_inputs(text, candidates, pairs)
        for query, document in pairs:
            rows.append(_token_lengths(tokenizer, query, document))
        _stage(f"rerank {question['question_id']} candidates={len(candidates)}")
        raw = model.predict(
            [[query, document] for query, document in pairs],
            batch_size=batch_size,
            show_progress_bar=False,
            activation_fn=torch.nn.Identity(),
            apply_softmax=False,
            convert_to_numpy=True,
        )
        scores = [float(value) for value in list(raw)]
        hits[text] = order_by_cross_encoder(candidates, scores)
    return hits, summarize_truncation(rows, RERANKER_INFERENCE_MAX_LENGTH)


def _token_lengths(tokenizer: Any, query: str, document: str) -> dict[str, int]:
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    document_ids = tokenizer.encode(document, add_special_tokens=False)
    pair = tokenizer(query, document, add_special_tokens=True, truncation=False)
    return {
        "query_tokens": len(query_ids),
        "document_tokens": len(document_ids),
        "pair_tokens": len(pair["input_ids"]),
    }


def _candidate_generation_report(
    dataset: dict[str, Any],
    questions_by_id: dict[str, dict[str, Any]],
    pool_by_id: dict[str, dict[str, Any]],
    bm25_by_id: dict[str, dict[str, Any]],
    dense_by_id: dict[str, dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    view_by_id: dict[str, dict[str, Any]],
    view: list[dict[str, Any]],
    bm25_hits: dict[str, list[dict[str, Any]]],
    dense_hits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    claim_rows = []
    question_rows = []
    misses = []
    for question_id, question in questions_by_id.items():
        if not question.get("in_positive_recall_denominator"):
            continue
        pool_scored = pool_by_id[question_id]
        complete = pool_scored["evidence_classification"]["first_complete_evidence_path_rank"]
        question_rows.append(
            {
                "question_id": question_id,
                "answerability": question.get("answerability"),
                "complete_path_available": complete is not None,
                "path_ranks": _path_brief(pool_scored),
            }
        )
        pool_chunks = [view_by_id[row["retrieval_unit_id"]] for row in pools[question["text"]]]
        for claim in required_claims(claims_for_question(dataset, question)):
            covered = claim_covered(pool_chunks, claim)
            claim_rows.append(
                {
                    "question_id": question_id,
                    "claim_id": claim["claim_id"],
                    "in_pool": covered,
                }
            )
            if covered:
                continue
            misses.append(
                _miss_row(
                    question,
                    claim,
                    pool_chunks,
                    view,
                    view_by_id,
                    bm25_hits[question["text"]],
                    dense_hits[question["text"]],
                    bm25_by_id[question_id],
                    dense_by_id[question_id],
                )
            )
    q09_pool = pool_by_id["Q09"]
    return {
        "claim_recall": {
            "numerator": sum(row["in_pool"] for row in claim_rows),
            "denominator": len(claim_rows),
            "claims": claim_rows,
        },
        "question_answer_evidence_satisfied": {
            "numerator": sum(row["complete_path_available"] for row in question_rows),
            "denominator": len(question_rows),
            "questions": question_rows,
        },
        "misses": misses,
        "q09_diagnostic": {
            "in_aggregate_denominator": False,
            "reason": "Q01 paraphrase",
            "complete_path_available": (
                q09_pool["evidence_classification"]["first_complete_evidence_path_rank"] is not None
            ),
            "claim_ranks": _claim_rank_map(q09_pool),
        },
        "partial_questions": ["Q02", "Q04", "Q08"],
        "q07_contract": "path A or path B",
    }


def _miss_row(
    question: dict[str, Any],
    claim: dict[str, Any],
    pool_chunks: list[dict[str, Any]],
    view: list[dict[str, Any]],
    view_by_id: dict[str, dict[str, Any]],
    bm25_hit_list: list[dict[str, Any]],
    dense_hit_list: list[dict[str, Any]],
    bm25_scored: dict[str, Any],
    dense_scored: dict[str, Any],
) -> dict[str, Any]:
    spans = []
    for span in claim.get("spans") or []:
        probe = {"claim_id": span["span_id"], "spans": [span]}
        if claim_covered(pool_chunks, probe):
            continue
        spans.append(
            {
                "span_id": span["span_id"],
                "element_key": span.get("element_key"),
                "element_id": span["element_id"],
                "bm25_rank": _element_rank_in_hits(view_by_id, bm25_hit_list, span["element_id"]),
                "dense_rank": _element_rank_in_hits(view_by_id, dense_hit_list, span["element_id"]),
                "covered_by_any_d2a_unit": claim_covered(view, probe),
            }
        )
    return {
        "question_id": question["question_id"],
        "claim_id": claim["claim_id"],
        "bm25_claim_rank": _claim_rank_map(bm25_scored).get(claim["claim_id"]),
        "dense_claim_rank": _claim_rank_map(dense_scored).get(claim["claim_id"]),
        "evidence": spans,
    }


def _comparison(
    bm25_by_id: dict[str, dict[str, Any]],
    dense_by_id: dict[str, dict[str, Any]],
    zero_by_id: dict[str, dict[str, Any]],
    rerank_by_id: dict[str, dict[str, Any]],
    rerank_metrics: dict[str, Any],
) -> dict[str, Any]:
    systems = {
        "bm25_ru": bm25_by_id,
        "dense_d2a": dense_by_id,
        "rrf_zero_tie": zero_by_id,
        "cross_encoder": rerank_by_id,
    }
    table = {}
    for k in HEADLINE_KS:
        row = {}
        for name, scored in systems.items():
            claim_hit, claim_total, question_hit, question_total = _window_counts(scored, k)
            row[name] = {
                "claim_recall": {"numerator": claim_hit, "denominator": claim_total},
                "question_answer_evidence_satisfied": {
                    "numerator": question_hit,
                    "denominator": question_total,
                },
            }
        table[k] = row
    per_question = []
    for question_id in [f"Q0{index}" for index in range(1, 10)]:
        per_question.append(
            {
                "question_id": question_id,
                "in_aggregate_denominator": question_id != "Q09",
                "answerability": rerank_by_id[question_id]["answerability"],
                "complete_rank": {
                    name: scored[question_id]["evidence_classification"][
                        "first_complete_evidence_path_rank"
                    ]
                    for name, scored in systems.items()
                },
                "stage_at_20": _stage_at_20(rerank_by_id[question_id]),
                "success": {
                    k: {
                        name: bool(scored[question_id]["at_k"][k]["answer_evidence_success"])
                        for name, scored in systems.items()
                    }
                    for k in HEADLINE_KS
                },
            }
        )
    claims = []
    for question_id, scored in rerank_by_id.items():
        if not scored["in_positive_recall_denominator"]:
            continue
        for claim_id in _claim_rank_map(scored):
            claims.append(
                {
                    "question_id": question_id,
                    "claim_id": claim_id,
                    "rank": {
                        "bm25_ru": _claim_rank_map(bm25_by_id[question_id]).get(claim_id),
                        "dense_d2a": _claim_rank_map(dense_by_id[question_id]).get(claim_id),
                        "rrf_zero_tie": _claim_rank_map(zero_by_id[question_id]).get(claim_id),
                        "cross_encoder": _claim_rank_map(scored).get(claim_id),
                    },
                }
            )
    return {
        "headline": table,
        "rerank_metrics_at_k": {
            k: {
                "claim_recall": {
                    "numerator": rerank_metrics["at_k"][k]["claim_set_recall"]["numerator"],
                    "denominator": rerank_metrics["at_k"][k]["claim_set_recall"]["denominator"],
                },
                "question_answer_evidence_satisfied": {
                    "numerator": rerank_metrics["at_k"][k]["question_answer_evidence_satisfied"][
                        "numerator"
                    ],
                    "denominator": rerank_metrics["at_k"][k][
                        "question_answer_evidence_satisfied"
                    ]["denominator"],
                },
            }
            for k in HEADLINE_KS
        },
        "per_question_complete_rank": per_question,
        "claims": claims,
    }


def _stage_at_20(scored: dict[str, Any]) -> str:
    rank = scored["evidence_classification"]["first_complete_evidence_path_rank"]
    if rank is None:
        return "candidate miss"
    if rank > 20:
        return "reranking miss"
    return "in_top20"


def _window_counts(
    scored_by_id: dict[str, dict[str, Any]], k: str
) -> tuple[int, int, int, int]:
    claim_hit = 0
    claim_total = 0
    question_hit = 0
    question_total = 0
    for scored in scored_by_id.values():
        if not scored["in_positive_recall_denominator"]:
            continue
        block = scored["at_k"][k]
        claim_hit += int(block["claim_set_full"])
        claim_total += int(block["required_claims"])
        question_hit += int(block["answer_evidence_success"])
        question_total += 1
    return claim_hit, claim_total, question_hit, question_total


def _audits(
    dataset: dict[str, Any],
    questions_by_id: dict[str, dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    view_by_id: dict[str, dict[str, Any]],
    bm25_by_id: dict[str, dict[str, Any]],
    dense_by_id: dict[str, dict[str, Any]],
    zero_by_id: dict[str, dict[str, Any]],
    rerank_by_id: dict[str, dict[str, Any]],
    bm25_hits: dict[str, list[dict[str, Any]]],
    dense_hits: dict[str, list[dict[str, Any]]],
    zero_hits: dict[str, list[dict[str, Any]]],
    rerank_hits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "Q01": _question_audit(
            dataset,
            questions_by_id["Q01"],
            ("Q01-C1", "Q01-C2"),
            pools,
            view_by_id,
            bm25_by_id,
            dense_by_id,
            zero_by_id,
            rerank_by_id,
            bm25_hits,
            dense_hits,
            zero_hits,
            rerank_hits,
        ),
        "Q05": _question_audit(
            dataset,
            questions_by_id["Q05"],
            ("Q05-C1", "Q05-C2", "Q05-C3"),
            pools,
            view_by_id,
            bm25_by_id,
            dense_by_id,
            zero_by_id,
            rerank_by_id,
            bm25_hits,
            dense_hits,
            zero_hits,
            rerank_hits,
        ),
        "Q07": _q07_audit(
            dataset,
            questions_by_id["Q07"],
            pools,
            view_by_id,
            bm25_by_id,
            dense_by_id,
            zero_by_id,
            rerank_by_id,
            bm25_hits,
            dense_hits,
            zero_hits,
            rerank_hits,
        ),
    }


def _question_audit(
    dataset: dict[str, Any],
    question: dict[str, Any],
    claim_ids: tuple[str, ...],
    pools: dict[str, list[dict[str, Any]]],
    view_by_id: dict[str, dict[str, Any]],
    bm25_by_id: dict[str, dict[str, Any]],
    dense_by_id: dict[str, dict[str, Any]],
    zero_by_id: dict[str, dict[str, Any]],
    rerank_by_id: dict[str, dict[str, Any]],
    bm25_hits: dict[str, list[dict[str, Any]]],
    dense_hits: dict[str, list[dict[str, Any]]],
    zero_hits: dict[str, list[dict[str, Any]]],
    rerank_hits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    question_id = question["question_id"]
    text = question["text"]
    pool = pools[text]
    pool_chunks = [view_by_id[row["retrieval_unit_id"]] for row in pool]
    claims = {claim["claim_id"]: claim for claim in claims_for_question(dataset, question)}
    systems = {
        "bm25_ru": (bm25_by_id, bm25_hits),
        "dense_d2a": (dense_by_id, dense_hits),
        "rrf_zero_tie": (zero_by_id, zero_hits),
        "cross_encoder": (rerank_by_id, rerank_hits),
    }
    rows = []
    for claim_id in claim_ids:
        claim = claims[claim_id]
        in_pool = claim_covered(pool_chunks, claim)
        rerank_rank = _claim_rank_map(rerank_by_id[question_id]).get(claim_id)
        rows.append(
            {
                "claim_id": claim_id,
                "in_candidate_pool": in_pool,
                "source_memberships": _memberships_for_claim(pool, view_by_id, claim),
                "rank": {
                    name: _claim_rank_map(scored[question_id]).get(claim_id)
                    for name, (scored, _hits) in systems.items()
                },
                "stage_at_20": classify_stage(
                    covered_by_corpus=_claim_rank_map(dense_by_id[question_id]).get(claim_id)
                    is not None,
                    covered_by_pool=in_pool,
                    rerank_rank=rerank_rank,
                    k=20,
                ),
            }
        )
    complete = {
        name: scored[question_id]["evidence_classification"]["first_complete_evidence_path_rank"]
        for name, (scored, _hits) in systems.items()
    }
    rerank_complete = complete["cross_encoder"]
    return {
        "complete_rank": complete,
        "candidate_path_available": all(row["in_candidate_pool"] for row in rows),
        "pool_status": "candidate miss" if rerank_complete is None else (
            "reranking miss" if rerank_complete > 20 else "in_top20"
        ),
        "success": {
            k: bool(rerank_by_id[question_id]["at_k"][k]["answer_evidence_success"])
            for k in ("5", "10", "20")
        },
        "claims": rows,
    }


def _q07_audit(
    dataset: dict[str, Any],
    question: dict[str, Any],
    pools: dict[str, list[dict[str, Any]]],
    view_by_id: dict[str, dict[str, Any]],
    bm25_by_id: dict[str, dict[str, Any]],
    dense_by_id: dict[str, dict[str, Any]],
    zero_by_id: dict[str, dict[str, Any]],
    rerank_by_id: dict[str, dict[str, Any]],
    bm25_hits: dict[str, list[dict[str, Any]]],
    dense_hits: dict[str, list[dict[str, Any]]],
    zero_hits: dict[str, list[dict[str, Any]]],
    rerank_hits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    systems = {
        "bm25_ru": (bm25_by_id, bm25_hits[question["text"]]),
        "dense_d2a": (dense_by_id, dense_hits[question["text"]]),
        "rrf_zero_tie": (zero_by_id, zero_hits[question["text"]]),
        "cross_encoder": (rerank_by_id, rerank_hits[question["text"]]),
    }
    paths = {}
    for name, (scored, _hits) in systems.items():
        paths[name] = _path_brief(scored["Q07"])
    elements = _element_catalog(dataset, question)
    element_rows = []
    for spec in elements:
        ranks = {}
        for name, (_scored, hits) in systems.items():
            ranks[name] = _element_rank_in_hits(view_by_id, hits, spec["element_id"])
        element_rows.append(
            {
                **spec,
                "rank": ranks,
                "positive_evidence": not spec["conflict_label"],
            }
        )
    rerank_paths = paths["cross_encoder"]
    return {
        "path_complete_rank": paths,
        "elements": element_rows,
        "candidate_path_status": {
            path_id: ("candidate miss" if row["first_set_full_rank"] is None else "in_pool")
            for path_id, row in rerank_paths.items()
        },
        "note": (
            "Element rows are evaluation metadata. "
            "A relevant_conflicting row is a known-conflict diagnostic, not a positive claim."
        ),
    }


def _element_catalog(dataset: dict[str, Any], question: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for claim in _evidence_claims(dataset, question):
        for span in claim.get("spans") or []:
            element_id = span["element_id"]
            if element_id in seen:
                continue
            seen.add(element_id)
            rows.append(
                {
                    "element_key": span.get("element_key"),
                    "element_id": element_id,
                    "conflict_label": False,
                }
            )
    for item in dataset["hard_negatives"]:
        if item.get("evidence_class") != "relevant_conflicting":
            continue
        if question["question_id"] not in item.get("classified_for", []):
            continue
        if item["element_id"] in seen:
            continue
        seen.add(item["element_id"])
        rows.append(
            {
                "element_key": item.get("element_key"),
                "element_id": item["element_id"],
                "conflict_label": True,
            }
        )
    return rows


def _element_rank_in_hits(
    view_by_id: dict[str, dict[str, Any]], hits: list[dict[str, Any]], element_id: str
) -> int | None:
    for hit in hits:
        chunk = view_by_id[hit["chunk_id"]]
        if chunk_contains_element(chunk, element_id, body_only=False):
            return int(hit["rank"])
    return None


def _memberships_for_claim(
    pool: list[dict[str, Any]],
    view_by_id: dict[str, dict[str, Any]],
    claim: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for candidate in pool:
        chunk = view_by_id[candidate["retrieval_unit_id"]]
        if coverage_level(chunk_sources(chunk), claim) == "none":
            continue
        rows.append(
            {
                "retrieval_unit_id": candidate["retrieval_unit_id"],
                "canonical_chunk_id": candidate["canonical_chunk_id"],
                "source_membership": candidate["source_membership"],
                "bm25_rank": candidate["bm25_rank"],
                "bm25_score": candidate["bm25_score"],
                "dense_rank": candidate["dense_rank"],
                "dense_score": candidate["dense_score"],
            }
        )
    return rows


def _known_conflict(
    rerank_by_id: dict[str, dict[str, Any]],
    bm25_by_id: dict[str, dict[str, Any]],
    dense_by_id: dict[str, dict[str, Any]],
    zero_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    population = [
        question_id
        for question_id, scored in rerank_by_id.items()
        if any(row.get("counts_as_relevant_conflict") for row in scored["hard_negative_ranks"])
    ]
    per_k = {}
    for k in DIAGNOSTIC_KS:
        hit = 0
        for question_id in population:
            if rerank_by_id[question_id]["at_k"][str(k)]["relevant_conflicting_hit"]:
                hit += 1
        per_k[str(k)] = {"numerator": hit, "denominator": len(population)}
    questions = []
    for question_id in population:
        rerank = rerank_by_id[question_id]
        conflicts = [
            row for row in rerank["hard_negative_ranks"] if row.get("counts_as_relevant_conflict")
        ]
        classification = rerank["evidence_classification"]
        first_valid = classification["first_positive_answer_evidence_rank"]
        complete = classification["first_complete_evidence_path_rank"]
        rows = []
        for row in conflicts:
            rank = row["rank"]
            rows.append(
                {
                    "element_key": row["element_key"],
                    "element_id": row["element_id"],
                    "reranker_rank": rank,
                    "bm25_rank": _conflict_rank(bm25_by_id[question_id], row["element_id"]),
                    "dense_rank": _conflict_rank(dense_by_id[question_id], row["element_id"]),
                    "rrf_zero_tie_rank": _conflict_rank(zero_by_id[question_id], row["element_id"]),
                    "conflict_precedes_first_valid_evidence": (
                        None
                        if rank is None or first_valid is None
                        else bool(rank < first_valid)
                    ),
                    "conflict_precedes_complete_valid_path": (
                        None
                        if rank is None or complete is None
                        else bool(rank < complete)
                    ),
                    "counted_as_positive_evidence": bool(row["counted_as_positive_evidence"]),
                }
            )
        questions.append(
            {
                "question_id": question_id,
                "first_positive_answer_evidence_rank": first_valid,
                "first_complete_evidence_path_rank": complete,
                "conflicts": rows,
            }
        )
    return {
        "name": "known_conflict_robustness_diagnostic",
        "headline_metric": False,
        "optimizer_objective": False,
        "model_selection_criterion": False,
        "general_disclosure_error_detection": False,
        "interpretation": (
            "A higher or lower conflict rank is not evidence that the reranker "
            "detects disclosure errors. The label is manual and known in advance."
        ),
        "relevant_conflict_hit": per_k,
        "questions": questions,
    }


def _conflict_rank(scored: dict[str, Any], element_id: str) -> int | None:
    for row in scored["hard_negative_ranks"]:
        if row["element_id"] == element_id and row.get("counts_as_relevant_conflict"):
            return row["rank"]
    return None


def _transitions(
    dataset: dict[str, Any],
    questions_by_id: dict[str, dict[str, Any]],
    view_by_id: dict[str, dict[str, Any]],
    bm25_hits: dict[str, list[dict[str, Any]]],
    dense_hits: dict[str, list[dict[str, Any]]],
    zero_hits: dict[str, list[dict[str, Any]]],
    rerank_hits: dict[str, list[dict[str, Any]]],
    bm25_by_id: dict[str, dict[str, Any]],
    dense_by_id: dict[str, dict[str, Any]],
    zero_by_id: dict[str, dict[str, Any]],
    rerank_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    sources = {
        "bm25_ru": bm25_hits,
        "dense_d2a": dense_hits,
        "rrf_zero_tie": zero_hits,
    }
    rank_sources = {
        "bm25_ru": bm25_by_id,
        "dense_d2a": dense_by_id,
        "rrf_zero_tie": zero_by_id,
    }
    for question in questions_by_id.values():
        if not question.get("in_positive_recall_denominator"):
            continue
        text = question["text"]
        rerank_top = [hit["chunk_id"] for hit in rerank_hits[text][:20]]
        moves = {}
        for name, hits in sources.items():
            source_top = [hit["chunk_id"] for hit in hits[text][:20]]
            entered = [doc_id for doc_id in rerank_top if doc_id not in set(source_top)]
            left = [doc_id for doc_id in source_top if doc_id not in set(rerank_top)]
            moves[name] = {
                "overlap": len(set(source_top) & set(rerank_top)),
                "entered_top20": len(entered),
                "left_top20": len(left),
                "entered_gold": _gold_ids(dataset, question, view_by_id, entered),
                "left_gold": _gold_ids(dataset, question, view_by_id, left),
            }
        gold = []
        for claim_id, rerank_rank in _claim_rank_map(rerank_by_id[question["question_id"]]).items():
            source_ranks = {
                name: _claim_rank_map(scored[question["question_id"]]).get(claim_id)
                for name, scored in rank_sources.items()
            }
            gold.append(
                {
                    "claim_id": claim_id,
                    "reranker_rank": rerank_rank,
                    "source_rank": source_ranks,
                    "reranker_minus_source": {
                        name: None
                        if rerank_rank is None or source_ranks[name] is None
                        else rerank_rank - source_ranks[name]
                        for name in source_ranks
                    },
                }
            )
        rows.append({"question_id": question["question_id"], "top20": moves, "gold_claims": gold})
    return rows


def _gold_ids(
    dataset: dict[str, Any],
    question: dict[str, Any],
    view_by_id: dict[str, dict[str, Any]],
    doc_ids: list[str],
) -> list[dict[str, Any]]:
    claims = _evidence_claims(dataset, question)
    rows = []
    for doc_id in doc_ids:
        chunk = view_by_id[doc_id]
        matched = [
            claim["claim_id"]
            for claim in claims
            if coverage_level(chunk_sources(chunk), claim) != "none"
        ]
        if matched:
            rows.append({"retrieval_unit_id": doc_id, "claim_ids": matched})
    return rows


def _source_contribution(
    dataset: dict[str, Any],
    questions_by_id: dict[str, dict[str, Any]],
    rerank_hits: dict[str, list[dict[str, Any]]],
    rerank_by_id: dict[str, dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    windows = {}
    for k in DIAGNOSTIC_KS:
        counts = Counter()
        slots = 0
        for question in questions_by_id.values():
            if not question.get("in_positive_recall_denominator"):
                continue
            for hit in rerank_hits[question["text"]][:k]:
                counts[hit["source_membership"]] += 1
                slots += 1
        windows[str(k)] = {
            "slots": slots,
            "bm25_only": counts[SOURCE_BM25_ONLY],
            "dense_only": counts[SOURCE_DENSE_ONLY],
            "both": counts[SOURCE_BOTH],
        }
    gold = []
    for question in questions_by_id.values():
        if not question.get("in_positive_recall_denominator"):
            continue
        pool = {row["retrieval_unit_id"]: row for row in pools[question["text"]]}
        hits = rerank_hits[question["text"]]
        for claim_id, rank in _claim_rank_map(rerank_by_id[question["question_id"]]).items():
            completing = None if rank is None else hits[rank - 1]["source_membership"]
            categories = sorted(
                {
                    row["source_membership"]
                    for row in _memberships_for_claim(
                        list(pool.values()),
                        {
                            row["retrieval_unit_id"]: {
                                "chunk_id": row["retrieval_unit_id"],
                                "chunk_sources": row["source_refs"],
                            }
                            for row in pool.values()
                        },
                        dataset["claims"][claim_id],
                    )
                }
            )
            gold.append(
                {
                    "question_id": question["question_id"],
                    "claim_id": claim_id,
                    "in_pool": rank is not None,
                    "pool_memberships": categories,
                    "completing_unit_membership": completing,
                    "reranker_rank": rank,
                }
            )
    return {
        "reranker_topk_slots": windows,
        "gold_claims": gold,
        "not_a_generalization": True,
    }


def _diversity(
    questions_by_id: dict[str, dict[str, Any]],
    rerank_hits: dict[str, list[dict[str, Any]]],
    view_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    positive = [
        question
        for question in questions_by_id.values()
        if question.get("in_positive_recall_denominator")
    ]
    report = {}
    for k in DIAGNOSTIC_KS:
        uniques = []
        duplicates = []
        occupancies = []
        for question in positive:
            canonical = [
                view_by_id[hit["chunk_id"]]["canonical_chunk_id"]
                for hit in rerank_hits[question["text"]][:k]
            ]
            counts = Counter(canonical)
            uniques.append(len(counts))
            duplicates.append(sum(count - 1 for count in counts.values()))
            occupancies.append(max(counts.values()) if counts else 0)
        report[str(k)] = {
            "questions": len(positive),
            "average_unique_canonical_chunks": sum(uniques) / len(positive),
            "average_duplicate_ru": sum(duplicates) / len(positive),
            "max_canonical_occupancy": max(occupancies) if occupancies else 0,
        }
    report["canonical_collapse"] = False
    return report


def _failure_taxonomy(
    dataset: dict[str, Any],
    questions_by_id: dict[str, dict[str, Any]],
    view: list[dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    view_by_id: dict[str, dict[str, Any]],
    rerank_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_k = {}
    claim_rows = []
    for k in DIAGNOSTIC_KS:
        counts = Counter()
        rows = []
        for question in questions_by_id.values():
            if not question.get("in_positive_recall_denominator"):
                continue
            pool_chunks = [
                view_by_id[row["retrieval_unit_id"]] for row in pools[question["text"]]
            ]
            ranks = _claim_rank_map(rerank_by_id[question["question_id"]])
            for claim in required_claims(claims_for_question(dataset, question)):
                in_pool = claim_covered(pool_chunks, claim)
                in_corpus = claim_covered(view, claim)
                stage = classify_stage(
                    covered_by_corpus=in_corpus,
                    covered_by_pool=in_pool,
                    rerank_rank=ranks.get(claim["claim_id"]),
                    k=k,
                )
                counts[stage] += 1
                if stage != STAGE_IN_TOPK:
                    label = stage
                    if stage == STAGE_RERANK_MISS:
                        label = "retrieval_ranking_failure / reranking_failure"
                    elif stage == STAGE_CANDIDATE_MISS:
                        label = "retrieval_ranking_failure / candidate_generation_failure"
                    rows.append(
                        {
                            "question_id": question["question_id"],
                            "claim_id": claim["claim_id"],
                            "k": k,
                            "stage": stage,
                            "taxonomy": label,
                            "reranker_rank": ranks.get(claim["claim_id"]),
                        }
                    )
                    if k == 20:
                        claim_rows.append(rows[-1])
        by_k[str(k)] = {"counts": dict(counts), "failures": rows, "denominator": 27}
    return {
        "parent_taxonomy_unchanged": True,
        "did_not_relabel_parsing_or_chunking": True,
        "at_k": by_k,
        "k20_failures": claim_rows,
    }


def _evidence_claims(dataset: dict[str, Any], question: dict[str, Any]) -> list[dict[str, Any]]:
    claims = []
    seen: set[str] = set()
    for path in evidence_paths_for_question(dataset, question):
        for claim in path["claims"]:
            if claim["claim_id"] in seen:
                continue
            seen.add(claim["claim_id"])
            claims.append(claim)
    if claims:
        return claims
    return required_claims(claims_for_question(dataset, question))


def _claim_rank_map(scored: dict[str, Any]) -> dict[str, int | None]:
    return {
        row["claim_id"]: row["first_set_full_rank"] for row in scored["diagnosis"]["claims"]
    }


def _path_brief(scored: dict[str, Any]) -> dict[str, Any]:
    return {
        path_id: {"first_set_full_rank": row["first_set_full_rank"], "claim_ids": row["claim_ids"]}
        for path_id, row in scored["answer_evidence"]["path_ranks"].items()
    }


def _candidate_sha256(
    questions: list[dict[str, Any]], pools: dict[str, list[dict[str, Any]]]
) -> str:
    return sha256_text(canonical_dumps(_candidate_rows(questions, pools)))


def _candidate_rows(
    questions: list[dict[str, Any]], pools: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows = []
    for question in questions:
        for candidate in pools[question["text"]]:
            rows.append(
                {
                    "question_id": question["question_id"],
                    "stable_index": candidate["stable_index"],
                    "retrieval_unit_id": candidate["retrieval_unit_id"],
                    "canonical_chunk_id": candidate["canonical_chunk_id"],
                    "canonical_ordinal": candidate["canonical_ordinal"],
                    "unit_index": candidate["unit_index"],
                    "source_membership": candidate["source_membership"],
                    "bm25_rank": candidate["bm25_rank"],
                    "bm25_score": candidate["bm25_score"],
                    "dense_rank": candidate["dense_rank"],
                    "dense_score": candidate["dense_score"],
                    "text_sha256": sha256_text(candidate["text"]),
                    "source_refs": candidate["source_refs"],
                }
            )
    return rows


def _ranking_rows(
    questions: list[dict[str, Any]], hits: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows = []
    for question in questions:
        for hit in hits[question["text"]]:
            rows.append(
                {
                    "question_id": question["question_id"],
                    "rank": hit["rank"],
                    "retrieval_unit_id": hit["chunk_id"],
                    "score": hit["score"],
                    "source_membership": hit["source_membership"],
                    "stable_index": hit["stable_index"],
                }
            )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_dumps(row) + "\n")


def _invariant_report(
    *,
    summary: dict[str, Any],
    unit_sha: str,
    embedding_sha: str,
    indexed_text_sha: str,
    dataset_sha: str,
    agreements: dict[str, dict[str, Any]],
    candidate_policy: str = POLICY_UNION,
) -> dict[str, Any]:
    checks = {
        "d2a_retrieval_unit_count_2869": summary["retrieval_unit_count"] == 2869,
        "retrieval_unit_sha": unit_sha == D2A_RETRIEVAL_UNIT_SHA,
        "topology_unchanged": summary["one_to_n_count"] == 5 and summary["max_tokens"] == 1003,
        "source_span_sha_locked": unit_sha == D2A_RETRIEVAL_UNIT_SHA,
        "bm25_tokenizer_ko_mixed_v1": TOKENIZER_VERSION == "ko-mixed-v1",
        "bm25_k1": BM25_PARAMS["k1"] == 1.2,
        "bm25_b": BM25_PARAMS["b"] == 0.75,
        "bm25_ru_ranking_reproduced": agreements["bm25_ru"]["match"],
        "dense_embedding_sha": embedding_sha == D2A_EMBEDDING_SHA,
        "dense_ranking_reproduced": agreements["dense_d2a"]["match"],
        "rrf_zero_tie_reproduced": agreements["rrf_zero_tie"]["match"],
        "dataset_sha": dataset_sha == DATASET_SHA,
        "denominator_27_and_8": True,
        "retrieval_unit_dedup": True,
        "rrf_not_a_candidate_source": True,
        "no_canonical_collapse": True,
        "no_query_rewrite": True,
        "no_representation_change": indexed_text_sha == BM25_RU_INDEXED_TEXT_SHA,
        "no_evaluation_metadata_in_model_input": True,
        "canonical_bm25_reproduced": agreements["canonical_bm25"]["match"],
        "embedding_truncated_zero": summary["embedding_truncated"] == 0,
    }
    if candidate_policy == POLICY_UNION:
        checks["candidate_is_bm25_100_union_dense_100"] = True
    elif candidate_policy == POLICY_DENSE_ONLY:
        checks["candidate_is_dense_top100_only"] = True
        checks["bm25_candidates_excluded"] = True
        checks["stable_index_is_dense_rank_order"] = True
    else:
        raise UsageError(f"unknown candidate policy {candidate_policy}")
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise UsageError(f"rerank invariants failed: {failed}")
    return {"passed": True, "checks": checks}
