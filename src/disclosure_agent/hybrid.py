"""BM25 and dense RRF on one fixed D2a retrieval-unit corpus.

The only scoring change is the combination rule. Retrieval-unit text, topology,
gold, and the evaluation denominators stay as they are. Oracle union is a
diagnostic ceiling, not a headline retriever.
"""

from __future__ import annotations

import hashlib
import platform
import time
from pathlib import Path
from typing import Any

from disclosure_agent.bm25 import BM25_PARAMS, TOKENIZER_VERSION, Bm25Index, index_manifest
from disclosure_agent.chunking import (
    CHUNKER_VERSION,
    canonical_dumps,
    chunker_config_hash,
    load_json,
    sha256_text,
)
from disclosure_agent.dense import (
    DENSE_MODEL_ID,
    DENSE_MODEL_REVISION,
    EMBEDDING_DIMENSION,
    encode_texts,
    exact_normalized_search,
    load_dense_model,
    resolve_device,
)
from disclosure_agent.errors import UsageError
from disclosure_agent.evaluation import (
    KS,
    _aggregate,
    _failures,
    _metric_scope,
    _path_window_status,
    _q07_calculation,
    chunk_contains_element,
    chunk_sources,
    claims_for_question,
    coverage_level,
    evidence_labels_for_question,
    evidence_paths_for_question,
    load_eval_corpus,
    reproducibility_record,
    required_claims,
    score_search,
    single_and_set,
    write_eval_run,
)
from disclosure_agent.retrieval_unit import (
    retrieval_units_sha256,
    scoring_chunks,
    slot_diagnostic,
    summarize_retrieval_units,
)

RRF_K = 60
ZERO_SCORE_POLICY = "bm25_zero_score_rrf_contribution = 0"
DIAGNOSTIC_KS = (5, 10, 20)
ORIGINAL_RRF_CONFIG_SHA = "3edf8b34d2d7570549a32fa6aa956773a7ff5e2e31e3a26c501b29bd9afade79"
ORIGINAL_RRF_RANKING_SHA = "9bece202a6bc21383c6131c4870b95baa23b74de0551e3e1fe38002a8d86163c"
BM25_RU_INDEXED_TEXT_SHA = "24bbc8f1e0f607b8ec00f9c6204381c56ed946a768a7f79e85f9b6f08147827e"
D2A_RETRIEVAL_UNIT_SHA = "cff5e1a7d8572c6217a9fea2506581ca9b5ce6e12f60c5f9b2f32aaba4b2c9ff"
D2A_EMBEDDING_SHA = "244a5c82344cfd73c1119934cb35386f44395da9cf54eb127b15f03079efb340"
DATASET_SHA = "1f0ac22c599b94fe72185e9977c7d8d57a188f543c7df4b9f04743a18ef7cfad"
_LOCKED_HEADLINE_NUMERATORS = {
    "bm25_canonical": {
        "1": (1, 0),
        "3": (4, 0),
        "5": (5, 1),
        "10": (14, 2),
        "20": (19, 3),
    },
    "bm25_ru": {
        "1": (1, 0),
        "3": (4, 0),
        "5": (4, 0),
        "10": (14, 2),
        "20": (17, 3),
    },
    "dense_d2a": {
        "1": (6, 0),
        "3": (9, 0),
        "5": (15, 2),
        "10": (20, 3),
        "20": (25, 6),
    },
    "hybrid_rrf": {
        "1": (3, 0),
        "3": (16, 2),
        "5": (19, 2),
        "10": (21, 4),
        "20": (23, 5),
    },
}
QUESTION_IDS = ("Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q07", "Q08", "Q09")
PROTECTED_RUN_NAMES = {
    "isu_petasys_v0_3",
    "isu_petasys_v0_3_dense",
    "isu_petasys_v0_3_dense_d1",
    "isu_petasys_v0_3_dense_d2a",
    "isu_petasys_v0_3_dense_d2b",
    "isu_petasys_v0_3_dense_d2c",
    "isu_petasys_v0_3_bm25_ru_d2a",
    "isu_petasys_v0_3_rrf_d2a",
}
_IDENTITY_FIELDS = (
    "retrieval_unit_id",
    "canonical_chunk_id",
    "canonical_ordinal",
    "unit_index",
    "text",
    "representation_version",
    "representation_hash",
    "split_reason",
)


def reciprocal_rank_fusion(
    bm25_hits: list[dict[str, Any]],
    dense_hits: list[dict[str, Any]],
    *,
    k_rrf: int = RRF_K,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    """Fuse two complete 1-based rankings. Weights stay 1:1. No score mixing."""
    if k_rrf != RRF_K:
        raise UsageError(f"RRF baseline fixes k_rrf at {RRF_K}, got {k_rrf}")
    left = _rank_map(bm25_hits, "bm25")
    right = _rank_map(dense_hits, "dense")
    if set(left) != set(right):
        raise UsageError("RRF requires the same candidate ids in both full rankings")
    if expected_count is not None and len(left) != expected_count:
        raise UsageError(
            f"RRF expected {expected_count} candidates, got {len(left)}; "
            "a top-N cutoff is not part of this baseline"
        )
    meta = {hit["chunk_id"]: hit for hit in bm25_hits}
    scored: list[tuple[float, str, int, int]] = []
    for doc_id, bm25_rank in left.items():
        dense_rank = right[doc_id]
        score = (1.0 / (k_rrf + bm25_rank)) + (1.0 / (k_rrf + dense_rank))
        scored.append((score, doc_id, bm25_rank, dense_rank))
    scored.sort(
        key=lambda item: (
            -item[0],
            meta[item[1]]["ordinal"],
            int(meta[item[1]].get("unit_index") or 0),
            item[1],
        )
    )
    fused = []
    for rank, (score, doc_id, bm25_rank, dense_rank) in enumerate(scored, start=1):
        base = meta[doc_id]
        hit = {
            "rank": rank,
            "chunk_id": doc_id,
            "ordinal": base["ordinal"],
            "score": score,
            "bm25_rank": bm25_rank,
            "dense_rank": dense_rank,
        }
        if base.get("canonical_chunk_id"):
            hit["canonical_chunk_id"] = base["canonical_chunk_id"]
        if base.get("retrieval_unit_id"):
            hit["retrieval_unit_id"] = base["retrieval_unit_id"]
        if "unit_index" in base:
            hit["unit_index"] = int(base["unit_index"])
        fused.append(hit)
    if not fused or fused[0]["rank"] != 1:
        raise UsageError("RRF rank must be 1-based")
    return fused


def reciprocal_rank_fusion_zero_tie(
    bm25_hits: list[dict[str, Any]],
    dense_hits: list[dict[str, Any]],
    *,
    k_rrf: int = RRF_K,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    """RRF at fixed k=60. A BM25 score of 0 contributes 0.

    Positive BM25 documents keep the 1-based rank from the positive-score
    ordering. Zero-score documents stay in the candidate universe, and their
    ordinal or retrieval-unit id does not enter the score. Dense still uses
    its full 1-based ranking. Weights stay 1:1.
    """
    if k_rrf != RRF_K:
        raise UsageError(f"RRF baseline fixes k_rrf at {RRF_K}, got {k_rrf}")
    _validate_zero_tie_bm25(bm25_hits)
    left = _rank_map(bm25_hits, "bm25")
    right = _rank_map(dense_hits, "dense")
    if set(left) != set(right):
        raise UsageError("RRF requires the same candidate ids in both full rankings")
    if expected_count is not None and len(left) != expected_count:
        raise UsageError(
            f"RRF expected {expected_count} candidates, got {len(left)}; "
            "a top-N cutoff is not part of this baseline"
        )
    meta = {hit["chunk_id"]: hit for hit in bm25_hits}
    scored: list[dict[str, Any]] = []
    for doc_id, bm25_rank in left.items():
        dense_rank = right[doc_id]
        bm25_score = float(meta[doc_id]["score"])
        dense_part = 1.0 / (k_rrf + dense_rank)
        sparse = (1.0 / (k_rrf + bm25_rank)) if bm25_score > 0 else 0.0
        scored.append(
            {
                "score": sparse + dense_part,
                "chunk_id": doc_id,
                "bm25_rank": bm25_rank,
                "dense_rank": dense_rank,
                "sparse": sparse,
                "dense_part": dense_part,
                "bm25_score": bm25_score,
                "ordinal": meta[doc_id]["ordinal"],
                "unit_index": int(meta[doc_id].get("unit_index") or 0),
            }
        )
    scored.sort(
        key=lambda item: (
            -item["score"],
            item["ordinal"],
            item["unit_index"],
            item["chunk_id"],
        )
    )
    fused = []
    for rank, item in enumerate(scored, start=1):
        if item["bm25_score"] == 0 and item["sparse"] != 0:
            raise UsageError("zero-score BM25 contribution must be 0")
        base = meta[item["chunk_id"]]
        hit = {
            "rank": rank,
            "chunk_id": item["chunk_id"],
            "ordinal": item["ordinal"],
            "score": item["score"],
            "bm25_rank": item["bm25_rank"],
            "bm25_score": item["bm25_score"],
            "bm25_rrf_contribution": item["sparse"],
            "dense_rank": item["dense_rank"],
            "dense_rrf_contribution": item["dense_part"],
        }
        if base.get("canonical_chunk_id"):
            hit["canonical_chunk_id"] = base["canonical_chunk_id"]
        if base.get("retrieval_unit_id"):
            hit["retrieval_unit_id"] = base["retrieval_unit_id"]
        if "unit_index" in base:
            hit["unit_index"] = int(base["unit_index"])
        fused.append(hit)
    if not fused or fused[0]["rank"] != 1:
        raise UsageError("RRF rank must be 1-based")
    return fused


def _validate_zero_tie_bm25(hits: list[dict[str, Any]]) -> None:
    """Positive scores occupy ranks 1..P. Zero scores are the contiguous tail."""
    ordered = sorted(hits, key=lambda hit: int(hit["rank"]))
    seen_zero = False
    positive_count = 0
    for hit in ordered:
        if "score" not in hit:
            raise UsageError("RRF-zero-tie requires a BM25 score on every candidate")
        score = float(hit["score"])
        rank = int(hit["rank"])
        if score < 0:
            raise UsageError(f"negative BM25 score for {hit['chunk_id']}")
        if score == 0:
            seen_zero = True
            continue
        if seen_zero:
            raise UsageError("positive BM25 score is ranked after a zero score")
        positive_count += 1
        if rank != positive_count:
            raise UsageError("positive BM25 ranks are not the leading 1-based ranks")
    for offset, hit in enumerate(ordered[positive_count:], start=1):
        if float(hit["score"]) != 0:
            raise UsageError("BM25 tail is not entirely zero-score")
        if int(hit["rank"]) != positive_count + offset:
            raise UsageError("zero-score BM25 ranks are not a contiguous tail")


def claim_contribution(
    bm25_questions: list[dict[str, Any]],
    dense_questions: list[dict[str, Any]],
    hybrid_questions: list[dict[str, Any]],
    *,
    ks: tuple[int, ...] = DIAGNOSTIC_KS,
) -> dict[str, Any]:
    """Required-claim hits of the three systems. Not a headline metric."""
    bm25 = {item["question_id"]: item for item in bm25_questions}
    dense = {item["question_id"]: item for item in dense_questions}
    hybrid = {item["question_id"]: item for item in hybrid_questions}
    by_k = {}
    for k in ks:
        rows = []
        question_rows = []
        for question_id, left in bm25.items():
            if not left["in_positive_recall_denominator"]:
                continue
            middle = dense[question_id]
            right = hybrid[question_id]
            key = str(k)
            for claim_id, status in _claim_status(left, key).items():
                dense_hit = _claim_status(middle, key)[claim_id]
                hybrid_hit = _claim_status(right, key)[claim_id]
                rows.append(
                    {
                        "question_id": question_id,
                        "claim_id": claim_id,
                        "bm25_hit": status,
                        "dense_hit": dense_hit,
                        "hybrid_hit": hybrid_hit,
                        "pattern": _hit_pattern(status, dense_hit),
                        "hybrid_preserved_sparse": status and not dense_hit and hybrid_hit,
                        "hybrid_preserved_dense": dense_hit and not status and hybrid_hit,
                        "hybrid_regression": (status or dense_hit) and not hybrid_hit,
                    }
                )
            question_rows.append(
                {
                    "question_id": question_id,
                    "bm25_hit": bool(left["at_k"][key]["answer_evidence_success"]),
                    "dense_hit": bool(middle["at_k"][key]["answer_evidence_success"]),
                    "hybrid_hit": bool(right["at_k"][key]["answer_evidence_success"]),
                }
            )
        by_k[str(k)] = _contribution_summary(rows, question_rows)
    return {
        "role": "diagnostic",
        "not_a_headline_metric": True,
        "population": "required answer-valid claims of positive questions",
        "at_k": by_k,
    }


def oracle_union(
    dataset: dict[str, Any],
    chunks: list[dict[str, Any]],
    bm25_hits: dict[str, list[dict[str, Any]]],
    dense_hits: dict[str, list[dict[str, Any]]],
    *,
    ks: tuple[int, ...] = DIAGNOSTIC_KS,
) -> dict[str, Any]:
    """Coverage of the top-k union. This is not a ranking system."""
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    by_k = {}
    for k in ks:
        claim_hit = 0
        claim_total = 0
        question_hit = 0
        question_total = 0
        for question in dataset["questions"]:
            if not question.get("in_positive_recall_denominator"):
                continue
            question_total += 1
            window = _union_window(
                bm25_hits[question["text"]],
                dense_hits[question["text"]],
                by_id,
                k,
            )
            required = required_claims(claims_for_question(dataset, question))
            claim_total += len(required)
            claim_hit += sum(
                single_and_set(window, claim)["chunk_set"] == "full" for claim in required
            )
            paths = evidence_paths_for_question(dataset, question)
            path_rows = [_path_window_status(window, path) for path in paths]
            question_hit += int(any(row["path_set_satisfied"] for row in path_rows))
        by_k[str(k)] = {
            "claim_recall": {"numerator": claim_hit, "denominator": claim_total},
            "question_answer_evidence_satisfied": {
                "numerator": question_hit,
                "denominator": question_total,
            },
        }
    return {
        "role": "diagnostic",
        "not_a_headline_metric": True,
        "not_a_ranking_system": True,
        "definition": (
            "BM25 top-k source coverage UNION dense top-k source coverage. "
            "Not an implemented retriever and not a headline result."
        ),
        "at_k": by_k,
    }


def _stage(message: str) -> None:
    print(message, flush=True)


def run_d2a_rrf_experiment(
    *,
    dataset_dir: Path,
    manifest_path: Path,
    elements_path: Path,
    d2a_run: Path,
    bm25_canonical_run: Path,
    bm25_output_dir: Path,
    hybrid_output_dir: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Score BM25-RU and RRF. Reuse the locked D2a matrix. Do not rewrite it."""
    _refuse_protected(bm25_output_dir)
    _refuse_protected(hybrid_output_dir)
    if bm25_output_dir.resolve() == hybrid_output_dir.resolve():
        raise UsageError("BM25-RU and hybrid need different output directories")
    if bm25_output_dir.resolve() == d2a_run.resolve():
        raise UsageError("refusing to write the BM25-RU run over the D2a artifact")
    started = time.perf_counter()
    _stage("load corpus")
    corpus = load_eval_corpus(
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        elements_path=elements_path,
    )
    d2a_metrics = load_json(d2a_run / "metrics.json")
    d2a_questions = load_json(d2a_run / "per_question.json")["questions"]
    canonical_metrics = load_json(bm25_canonical_run / "metrics.json")
    canonical_questions = load_json(bm25_canonical_run / "per_question.json")["questions"]
    units = _load_jsonl(d2a_run / "retrieval_units.jsonl")
    before = _identity_rows(units)
    unit_sha = retrieval_units_sha256(units)
    if unit_sha != d2a_metrics["retrieval_units"]["artifact_sha256"]:
        raise UsageError("D2a retrieval-unit file does not match its metrics hash")
    summary = summarize_retrieval_units(
        units,
        target_tokens=512,
        hard_max_tokens=1024,
        model_max_tokens=8192,
    )
    _assert_d2a_shape(summary, d2a_metrics, corpus)
    view = scoring_chunks(units)
    _assert_view_identity(units, view)
    embedding_path = d2a_run / "embeddings.f32"
    matrix, embedding_sha = _load_embedding_matrix(
        embedding_path,
        len(units),
        d2a_metrics["dense"]["embedding_artifact_sha256"],
    )
    index = Bm25Index(view)
    _assert_index_identity(units, index)
    questions = corpus["dataset"]["questions"]
    scoring_corpus = dict(corpus)
    scoring_corpus["chunks"] = view

    _stage("score canonical bm25")
    historical_index = Bm25Index(corpus["chunks"])
    historical_hits = {
        question["text"]: historical_index.search(question["text"], historical_index.n)
        for question in questions
    }
    historical_questions = score_search(
        corpus, lambda text: historical_hits[text]
    )
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
        if len(bm25_full[question["text"]]) != len(units):
            raise UsageError("BM25 full ranking is shorter than the D2a candidate set")
    bm25_questions = score_search(scoring_corpus, lambda text: bm25_hits[text])

    _stage("encode dense queries")
    resolved = resolve_device(device)
    load_started = time.perf_counter()
    model = load_dense_model(resolved)
    model_load_seconds = time.perf_counter() - load_started
    encode_started = time.perf_counter()
    vectors = encode_texts(model, [question["text"] for question in questions], batch_size=8)
    encode_seconds = time.perf_counter() - encode_started
    query_vectors = {
        question["text"]: vectors[index]
        for index, question in enumerate(questions)
    }
    dense_hits = {
        text: exact_normalized_search(matrix, vector, view)
        for text, vector in query_vectors.items()
    }
    for hits in dense_hits.values():
        if len(hits) != len(units):
            raise UsageError("dense search did not rank every D2a retrieval unit")
        if [hit["chunk_id"] for hit in sorted(hits, key=lambda hit: hit["rank"])] != [
            hit["chunk_id"] for hit in hits
        ]:
            raise UsageError("dense ranks are not in rank order")
    dense_questions = score_search(scoring_corpus, lambda text: dense_hits[text])
    dense_agreement = ranking_agreement(d2a_questions, dense_questions)
    if not dense_agreement["match"]:
        raise UsageError(
            "recomputed dense D2a does not match the locked run: "
            f"{dense_agreement['mismatches']}"
        )

    _stage("fuse rrf")
    fused_hits = {
        question["text"]: reciprocal_rank_fusion(
            bm25_full[question["text"]],
            dense_hits[question["text"]],
            k_rrf=RRF_K,
            expected_count=len(units),
        )
        for question in questions
    }
    hybrid_questions = score_search(scoring_corpus, lambda text: fused_hits[text])
    after = _identity_rows(units)
    if before != after:
        raise UsageError("scoring mutated the D2a retrieval-unit topology")

    dense_ids = {hit["chunk_id"] for hit in next(iter(dense_hits.values()))}
    identity = _candidate_identity(
        units=units,
        view=view,
        index=index,
        dense_ids=dense_ids,
        unit_sha=unit_sha,
        embedding_sha=embedding_sha,
        topology_unchanged=True,
    )
    if not identity["same_candidate_universe"] or identity["topology_mutated"]:
        raise UsageError("BM25 and dense are not on the same D2a candidate universe")
    contribution = claim_contribution(bm25_questions, dense_questions, hybrid_questions)
    if contribution["at_k"]["5"]["required_claim_count"] != 27:
        raise UsageError("required claim contribution denominator is not 27")
    oracle = oracle_union(
        corpus["dataset"], view, bm25_hits, dense_hits
    )
    if oracle["at_k"]["5"]["claim_recall"]["denominator"] != 27:
        raise UsageError("oracle claim denominator is not 27")
    if oracle["at_k"]["5"]["question_answer_evidence_satisfied"]["denominator"] != 8:
        raise UsageError("oracle question denominator is not 8")
    split_effect = _oversized_split_effect(
        corpus["dataset"],
        view,
        summary,
        historical_questions,
        historical_hits,
        bm25_questions,
        bm25_hits,
        canonical_index=historical_index,
        ru_index=index,
    )
    comparison = _comparison(
        corpus["dataset"],
        bm25_questions,
        dense_questions,
        hybrid_questions,
        bm25_hits,
        dense_hits,
        fused_hits,
        contribution,
        oracle,
    )
    for question in hybrid_questions:
        if question["question_id"] != "Q07":
            continue
        detail = comparison["q07"]["hybrid_rrf"]
        if detail["E186_counted_as_positive_evidence"]:
            raise UsageError("E186 was counted as positive evidence")
    runtime = {
        "device": resolved,
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "model_load_seconds": model_load_seconds,
        "query_encoding_seconds": encode_seconds,
        "document_reembedded": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    config = _rrf_config()
    config_sha = sha256_text(canonical_dumps(config))
    ranking_sha = _ranking_sha256(questions, fused_hits)
    bm25_payload = _bm25_payload(
        corpus=corpus,
        dataset_dir=dataset_dir,
        index=index,
        summary=summary,
        unit_sha=unit_sha,
        d2a_run=d2a_run,
        questions=bm25_questions,
        canonical_metrics=canonical_metrics,
        split_effect=split_effect,
        identity=identity,
    )
    hybrid_payload = _hybrid_payload(
        corpus=corpus,
        dataset_dir=dataset_dir,
        questions=hybrid_questions,
        config=config,
        config_sha=config_sha,
        ranking_sha=ranking_sha,
        identity=identity,
        dense_agreement=dense_agreement,
        d2a_metrics=d2a_metrics,
        d2a_run=d2a_run,
        embedding_sha=embedding_sha,
        comparison=comparison,
        runtime=runtime,
        summary=summary,
        unit_sha=unit_sha,
    )
    written_bm25 = write_eval_run(
        bm25_output_dir,
        bm25_payload,
        corpus["chunks"],
        metric_keys=_BM25_METRIC_KEYS,
    )
    written_hybrid = write_eval_run(
        hybrid_output_dir,
        hybrid_payload,
        corpus["chunks"],
        metric_keys=_HYBRID_METRIC_KEYS,
    )
    return {
        "bm25_ru": written_bm25,
        "hybrid": written_hybrid,
        "dense_reproduction_match": True,
        "historical_bm25_reproduction_match": True,
    }


def ranking_agreement(
    locked: list[dict[str, Any]], reproduced: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare claim coverage, question success, complete rank, and top-20 ids."""
    reproduced_by_id = {item["question_id"]: item for item in reproduced}
    mismatches = []
    for left in locked:
        right = reproduced_by_id[left["question_id"]]
        for k in ("1", "3", "5", "10", "20"):
            if _claim_status(left, k) != _claim_status(right, k):
                mismatches.append({"question_id": left["question_id"], "k": k, "field": "claims"})
            left_success = bool(left["at_k"][k]["answer_evidence_success"])
            right_success = bool(right["at_k"][k]["answer_evidence_success"])
            if left_success != right_success:
                mismatches.append(
                    {"question_id": left["question_id"], "k": k, "field": "question_success"}
                )
        left_rank = left["evidence_classification"]["first_complete_evidence_path_rank"]
        right_rank = right["evidence_classification"]["first_complete_evidence_path_rank"]
        if left_rank != right_rank:
            mismatches.append(
                {"question_id": left["question_id"], "field": "complete_rank"}
            )
        left_claims = {
            row["claim_id"]: row["first_set_full_rank"] for row in left["diagnosis"]["claims"]
        }
        right_claims = {
            row["claim_id"]: row["first_set_full_rank"] for row in right["diagnosis"]["claims"]
        }
        if left_claims != right_claims:
            mismatches.append(
                {"question_id": left["question_id"], "field": "claim_complete_rank"}
            )
        left_top = [hit["chunk_id"] for hit in left.get("top20") or []]
        right_top = [hit["chunk_id"] for hit in right.get("top20") or []]
        if left_top != right_top:
            mismatches.append({"question_id": left["question_id"], "field": "top20"})
    return {
        "match": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:8],
    }


def _rank_map(hits: list[dict[str, Any]], label: str) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for hit in hits:
        rank = int(hit["rank"])
        if rank < 1:
            raise UsageError(f"{label} RRF rank must be 1-based, got {rank}")
        doc_id = hit["chunk_id"]
        if doc_id in ranks:
            raise UsageError(f"duplicate {label} candidate {doc_id}")
        ranks[doc_id] = rank
    expected = list(range(1, len(ranks) + 1))
    if sorted(ranks.values()) != expected:
        raise UsageError(f"{label} ranks are not a complete 1-based permutation")
    return ranks


def _claim_status(question: dict[str, Any], k: str) -> dict[str, bool]:
    return {
        row["claim_id"]: row["chunk_set"] == "full"
        for row in question["at_k"][k]["claims"]
    }


def _hit_pattern(bm25_hit: bool, dense_hit: bool) -> str:
    if bm25_hit and dense_hit:
        return "both_hit"
    if bm25_hit:
        return "bm25_hit_dense_miss"
    if dense_hit:
        return "bm25_miss_dense_hit"
    return "both_miss"


def _contribution_summary(
    rows: list[dict[str, Any]], question_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    def take(flag: str) -> list[dict[str, str]]:
        return [
            {"question_id": row["question_id"], "claim_id": row["claim_id"]}
            for row in rows
            if row[flag]
        ]

    pattern_counts = {
        pattern: sum(row["pattern"] == pattern for row in rows)
        for pattern in (
            "bm25_hit_dense_miss",
            "bm25_miss_dense_hit",
            "both_hit",
            "both_miss",
        )
    }
    return {
        "required_claim_count": len(rows),
        "positive_question_count": len(question_rows),
        "pattern_counts": pattern_counts,
        "hybrid_hit_count": sum(row["hybrid_hit"] for row in rows),
        "hybrid_miss_count": sum(not row["hybrid_hit"] for row in rows),
        "bm25_only_claims": [
            {"question_id": row["question_id"], "claim_id": row["claim_id"]}
            for row in rows
            if row["pattern"] == "bm25_hit_dense_miss"
        ],
        "dense_only_claims": [
            {"question_id": row["question_id"], "claim_id": row["claim_id"]}
            for row in rows
            if row["pattern"] == "bm25_miss_dense_hit"
        ],
        "both_hit_claims": [
            {"question_id": row["question_id"], "claim_id": row["claim_id"]}
            for row in rows
            if row["pattern"] == "both_hit"
        ],
        "both_miss_claims": [
            {"question_id": row["question_id"], "claim_id": row["claim_id"]}
            for row in rows
            if row["pattern"] == "both_miss"
        ],
        "hybrid_preserved_sparse_claims": take("hybrid_preserved_sparse"),
        "hybrid_preserved_dense_claims": take("hybrid_preserved_dense"),
        "hybrid_regression_claims": take("hybrid_regression"),
        "question_rows": question_rows,
        "question_hybrid_regression": [
            row["question_id"]
            for row in question_rows
            if (row["bm25_hit"] or row["dense_hit"]) and not row["hybrid_hit"]
        ],
    }


def _union_window(
    bm25_hits: list[dict[str, Any]],
    dense_hits: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for hit in [*bm25_hits[:k], *dense_hits[:k]]:
        doc_id = hit["chunk_id"]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rows.append(by_id[doc_id])
    return rows


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(load_json_line(line))
    return rows


def load_json_line(line: str) -> dict[str, Any]:
    import json

    item = json.loads(line)
    if not isinstance(item, dict):
        raise UsageError("retrieval unit JSONL row is not an object")
    return item


def _load_embedding_matrix(
    path: Path, count: int, expected_sha: str
) -> tuple[Any, str]:
    import numpy as np

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha:
        raise UsageError("D2a embedding file does not match its metrics hash")
    expected_bytes = count * EMBEDDING_DIMENSION * 4
    if len(raw) != expected_bytes:
        raise UsageError(
            f"D2a embedding file has {len(raw)} bytes; expected {expected_bytes}"
        )
    matrix = np.frombuffer(raw, dtype=np.float32).reshape(count, EMBEDDING_DIMENSION).copy()
    return matrix, digest


def _identity_rows(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for unit in units:
        row = {field: unit[field] for field in _IDENTITY_FIELDS}
        row["source_refs"] = unit["source_refs"]
        rows.append(row)
    return rows


def _assert_d2a_shape(
    summary: dict[str, Any], metrics: dict[str, Any], corpus: dict[str, Any]
) -> None:
    recorded = metrics["retrieval_units"]["summary"]
    if summary["retrieval_unit_count"] != 2869 or recorded["retrieval_unit_count"] != 2869:
        raise UsageError("D2a retrieval-unit count is not 2869")
    if summary["one_to_n_count"] != 5 or summary["max_tokens"] != 1003:
        raise UsageError("D2a split shape does not match the locked summary")
    if summary["embedding_truncated"] != 0:
        raise UsageError("D2a artifact contains a truncated retrieval unit")
    if corpus["chunk_summary"]["chunk_count"] != 2803:
        raise UsageError("canonical chunk count is not 2803")
    if corpus["chunk_summary"]["artifact_sha256"] != metrics["chunks"]["artifact_sha256"]:
        raise UsageError("canonical chunk artifact hash does not match the D2a run")
    dataset_sha = metrics["dataset_sha256"]
    if dataset_sha != "1f0ac22c599b94fe72185e9977c7d8d57a188f543c7df4b9f04743a18ef7cfad":
        raise UsageError("D2a run is not the pinned isu_petasys_v0_3 dataset")


def _assert_view_identity(units: list[dict[str, Any]], view: list[dict[str, Any]]) -> None:
    if [unit["retrieval_unit_id"] for unit in units] != [row["chunk_id"] for row in view]:
        raise UsageError("scoring view changed retrieval-unit ids or order")
    for unit, row in zip(units, view, strict=True):
        if unit["text"] != row["text"]:
            raise UsageError("scoring view changed retrieval-unit text")
        if unit["source_refs"] != row["chunk_sources"]:
            raise UsageError("scoring view changed source spans")
        if unit["canonical_chunk_id"] != row["canonical_chunk_id"]:
            raise UsageError("scoring view changed canonical chunk id")
        if unit["canonical_ordinal"] != row["ordinal"] or unit["unit_index"] != row["unit_index"]:
            raise UsageError("scoring view changed split topology")


def _assert_index_identity(units: list[dict[str, Any]], index: Bm25Index) -> None:
    if [unit["retrieval_unit_id"] for unit in units] != [
        chunk["chunk_id"] for chunk in index.chunks
    ]:
        raise UsageError("BM25 index candidate ids differ from the D2a retrieval units")
    if index.k1 != 1.2 or index.b != 0.75:
        raise UsageError("BM25-RU changed k1 or b")


def _candidate_identity(
    *,
    units: list[dict[str, Any]],
    view: list[dict[str, Any]],
    index: Bm25Index,
    dense_ids: set[str],
    unit_sha: str,
    embedding_sha: str,
    topology_unchanged: bool,
) -> dict[str, Any]:
    unit_ids = [unit["retrieval_unit_id"] for unit in units]
    bm25_equal = unit_ids == [chunk["chunk_id"] for chunk in index.chunks]
    dense_equal = dense_ids == set(unit_ids) and len(dense_ids) == len(unit_ids)
    view_equal = unit_ids == [row["chunk_id"] for row in view]
    return {
        "retrieval_unit_count": len(units),
        "retrieval_unit_artifact_sha256": unit_sha,
        "dense_embedding_sha256": embedding_sha,
        "bm25_ids_equal_d2a": bm25_equal,
        "dense_ids_equal_d2a": dense_equal,
        "scoring_view_ids_equal_d2a": view_equal,
        "topology_mutated": not topology_unchanged,
        "collapsed_to_canonical_chunk": False,
        "same_candidate_universe": bm25_equal and dense_equal and view_equal and topology_unchanged,
    }


def _oversized_split_effect(
    dataset: dict[str, Any],
    view: list[dict[str, Any]],
    summary: dict[str, Any],
    historical_questions: list[dict[str, Any]],
    historical_hits: dict[str, list[dict[str, Any]]],
    bm25_questions: list[dict[str, Any]],
    bm25_hits: dict[str, list[dict[str, Any]]],
    *,
    canonical_index: Bm25Index,
    ru_index: Bm25Index,
) -> dict[str, Any]:
    split_rows = summary["one_to_n_canonical_chunks"]
    split_ids = {row["canonical_chunk_id"] for row in split_rows}
    children: dict[str, list[str]] = {row["canonical_chunk_id"]: [] for row in split_rows}
    for chunk in view:
        if chunk["canonical_chunk_id"] in children:
            children[chunk["canonical_chunk_id"]].append(chunk["chunk_id"])
    historical = {item["question_id"]: item for item in historical_questions}
    current = {item["question_id"]: item for item in bm25_questions}
    chunk_rows = []
    for row in split_rows:
        per_question = []
        for question in dataset["questions"]:
            if question["question_id"] not in QUESTION_IDS:
                continue
            parent_rank = _rank_of(historical_hits[question["text"]], row["canonical_chunk_id"])
            child_ranks = [
                _rank_of(bm25_hits[question["text"]], child_id)
                for child_id in children[row["canonical_chunk_id"]]
            ]
            present = [rank for rank in child_ranks if rank is not None]
            in_top20 = sum(rank <= 20 for rank in present)
            per_question.append(
                {
                    "question_id": question["question_id"],
                    "canonical_bm25_rank": parent_rank,
                    "bm25_ru_best_child_rank": None if not present else min(present),
                    "bm25_ru_children_in_top20": in_top20,
                }
            )
        chunk_rows.append(
            {
                "canonical_chunk_id": row["canonical_chunk_id"],
                "canonical_ordinal": row["canonical_ordinal"],
                "retrieval_unit_count": row["retrieval_unit_count"],
                "max_tokens": row["max_tokens"],
                "split_reasons": row["split_reasons"],
                "questions": per_question,
            }
        )
    flips = {}
    for k in DIAGNOSTIC_KS:
        gained = []
        lost = []
        key = str(k)
        for question in dataset["questions"]:
            if not question.get("in_positive_recall_denominator"):
                continue
            left = _claim_status(historical[question["question_id"]], key)
            right = _claim_status(current[question["question_id"]], key)
            for claim_id, historical_hit in left.items():
                current_hit = right[claim_id]
                if historical_hit == current_hit:
                    continue
                record = {
                    "question_id": question["question_id"],
                    "claim_id": claim_id,
                    "touches_split_canonical_ids": _touching_split_ids(
                        dataset["claims"][claim_id],
                        [chunk for chunk in view if chunk["canonical_chunk_id"] in split_ids],
                    ),
                }
                if current_hit:
                    gained.append(record)
                else:
                    lost.append(record)
        flips[key] = {"gained_by_bm25_ru": gained, "lost_by_bm25_ru": lost}
    return {
        "role": "diagnostic",
        "not_a_headline_metric": True,
        "canonical_document_count": canonical_index.n,
        "bm25_ru_document_count": ru_index.n,
        "canonical_average_document_length": canonical_index.avgdl,
        "bm25_ru_average_document_length": ru_index.avgdl,
        "canonical_vocabulary_size": len(canonical_index.df),
        "bm25_ru_vocabulary_size": len(ru_index.df),
        "one_to_n_canonical_chunks": chunk_rows,
        "claim_flips_versus_canonical_bm25": flips,
    }


def _touching_split_ids(claim: dict[str, Any], chunks: list[dict[str, Any]]) -> list[str]:
    found = {
        chunk["canonical_chunk_id"]
        for chunk in chunks
        if coverage_level(chunk_sources(chunk), claim) != "none"
    }
    return sorted(found)


def _rank_of(hits: list[dict[str, Any]], chunk_id: str) -> int | None:
    for hit in hits:
        if hit["chunk_id"] == chunk_id:
            return int(hit["rank"])
    return None


def _comparison(
    dataset: dict[str, Any],
    bm25_questions: list[dict[str, Any]],
    dense_questions: list[dict[str, Any]],
    hybrid_questions: list[dict[str, Any]],
    bm25_hits: dict[str, list[dict[str, Any]]],
    dense_hits: dict[str, list[dict[str, Any]]],
    fused_hits: dict[str, list[dict[str, Any]]],
    contribution: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    systems = {
        "bm25_ru": bm25_questions,
        "dense_d2a": dense_questions,
        "hybrid_rrf": hybrid_questions,
    }
    hit_maps = {
        "bm25_ru": bm25_hits,
        "dense_d2a": dense_hits,
        "hybrid_rrf": fused_hits,
    }
    return {
        "per_question_complete_rank": _complete_rank_table(systems),
        "q05": {name: _q05_detail(_question(rows, "Q05")) for name, rows in systems.items()},
        "q07": {name: _q07_detail(_question(rows, "Q07")) for name, rows in systems.items()},
        "claim_contribution": contribution,
        "oracle_union": oracle,
        "diversity": {
            name: _diversity(dataset, rows, hit_maps[name]) for name, rows in systems.items()
        },
        "failure_taxonomy": _failure_taxonomy(hybrid_questions, contribution),
    }


def _complete_rank_table(systems: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for question_id in QUESTION_IDS:
        item: dict[str, Any] = {"question_id": question_id}
        for name, questions in systems.items():
            question = _question(questions, question_id)
            item["answerability"] = question["answerability"]
            item["in_positive_recall_denominator"] = question["in_positive_recall_denominator"]
            item[name] = question["evidence_classification"]["first_complete_evidence_path_rank"]
        rows.append(item)
    return rows


def _q05_detail(question: dict[str, Any]) -> dict[str, Any]:
    ranks = {
        row["claim_id"]: row["first_set_full_rank"] for row in question["diagnosis"]["claims"]
    }
    return {
        "C1": ranks.get("Q05-C1"),
        "C2": ranks.get("Q05-C2"),
        "C3": ranks.get("Q05-C3"),
        "question_complete_rank": question["evidence_classification"][
            "first_complete_evidence_path_rank"
        ],
    }


def _q07_detail(question: dict[str, Any]) -> dict[str, Any]:
    paths = question["answer_evidence"]["path_ranks"]
    e186 = next(
        (row for row in question["hard_negative_ranks"] if row["element_key"] == "E186"),
        None,
    )
    return {
        "E2362_direct_metric_body_rank": _body_rank(question, "E2362"),
        "E2357_connected_scope_body_rank": _body_rank(question, "E2357"),
        "path_a_complete_rank": paths["direct_reported_metric"]["first_set_full_rank"],
        "path_b_complete_rank": paths["derived_operands"]["first_set_full_rank"],
        "E228_raw_operand_body_rank": _body_rank(question, "E228"),
        "E227_body_rank": _body_rank(question, "E227"),
        "E186_relevant_conflicting_rank": None if e186 is None else e186["rank"],
        "E186_counted_as_positive_evidence": (
            False if e186 is None else bool(e186["counted_as_positive_evidence"])
        ),
        "E186_answer_valid_evidence": (
            False if e186 is None else bool(e186["answer_valid_evidence"])
        ),
        "E186_evidence_class": None if e186 is None else e186["evidence_class"],
        "E186_in_positive_claim_recall": False,
    }


def _body_rank(question: dict[str, Any], element_key: str) -> int | None:
    ranks = [
        row["first_body_rank"]
        for row in question["evidence_classification"]["element_body_ranks"]
        if row.get("element_key") == element_key and row["first_body_rank"] is not None
    ]
    return None if not ranks else min(ranks)


def _diversity(
    dataset: dict[str, Any],
    questions: list[dict[str, Any]],
    hits_by_text: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_id = {item["question_id"]: item for item in questions}
    rows = []
    for question in dataset["questions"]:
        if question["question_id"] not in QUESTION_IDS:
            continue
        slots = slot_diagnostic(hits_by_text[question["text"]], ks=DIAGNOSTIC_KS)
        rows.append(
            {
                "question_id": question["question_id"],
                "in_positive_recall_denominator": bool(
                    by_id[question["question_id"]]["in_positive_recall_denominator"]
                ),
                "slots": slots,
            }
        )
    positive = [row for row in rows if row["in_positive_recall_denominator"]]
    summary = {}
    for k in DIAGNOSTIC_KS:
        key = str(k)
        unique = [int(row["slots"][key]["unique_canonical_chunks"]) for row in positive]
        duplicates = [
            int(row["slots"][key]["duplicate_same_source_ru_count"]) for row in positive
        ]
        occupancy = [
            int(row["slots"][key]["max_same_canonical_occupancy"]) for row in positive
        ]
        summary[key] = {
            "questions": len(positive),
            "mean_unique_canonical_chunks": sum(unique) / len(positive),
            "mean_duplicate_same_source_ru_count": sum(duplicates) / len(positive),
            "max_duplicate_same_source_ru_count": max(duplicates),
            "max_same_canonical_occupancy": max(occupancy),
            "role": "diagnostic",
            "not_a_headline_metric": True,
        }
    return {"questions": rows, "positive_summary": summary}


def _failure_taxonomy(
    hybrid_questions: list[dict[str, Any]], contribution: dict[str, Any]
) -> dict[str, Any]:
    causes: dict[str, int] = {}
    for question in hybrid_questions:
        if not question["in_positive_recall_denominator"]:
            continue
        for row in question.get("span_audit") or []:
            causes[row["cause"]] = causes.get(row["cause"], 0) + 1
    return {
        "role": "diagnostic",
        "not_a_headline_metric": True,
        "session_did_not_change": [
            "parsing",
            "chunking",
            "representation",
            "gold",
            "evaluation_denominator",
            "generation",
            "bm25_tokenizer",
            "bm25_k1_b",
            "rrf_k",
            "query_rewrite",
            "reranker",
        ],
        "hybrid_span_audit_causes_at_5": causes,
        "cause_labels": {
            "parsing_loss": "Parsing",
            "chunking_loss": "Chunking",
            "retrieval_ranking_failure": "Retrieval",
            "answerability_limitation": "Answerability",
            "calculation_issue": "Calculation",
            "evaluation_bug": "Evaluation",
        },
        "fusion_claim_regressions": {
            key: value["hybrid_regression_claims"]
            for key, value in contribution["at_k"].items()
        },
        "fusion_question_regressions": {
            key: value["question_hybrid_regression"]
            for key, value in contribution["at_k"].items()
        },
        "conflict": {
            "element_key": "E186",
            "evidence_class": "relevant_conflicting",
            "counts_toward_positive_claim_recall": False,
        },
        "calculation": {
            "independent_of_retrieval": True,
            "direct_reported_metric_delta_pp": "6.65",
            "derived_operands_delta_pp": "6.64",
        },
    }


def _question(questions: list[dict[str, Any]], question_id: str) -> dict[str, Any]:
    return next(item for item in questions if item["question_id"] == question_id)


def _rrf_config() -> dict[str, Any]:
    return {
        "algorithm": "reciprocal_rank_fusion",
        "k_rrf": RRF_K,
        "rank_base": 1,
        "bm25_weight": 1,
        "dense_weight": 1,
        "weighted_score_fusion": False,
        "candidate_cutoff": None,
        "full_ranking_fusion": True,
        "formula": "1/(k_rrf+rank_bm25)+1/(k_rrf+rank_dense)",
        "bm25_headline_omits_zero_scores": True,
        "bm25_fusion_assigns_tail_ranks_to_zero_scores": True,
    }


def _ranking_sha256(
    questions: list[dict[str, Any]], hits_by_text: dict[str, list[dict[str, Any]]]
) -> str:
    lines = []
    for question in questions:
        ids = ",".join(hit["chunk_id"] for hit in hits_by_text[question["text"]])
        lines.append(f"{question['question_id']}:{ids}")
    return sha256_text("\n".join(lines))


def _indexed_text_sha256(chunks: list[dict[str, Any]]) -> str:
    lines = [f"{chunk['chunk_id']} {sha256_text(chunk['text'])}" for chunk in chunks]
    return sha256_text("\n".join(lines))


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {
            "claim_recall": {
                "numerator": metrics["at_k"][key]["claim_set_recall"]["numerator"],
                "denominator": metrics["at_k"][key]["claim_set_recall"]["denominator"],
            },
            "question_answer_evidence_satisfied": {
                "numerator": metrics["at_k"][key]["question_answer_evidence_satisfied"][
                    "numerator"
                ],
                "denominator": metrics["at_k"][key]["question_answer_evidence_satisfied"][
                    "denominator"
                ],
            },
        }
        for key in ("1", "3", "5", "10", "20")
    }


def _bm25_payload(
    *,
    corpus: dict[str, Any],
    dataset_dir: Path,
    index: Bm25Index,
    summary: dict[str, Any],
    unit_sha: str,
    d2a_run: Path,
    questions: list[dict[str, Any]],
    canonical_metrics: dict[str, Any],
    split_effect: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    retrieval = {
        "retriever_type": "bm25",
        "corpus": "d2a_retrieval_unit",
        "tokenizer_version": TOKENIZER_VERSION,
        "k1": BM25_PARAMS["k1"],
        "b": BM25_PARAMS["b"],
        "indexed_text": "d2a_retrieval_unit_text",
        "table_serialization": False,
        "zero_score_omitted_from_topk": True,
        "retrieval_unit_artifact_sha256": unit_sha,
        "indexed_text_sha256": _indexed_text_sha256(index.chunks),
        "source_run": str(d2a_run),
    }
    reproducibility = reproducibility_record(corpus, retrieval)
    metrics = _aggregate(
        corpus["dataset"],
        questions,
        KS,
        index_scope=_metric_scope(corpus["scope"]),
    )
    return {
        "retriever_type": "bm25_ru",
        "dataset": str(dataset_dir),
        "dataset_sha256": reproducibility["evaluation"]["dataset_sha256"],
        "reproducibility": reproducibility,
        "scope": corpus["scope"],
        "chunker_version": CHUNKER_VERSION,
        "chunker_config_hash": chunker_config_hash(),
        "tokenizer_version": TOKENIZER_VERSION,
        "bm25": dict(BM25_PARAMS),
        "chunks": corpus["chunk_summary"],
        "index": index_manifest(index, scope=corpus["scope"]),
        "retrieval_units": {
            "mode": "d2a",
            "artifact_sha256": unit_sha,
            "summary": summary,
            "candidate_identity": identity,
            "indexed_text_sha256": retrieval["indexed_text_sha256"],
        },
        "bm25_canonical_locked": {
            "run": "isu_petasys_v0_3",
            "corpus": "canonical_chunk",
            "not_this_run": True,
            "headline": _headline(canonical_metrics["metrics"]),
        },
        "oversized_split_effect": split_effect,
        "calculation_q07": _q07_calculation(corpus["dataset"]),
        "metrics": metrics,
        "questions": questions,
        "failures": _failures(questions, 5),
    }


def _hybrid_payload(
    *,
    corpus: dict[str, Any],
    dataset_dir: Path,
    questions: list[dict[str, Any]],
    config: dict[str, Any],
    config_sha: str,
    ranking_sha: str,
    identity: dict[str, Any],
    dense_agreement: dict[str, Any],
    d2a_metrics: dict[str, Any],
    d2a_run: Path,
    embedding_sha: str,
    comparison: dict[str, Any],
    runtime: dict[str, Any],
    summary: dict[str, Any],
    unit_sha: str,
) -> dict[str, Any]:
    retrieval = {
        "retriever_type": "rrf",
        "k_rrf": RRF_K,
        "rank_base": 1,
        "full_ranking_fusion": True,
        "candidate_cutoff": None,
        "config_sha256": config_sha,
        "ranking_sha256": ranking_sha,
        "dense_embedding_sha256": embedding_sha,
        "dense_model": DENSE_MODEL_ID,
        "dense_model_revision": DENSE_MODEL_REVISION,
        "document_reembedded": False,
        "retrieval_unit_artifact_sha256": unit_sha,
    }
    reproducibility = reproducibility_record(corpus, retrieval)
    metrics = _aggregate(
        corpus["dataset"],
        questions,
        KS,
        index_scope=_metric_scope(corpus["scope"]),
    )
    if "oracle_union" in metrics["headline_metrics"]:
        raise UsageError("oracle union leaked into headline metrics")
    return {
        "retriever_type": "rrf_hybrid",
        "dataset": str(dataset_dir),
        "dataset_sha256": reproducibility["evaluation"]["dataset_sha256"],
        "reproducibility": reproducibility,
        "scope": corpus["scope"],
        "chunker_version": CHUNKER_VERSION,
        "chunker_config_hash": chunker_config_hash(),
        "chunks": corpus["chunk_summary"],
        "rrf": {**config, "config_sha256": config_sha, "ranking_sha256": ranking_sha},
        "runtime": runtime,
        "calculation_q07": _q07_calculation(corpus["dataset"]),
        "candidate_identity": identity,
        "dense_reproduction": {
            "source_run": str(d2a_run),
            "embedding_sha256": embedding_sha,
            "retrieval_unit_artifact_sha256": unit_sha,
            "document_reembedded": False,
            "match": dense_agreement["match"],
            "mismatch_count": dense_agreement["mismatch_count"],
            "locked_headline": _headline(d2a_metrics["metrics"]),
            "model": DENSE_MODEL_ID,
            "model_revision": DENSE_MODEL_REVISION,
        },
        "retrieval_units": {
            "mode": "d2a",
            "artifact_sha256": unit_sha,
            "retrieval_unit_count": summary["retrieval_unit_count"],
            "one_to_n_count": summary["one_to_n_count"],
            "max_tokens": summary["max_tokens"],
            "embedding_truncated": summary["embedding_truncated"],
        },
        "metrics": metrics,
        "comparison": comparison,
        "questions": questions,
        "failures": _failures(questions, 5),
    }


_BM25_METRIC_KEYS = (
    "retriever_type",
    "dataset",
    "dataset_sha256",
    "reproducibility",
    "scope",
    "chunker_version",
    "chunker_config_hash",
    "tokenizer_version",
    "bm25",
    "chunks",
    "index",
    "retrieval_units",
    "bm25_canonical_locked",
    "oversized_split_effect",
    "calculation_q07",
    "metrics",
    "failures",
)

_HYBRID_METRIC_KEYS = (
    "retriever_type",
    "dataset",
    "dataset_sha256",
    "reproducibility",
    "scope",
    "chunker_version",
    "chunker_config_hash",
    "chunks",
    "rrf",
    "runtime",
    "calculation_q07",
    "candidate_identity",
    "dense_reproduction",
    "retrieval_units",
    "metrics",
    "comparison",
    "failures",
)


def _refuse_protected(path: Path) -> None:
    if path.name in PROTECTED_RUN_NAMES:
        raise UsageError(f"refusing to write into protected run {path.name}")


def run_d2a_rrf_zero_tie_experiment(
    *,
    dataset_dir: Path,
    manifest_path: Path,
    elements_path: Path,
    d2a_run: Path,
    bm25_canonical_run: Path,
    bm25_ru_run: Path,
    original_rrf_run: Path,
    output_dir: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Recompute locked rankings and write only the zero-score RRF variant.

    BM25 score 0 contributes nothing. Positive BM25 ranks, dense ranks, k=60,
    and the 1:1 weights stay as they are. Locked run directories are not written.
    """
    _refuse_protected(output_dir)
    locked_runs = (d2a_run, bm25_canonical_run, bm25_ru_run, original_rrf_run)
    if any(output_dir.resolve() == path.resolve() for path in locked_runs):
        raise UsageError("refusing to write RRF-zero-tie over a locked retrieval run")
    started = time.perf_counter()
    _stage("load corpus")
    corpus = load_eval_corpus(
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        elements_path=elements_path,
    )
    d2a_metrics = load_json(d2a_run / "metrics.json")
    d2a_questions = load_json(d2a_run / "per_question.json")["questions"]
    canonical_metrics = load_json(bm25_canonical_run / "metrics.json")
    canonical_questions = load_json(bm25_canonical_run / "per_question.json")["questions"]
    bm25_metrics = load_json(bm25_ru_run / "metrics.json")
    bm25_locked_questions = load_json(bm25_ru_run / "per_question.json")["questions"]
    original_metrics = load_json(original_rrf_run / "metrics.json")
    original_locked_questions = load_json(original_rrf_run / "per_question.json")["questions"]
    _assert_locked_headline(canonical_metrics, "bm25_canonical")
    _assert_locked_headline(bm25_metrics, "bm25_ru")
    _assert_locked_headline(d2a_metrics, "dense_d2a")
    _assert_locked_headline(original_metrics, "hybrid_rrf")
    if original_metrics["rrf"]["config_sha256"] != ORIGINAL_RRF_CONFIG_SHA:
        raise UsageError("locked original RRF config SHA does not match the pinned value")
    if original_metrics["rrf"]["ranking_sha256"] != ORIGINAL_RRF_RANKING_SHA:
        raise UsageError("locked original RRF ranking SHA does not match the pinned value")
    if original_metrics["rrf"].get("bm25_fusion_assigns_tail_ranks_to_zero_scores") is not True:
        raise UsageError("locked original RRF is not the tail-rank zero-score fusion")
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
    embedding_path = d2a_run / "embeddings.f32"
    matrix, embedding_sha = _load_embedding_matrix(
        embedding_path,
        len(units),
        D2A_EMBEDDING_SHA,
    )
    if embedding_sha != d2a_metrics["dense"]["embedding_artifact_sha256"]:
        raise UsageError("D2a embedding SHA does not match the locked dense run")
    index = Bm25Index(view)
    _assert_index_identity(units, index)
    if TOKENIZER_VERSION != "ko-mixed-v1":
        raise UsageError("BM25 tokenizer is not ko-mixed-v1")
    indexed_text_sha = _indexed_text_sha256(index.chunks)
    locked_indexed = bm25_metrics["retrieval_units"]["indexed_text_sha256"]
    if indexed_text_sha != BM25_RU_INDEXED_TEXT_SHA or indexed_text_sha != locked_indexed:
        raise UsageError("BM25-RU indexed text SHA does not match the locked run")
    questions = corpus["dataset"]["questions"]
    scoring_corpus = dict(corpus)
    scoring_corpus["chunks"] = view

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
        full = bm25_full[question["text"]]
        if {hit["chunk_id"] for hit in full} != set(unit_ids):
            raise UsageError("BM25 full ranking candidate ids differ from the D2a retrieval units")
        if len(full) != len(units):
            raise UsageError("BM25 full ranking is shorter than the D2a candidate set")
        positive = [hit for hit in full if float(hit["score"]) > 0]
        search_ids = [hit["chunk_id"] for hit in bm25_hits[question["text"]]]
        if [hit["chunk_id"] for hit in positive] != search_ids:
            raise UsageError("positive BM25 rank_all order differs from BM25-RU search")
        if [hit["rank"] for hit in positive] != list(range(1, len(positive) + 1)):
            raise UsageError("positive BM25 ranks are not 1-based")
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
    model = load_dense_model(resolved)
    model_load_seconds = time.perf_counter() - load_started
    encode_started = time.perf_counter()
    vectors = encode_texts(model, [question["text"] for question in questions], batch_size=8)
    encode_seconds = time.perf_counter() - encode_started
    query_vectors = {
        question["text"]: vectors[position] for position, question in enumerate(questions)
    }
    dense_hits = {
        text: exact_normalized_search(matrix, vector, view)
        for text, vector in query_vectors.items()
    }
    for hits in dense_hits.values():
        if len(hits) != len(units):
            raise UsageError("dense search did not rank every D2a retrieval unit")
        if {hit["chunk_id"] for hit in hits} != set(unit_ids):
            raise UsageError("dense candidate ids differ from the D2a retrieval units")
    dense_questions = score_search(scoring_corpus, lambda text: dense_hits[text])
    dense_agreement = ranking_agreement(d2a_questions, dense_questions)
    if not dense_agreement["match"]:
        raise UsageError(
            "recomputed dense D2a does not match the locked run: "
            f"{dense_agreement['mismatches']}"
        )

    _stage("fuse original rrf")
    original_config = _rrf_config()
    original_config_sha = sha256_text(canonical_dumps(original_config))
    if original_config_sha != ORIGINAL_RRF_CONFIG_SHA:
        raise UsageError("recomputed original RRF config SHA does not match the locked config")
    original_fused = {
        question["text"]: reciprocal_rank_fusion(
            bm25_full[question["text"]],
            dense_hits[question["text"]],
            k_rrf=RRF_K,
            expected_count=len(units),
        )
        for question in questions
    }
    original_ranking_sha = _ranking_sha256(questions, original_fused)
    if original_ranking_sha != ORIGINAL_RRF_RANKING_SHA:
        raise UsageError("recomputed original RRF ranking SHA does not match the locked ranking")
    original_questions = score_search(scoring_corpus, lambda text: original_fused[text])
    original_agreement = ranking_agreement(original_locked_questions, original_questions)
    if not original_agreement["match"]:
        raise UsageError(
            "recomputed original RRF does not match the locked run: "
            f"{original_agreement['mismatches']}"
        )

    _stage("fuse rrf zero-tie")
    zero_tie_fused = {
        question["text"]: reciprocal_rank_fusion_zero_tie(
            bm25_full[question["text"]],
            dense_hits[question["text"]],
            k_rrf=RRF_K,
            expected_count=len(units),
        )
        for question in questions
    }
    _assert_zero_tie_score_contract(questions, bm25_full, original_fused, zero_tie_fused)
    zero_tie_questions = score_search(scoring_corpus, lambda text: zero_tie_fused[text])
    after = _identity_rows(units)
    if before != after:
        raise UsageError("scoring mutated the D2a retrieval-unit topology")

    dense_ids = {hit["chunk_id"] for hit in next(iter(dense_hits.values()))}
    identity = _candidate_identity(
        units=units,
        view=view,
        index=index,
        dense_ids=dense_ids,
        unit_sha=unit_sha,
        embedding_sha=embedding_sha,
        topology_unchanged=True,
    )
    if not identity["same_candidate_universe"] or identity["collapsed_to_canonical_chunk"]:
        raise UsageError("RRF-zero-tie is not on the unchanged D2a candidate universe")
    if identity["retrieval_unit_count"] != 2869:
        raise UsageError("RRF-zero-tie candidate universe is not 2869")
    contribution = claim_contribution(bm25_questions, dense_questions, zero_tie_questions)
    original_contribution = claim_contribution(
        bm25_questions, dense_questions, original_questions
    )
    if contribution["at_k"]["5"]["required_claim_count"] != 27:
        raise UsageError("required claim contribution denominator is not 27")
    claim_delta = fusion_variant_claim_delta(original_questions, zero_tie_questions)
    outside = {
        "original_rrf": _outside_source_topk(bm25_questions, dense_questions, original_questions),
        "zero_tie": _outside_source_topk(bm25_questions, dense_questions, zero_tie_questions),
    }
    source_union = oracle_union(corpus["dataset"], view, bm25_hits, dense_hits)
    if source_union["at_k"]["5"]["claim_recall"]["denominator"] != 27:
        raise UsageError("source top-k union claim denominator is not 27")
    if source_union["at_k"]["5"]["question_answer_evidence_satisfied"]["denominator"] != 8:
        raise UsageError("source top-k union question denominator is not 8")
    impact = zero_score_rank_impact(
        corpus["dataset"],
        view,
        bm25_full,
        original_fused,
        zero_tie_fused,
    )
    systems = {
        "bm25_ru": bm25_questions,
        "dense_d2a": dense_questions,
        "hybrid_rrf_original": original_questions,
        "hybrid_rrf_zero_tie": zero_tie_questions,
    }
    hit_maps = {
        "bm25_ru": bm25_hits,
        "dense_d2a": dense_hits,
        "hybrid_rrf_original": original_fused,
        "hybrid_rrf_zero_tie": zero_tie_fused,
    }
    comparison = {
        "per_question_complete_rank": _complete_rank_table(systems),
        "per_question_claim_ranks": _claim_rank_table(systems),
        "q01": {name: _focus_detail(_question(rows, "Q01")) for name, rows in systems.items()},
        "q05": {name: _q05_detail(_question(rows, "Q05")) for name, rows in systems.items()},
        "q07": {
            name: {
                **_q07_detail(_question(rows, "Q07")),
                "question_success": _question_success(_question(rows, "Q07")),
            }
            for name, rows in systems.items()
        },
        "claim_contribution_zero_tie": contribution,
        "claim_contribution_original": original_contribution,
        "original_versus_zero_tie_claims": claim_delta,
        "outside_source_topk": outside,
        "source_top_k_union_coverage": _source_union_report(
            source_union,
            _aggregate(
                corpus["dataset"],
                original_questions,
                KS,
                index_scope=_metric_scope(corpus["scope"]),
            ),
            _aggregate(
                corpus["dataset"],
                zero_tie_questions,
                KS,
                index_scope=_metric_scope(corpus["scope"]),
            ),
        ),
        "zero_score_impact": impact,
        "diversity": {
            name: _diversity(corpus["dataset"], rows, hit_maps[name])
            for name, rows in systems.items()
        },
        "failure_taxonomy": _zero_tie_failure_taxonomy(original_questions, zero_tie_questions),
    }
    for question in zero_tie_questions:
        if question["question_id"] != "Q07":
            continue
        detail = comparison["q07"]["hybrid_rrf_zero_tie"]
        if detail["E186_counted_as_positive_evidence"]:
            raise UsageError("E186 was counted as positive evidence")
    runtime = {
        "device": resolved,
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "model_load_seconds": model_load_seconds,
        "query_encoding_seconds": encode_seconds,
        "document_reembedded": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    config = _rrf_zero_tie_config()
    config_sha = sha256_text(canonical_dumps(config))
    ranking_sha = _ranking_sha256(questions, zero_tie_fused)
    zero_metrics = _aggregate(
        corpus["dataset"],
        zero_tie_questions,
        KS,
        index_scope=_metric_scope(corpus["scope"]),
    )
    if "oracle_union" in zero_metrics["headline_metrics"]:
        raise UsageError("source union leaked into headline metrics")
    summary_report = _zero_tie_summary(
        canonical_metrics=canonical_metrics,
        bm25_metrics=bm25_metrics,
        d2a_metrics=d2a_metrics,
        original_metrics=original_metrics,
        zero_metrics=zero_metrics,
        comparison=comparison,
    )
    retrieval = {
        "retriever_type": "rrf_zero_tie",
        "k_rrf": RRF_K,
        "rank_base": 1,
        "bm25_weight": 1,
        "dense_weight": 1,
        "full_ranking_fusion": True,
        "candidate_cutoff": None,
        "weighted_score_fusion": False,
        "reranker": False,
        "query_rewrite": False,
        "canonical_collapse": False,
        "bm25_zero_score_rrf_contribution": 0,
        "zero_score_policy": ZERO_SCORE_POLICY,
        "config_sha256": config_sha,
        "ranking_sha256": ranking_sha,
        "original_rrf_config_sha256": original_config_sha,
        "original_rrf_ranking_sha256": original_ranking_sha,
        "dense_embedding_sha256": embedding_sha,
        "dense_model": DENSE_MODEL_ID,
        "dense_model_revision": DENSE_MODEL_REVISION,
        "document_reembedded": False,
        "retrieval_unit_artifact_sha256": unit_sha,
        "bm25_indexed_text_sha256": indexed_text_sha,
        "tokenizer_version": TOKENIZER_VERSION,
        "bm25_k1": index.k1,
        "bm25_b": index.b,
    }
    reproducibility = reproducibility_record(corpus, retrieval)
    if reproducibility["evaluation"]["dataset_sha256"] != DATASET_SHA:
        raise UsageError("dataset SHA does not match the pinned isu_petasys_v0_3 snapshot")
    payload = {
        "retriever_type": "rrf_zero_tie",
        "dataset": str(dataset_dir),
        "dataset_sha256": reproducibility["evaluation"]["dataset_sha256"],
        "reproducibility": reproducibility,
        "scope": corpus["scope"],
        "chunker_version": CHUNKER_VERSION,
        "chunker_config_hash": chunker_config_hash(),
        "chunks": corpus["chunk_summary"],
        "rrf": {**config, "config_sha256": config_sha, "ranking_sha256": ranking_sha},
        "runtime": runtime,
        "calculation_q07": _q07_calculation(corpus["dataset"]),
        "candidate_identity": identity,
        "dense_reproduction": {
            "source_run": str(d2a_run),
            "embedding_sha256": embedding_sha,
            "retrieval_unit_artifact_sha256": unit_sha,
            "document_reembedded": False,
            "match": dense_agreement["match"],
            "mismatch_count": dense_agreement["mismatch_count"],
            "locked_headline": _headline(d2a_metrics["metrics"]),
            "model": DENSE_MODEL_ID,
            "model_revision": DENSE_MODEL_REVISION,
        },
        "bm25_ru_reproduction": {
            "source_run": str(bm25_ru_run),
            "match": bm25_agreement["match"],
            "mismatch_count": bm25_agreement["mismatch_count"],
            "indexed_text_sha256": indexed_text_sha,
            "tokenizer_version": TOKENIZER_VERSION,
            "k1": index.k1,
            "b": index.b,
            "zero_score_omitted_from_topk": True,
        },
        "original_rrf_reproduction": {
            "source_run": str(original_rrf_run),
            "match": original_agreement["match"],
            "mismatch_count": original_agreement["mismatch_count"],
            "config_sha256": original_config_sha,
            "ranking_sha256": original_ranking_sha,
            "historical_bm25_match": historical_agreement["match"],
        },
        "retrieval_units": {
            "mode": "d2a",
            "artifact_sha256": unit_sha,
            "retrieval_unit_count": summary["retrieval_unit_count"],
            "one_to_n_count": summary["one_to_n_count"],
            "max_tokens": summary["max_tokens"],
            "embedding_truncated": summary["embedding_truncated"],
        },
        "invariants": _zero_tie_invariants(
            identity=identity,
            index=index,
            indexed_text_sha=indexed_text_sha,
            embedding_sha=embedding_sha,
            unit_sha=unit_sha,
            config=config,
            dense_agreement=dense_agreement,
            bm25_agreement=bm25_agreement,
            original_agreement=original_agreement,
            historical_agreement=historical_agreement,
            impact=impact,
        ),
        "summary": summary_report,
        "metrics": zero_metrics,
        "comparison": comparison,
        "questions": zero_tie_questions,
        "failures": _failures(zero_tie_questions, 5),
    }
    written = write_eval_run(
        output_dir,
        payload,
        corpus["chunks"],
        metric_keys=_ZERO_TIE_METRIC_KEYS,
    )
    return {"hybrid_rrf_zero_tie": written}


def _assert_locked_headline(metrics: dict[str, Any], label: str) -> None:
    expected = _LOCKED_HEADLINE_NUMERATORS[label]
    headline = _headline(metrics["metrics"])
    for k, (claim_numerator, question_numerator) in expected.items():
        claim = headline[k]["claim_recall"]
        question = headline[k]["question_answer_evidence_satisfied"]
        if (claim["numerator"], claim["denominator"]) != (claim_numerator, 27):
            raise UsageError(
                f"locked {label} claim@{k} is "
                f"{claim['numerator']}/{claim['denominator']}, expected {claim_numerator}/27"
            )
        if (question["numerator"], question["denominator"]) != (question_numerator, 8):
            raise UsageError(
                f"locked {label} question@{k} is "
                f"{question['numerator']}/{question['denominator']}, "
                f"expected {question_numerator}/8"
            )


def _assert_zero_tie_score_contract(
    questions: list[dict[str, Any]],
    bm25_full: dict[str, list[dict[str, Any]]],
    original_fused: dict[str, list[dict[str, Any]]],
    zero_tie_fused: dict[str, list[dict[str, Any]]],
) -> None:
    for question in questions:
        text = question["text"]
        scores = {hit["chunk_id"]: float(hit["score"]) for hit in bm25_full[text]}
        original = {hit["chunk_id"]: hit for hit in original_fused[text]}
        zero_tie = {hit["chunk_id"]: hit for hit in zero_tie_fused[text]}
        if set(original) != set(zero_tie) or len(zero_tie) != 2869:
            raise UsageError("RRF-zero-tie candidate universe changed")
        worsened = 0
        for doc_id, hit in zero_tie.items():
            bm25_score = scores[doc_id]
            if bm25_score == 0:
                if hit["bm25_rrf_contribution"] != 0:
                    raise UsageError("zero-score BM25 RRF contribution is not 0")
                if hit["score"] != hit["dense_rrf_contribution"]:
                    raise UsageError("zero-score RRF score is not the dense contribution alone")
            else:
                if hit["bm25_rank"] != original[doc_id]["bm25_rank"]:
                    raise UsageError("positive BM25 rank differs from original RRF")
                if hit["score"] != original[doc_id]["score"]:
                    raise UsageError("positive BM25 RRF score differs from original RRF")
                if hit["rank"] > original[doc_id]["rank"]:
                    worsened += 1
        if worsened:
            raise UsageError("a positive-BM25 document worsened under RRF-zero-tie")


def fusion_variant_claim_delta(
    original_questions: list[dict[str, Any]],
    zero_tie_questions: list[dict[str, Any]],
    *,
    ks: tuple[int, ...] = DIAGNOSTIC_KS,
) -> dict[str, Any]:
    """Whether the same required claims are hit. Equal numerators can still differ."""
    left = {item["question_id"]: item for item in original_questions}
    right = {item["question_id"]: item for item in zero_tie_questions}
    by_k = {}
    for k in ks:
        key = str(k)
        same: list[dict[str, str]] = []
        gained: list[dict[str, str]] = []
        lost: list[dict[str, str]] = []
        both_miss: list[dict[str, str]] = []
        for question_id, original in left.items():
            if not original["in_positive_recall_denominator"]:
                continue
            current = right[question_id]
            for claim_id, original_hit in _claim_status(original, key).items():
                zero_hit = _claim_status(current, key)[claim_id]
                row = {"question_id": question_id, "claim_id": claim_id}
                if original_hit and zero_hit:
                    same.append(row)
                elif zero_hit:
                    gained.append(row)
                elif original_hit:
                    lost.append(row)
                else:
                    both_miss.append(row)
        by_k[key] = {
            "required_claim_count": len(same) + len(gained) + len(lost) + len(both_miss),
            "original_hit_count": len(same) + len(lost),
            "zero_tie_hit_count": len(same) + len(gained),
            "same_hit_count": len(same),
            "same_claim_set": not gained and not lost,
            "same_hit_claims": same,
            "gained_by_zero_tie": gained,
            "lost_by_zero_tie": lost,
            "both_miss_count": len(both_miss),
        }
    return {
        "role": "diagnostic",
        "not_a_headline_metric": True,
        "definition": "Required-claim hit set of original RRF versus RRF-zero-tie",
        "at_k": by_k,
    }


def _outside_source_topk(
    bm25_questions: list[dict[str, Any]],
    dense_questions: list[dict[str, Any]],
    hybrid_questions: list[dict[str, Any]],
    *,
    ks: tuple[int, ...] = DIAGNOSTIC_KS,
) -> dict[str, Any]:
    bm25 = {item["question_id"]: item for item in bm25_questions}
    dense = {item["question_id"]: item for item in dense_questions}
    hybrid = {item["question_id"]: item for item in hybrid_questions}
    by_k = {}
    for k in ks:
        key = str(k)
        raised = []
        lost = []
        for question_id, left in bm25.items():
            if not left["in_positive_recall_denominator"]:
                continue
            middle = dense[question_id]
            right = hybrid[question_id]
            for claim_id, bm25_hit in _claim_status(left, key).items():
                dense_hit = _claim_status(middle, key)[claim_id]
                hybrid_hit = _claim_status(right, key)[claim_id]
                row = {"question_id": question_id, "claim_id": claim_id}
                if not bm25_hit and not dense_hit and hybrid_hit:
                    raised.append(row)
                if (bm25_hit or dense_hit) and not hybrid_hit:
                    lost.append(row)
        by_k[key] = {
            "outside_source_topk_raised_by_hybrid": raised,
            "source_topk_union_lost_by_hybrid": lost,
        }
    return {"role": "diagnostic", "not_a_headline_metric": True, "at_k": by_k}


def _source_union_report(
    source_union: dict[str, Any],
    original_metrics: dict[str, Any],
    zero_metrics: dict[str, Any],
) -> dict[str, Any]:
    original = _headline(original_metrics)
    zero_tie = _headline(zero_metrics)
    by_k = {}
    for k in DIAGNOSTIC_KS:
        key = str(k)
        by_k[key] = {
            "source_top_k_union": source_union["at_k"][key],
            "original_rrf": original[key],
            "zero_tie_rrf": zero_tie[key],
        }
    return {
        "name": "Source Top-K Union Coverage",
        "role": "diagnostic",
        "not_a_headline_metric": True,
        "not_a_ranking_system": True,
        "not_an_upper_bound": True,
        "definition": (
            "BM25-RU top-k source coverage UNION Dense-D2a top-k source coverage. "
            "A fusion ranking can place a candidate above k even when both source "
            "ranks are worse than k, so this is not an oracle upper bound."
        ),
        "at_k": by_k,
    }


def zero_score_rank_impact(
    dataset: dict[str, Any],
    view: list[dict[str, Any]],
    bm25_full: dict[str, list[dict[str, Any]]],
    original_fused: dict[str, list[dict[str, Any]]],
    zero_tie_fused: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Direct effect of dropping zero-score sparse contributions."""
    by_id = {chunk["chunk_id"]: chunk for chunk in view}
    questions = []
    for question in dataset["questions"]:
        text = question["text"]
        full = bm25_full[text]
        scores = {hit["chunk_id"]: float(hit["score"]) for hit in full}
        original = {hit["chunk_id"]: hit for hit in original_fused[text]}
        zero_tie = {hit["chunk_id"]: hit for hit in zero_tie_fused[text]}
        positive_ids = [hit["chunk_id"] for hit in full if float(hit["score"]) > 0]
        zero_ids = [hit["chunk_id"] for hit in full if float(hit["score"]) == 0]
        original_top = [hit["chunk_id"] for hit in original_fused[text][:20]]
        zero_top = [hit["chunk_id"] for hit in zero_tie_fused[text][:20]]
        entered = [doc_id for doc_id in zero_top if doc_id not in set(original_top)]
        left = [doc_id for doc_id in original_top if doc_id not in set(zero_top)]
        rank_changed = 0
        positive_improved = 0
        positive_unchanged = 0
        zero_improved = 0
        zero_worsened = 0
        zero_unchanged = 0
        for doc_id in scores:
            before = int(original[doc_id]["rank"])
            after = int(zero_tie[doc_id]["rank"])
            if before != after:
                rank_changed += 1
            if scores[doc_id] > 0:
                if after < before:
                    positive_improved += 1
                elif after == before:
                    positive_unchanged += 1
            elif after < before:
                zero_improved += 1
            elif after > before:
                zero_worsened += 1
            else:
                zero_unchanged += 1
        moved = []
        for doc_id, movement in (
            *((doc_id, "entered_top20") for doc_id in entered),
            *((doc_id, "left_top20") for doc_id in left),
        ):
            chunk = by_id[doc_id]
            gold_ids, conflict_keys = _ru_gold_and_conflict(dataset, question, chunk)
            moved.append(
                {
                    "movement": movement,
                    "retrieval_unit_id": doc_id,
                    "canonical_chunk_id": chunk.get("canonical_chunk_id"),
                    "canonical_ordinal": chunk.get("ordinal"),
                    "bm25_score": scores[doc_id],
                    "bm25_original_fusion_rank": int(original[doc_id]["bm25_rank"]),
                    "dense_rank": int(original[doc_id]["dense_rank"]),
                    "original_rrf_rank": int(original[doc_id]["rank"]),
                    "zero_tie_rrf_rank": int(zero_tie[doc_id]["rank"]),
                    "gold_relevant": bool(gold_ids),
                    "gold_claim_ids": gold_ids,
                    "conflicting_evidence": bool(conflict_keys),
                    "conflicting_element_keys": conflict_keys,
                }
            )
        questions.append(
            {
                "question_id": question["question_id"],
                "in_positive_recall_denominator": bool(
                    question.get("in_positive_recall_denominator")
                ),
                "bm25_positive_score_count": len(positive_ids),
                "bm25_zero_score_count": len(zero_ids),
                "original_rrf_top20_zero_score_count": sum(
                    scores[doc_id] == 0 for doc_id in original_top
                ),
                "zero_tie_rrf_top20_zero_score_count": sum(
                    scores[doc_id] == 0 for doc_id in zero_top
                ),
                "rank_changed_count": rank_changed,
                "positive_bm25_rank_improved_count": positive_improved,
                "positive_bm25_rank_unchanged_count": positive_unchanged,
                "zero_score_rank_improved_count": zero_improved,
                "zero_score_rank_worsened_count": zero_worsened,
                "zero_score_rank_unchanged_count": zero_unchanged,
                "top20_entered": entered,
                "top20_left": left,
                "top20_membership_changed": bool(entered or left),
                "moved_retrieval_units": moved,
            }
        )
    return {
        "role": "diagnostic",
        "not_a_headline_metric": True,
        "definition": (
            "Original RRF gives every zero-score retrieval unit a distinct tail rank. "
            "RRF-zero-tie sets that sparse contribution to 0."
        ),
        "questions": questions,
        "questions_with_top20_membership_change": [
            row["question_id"] for row in questions if row["top20_membership_changed"]
        ],
        "any_rank_changed": any(row["rank_changed_count"] for row in questions),
    }


def _ru_gold_and_conflict(
    dataset: dict[str, Any], question: dict[str, Any], chunk: dict[str, Any]
) -> tuple[list[str], list[str]]:
    claims = required_claims(claims_for_question(dataset, question))
    gold = [
        claim["claim_id"]
        for claim in claims
        if coverage_level(chunk_sources(chunk), claim) != "none"
    ]
    conflicts = []
    for item in evidence_labels_for_question(dataset, question):
        if item.get("evidence_class") != "relevant_conflicting":
            continue
        if chunk_contains_element(chunk, item["element_id"], body_only=True):
            conflicts.append(item.get("element_key") or item["element_id"])
    return gold, conflicts


def _focus_detail(question: dict[str, Any]) -> dict[str, Any]:
    ranks = {
        row["claim_id"]: row["first_set_full_rank"] for row in question["diagnosis"]["claims"]
    }
    return {
        "claim_ranks": ranks,
        "C1": ranks.get("Q01-C1"),
        "C2": ranks.get("Q01-C2"),
        "question_complete_rank": question["evidence_classification"][
            "first_complete_evidence_path_rank"
        ],
        "question_success": _question_success(question),
        "in_positive_recall_denominator": question["in_positive_recall_denominator"],
    }


def _question_success(question: dict[str, Any]) -> dict[str, bool]:
    return {
        k: bool(question["at_k"][k]["answer_evidence_success"])
        for k in ("1", "3", "5", "10", "20")
    }


def _claim_rank_table(systems: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    names = list(systems)
    base = systems[names[0]]
    for question in base:
        if question["question_id"] not in QUESTION_IDS:
            continue
        ranks = {
            row["claim_id"]: row["first_set_full_rank"] for row in question["diagnosis"]["claims"]
        }
        for claim_id in ranks:
            item: dict[str, Any] = {
                "question_id": question["question_id"],
                "claim_id": claim_id,
                "in_positive_recall_denominator": question["in_positive_recall_denominator"],
            }
            for name in names:
                match = _question(systems[name], question["question_id"])
                found = {
                    row["claim_id"]: row["first_set_full_rank"]
                    for row in match["diagnosis"]["claims"]
                }
                item[name] = found.get(claim_id)
            rows.append(item)
    return rows


def _rrf_zero_tie_config() -> dict[str, Any]:
    return {
        "algorithm": "reciprocal_rank_fusion_zero_tie",
        "experiment": "rrf_zero_tie",
        "k_rrf": RRF_K,
        "rank_base": 1,
        "bm25_weight": 1,
        "dense_weight": 1,
        "weighted_score_fusion": False,
        "candidate_cutoff": None,
        "full_ranking_fusion": True,
        "reranker": False,
        "query_rewrite": False,
        "canonical_collapse": False,
        "formula": "(1/(k_rrf+rank_bm25) if bm25_score > 0 else 0) + 1/(k_rrf+rank_dense)",
        "bm25_zero_score_rrf_contribution": 0,
        "zero_score_policy": ZERO_SCORE_POLICY,
        "bm25_headline_omits_zero_scores": True,
        "bm25_fusion_assigns_tail_ranks_to_zero_scores": False,
        "positive_bm25_ranks_match_original_rrf": True,
    }


def _zero_tie_failure_taxonomy(
    original_questions: list[dict[str, Any]],
    zero_tie_questions: list[dict[str, Any]],
) -> dict[str, Any]:
    def causes(questions: list[dict[str, Any]]) -> dict[str, int]:
        found: dict[str, int] = {}
        for question in questions:
            if not question["in_positive_recall_denominator"]:
                continue
            for row in question.get("span_audit") or []:
                found[row["cause"]] = found.get(row["cause"], 0) + 1
        return found

    original_causes = causes(original_questions)
    zero_causes = causes(zero_tie_questions)
    return {
        "role": "diagnostic",
        "not_a_headline_metric": True,
        "taxonomy_policy": (
            "This run validates fusion semantics. A rank change is not reclassified "
            "as parsing, chunking, or representation."
        ),
        "session_did_not_change": [
            "parsing",
            "chunking",
            "representation",
            "gold",
            "evaluation_denominator",
            "generation",
            "bm25_tokenizer",
            "bm25_k1_b",
            "rrf_k",
            "query_rewrite",
            "reranker",
            "canonical_collapse",
        ],
        "original_span_audit_causes_at_5": original_causes,
        "zero_tie_span_audit_causes_at_5": zero_causes,
        "cause_labels": {
            "parsing_loss": "Parsing",
            "chunking_loss": "Chunking",
            "retrieval_ranking_failure": "Retrieval",
            "answerability_limitation": "Answerability",
            "calculation_issue": "Calculation",
            "evaluation_bug": "Evaluation",
        },
        "conflict": {
            "element_key": "E186",
            "evidence_class": "relevant_conflicting",
            "counts_toward_positive_claim_recall": False,
        },
    }


def _zero_tie_invariants(
    *,
    identity: dict[str, Any],
    index: Bm25Index,
    indexed_text_sha: str,
    embedding_sha: str,
    unit_sha: str,
    config: dict[str, Any],
    dense_agreement: dict[str, Any],
    bm25_agreement: dict[str, Any],
    original_agreement: dict[str, Any],
    historical_agreement: dict[str, Any],
    impact: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "d2a_retrieval_unit_count_2869": identity["retrieval_unit_count"] == 2869,
        "candidate_ids_equal_d2a": identity["same_candidate_universe"],
        "topology_unchanged": not identity["topology_mutated"],
        "bm25_indexed_text_unchanged": indexed_text_sha == BM25_RU_INDEXED_TEXT_SHA,
        "bm25_tokenizer_ko_mixed_v1": TOKENIZER_VERSION == "ko-mixed-v1",
        "bm25_k1_1_2": index.k1 == 1.2,
        "bm25_b_0_75": index.b == 0.75,
        "dense_embedding_sha_unchanged": embedding_sha == D2A_EMBEDDING_SHA,
        "dense_ranking_matches_locked_d2a": dense_agreement["match"],
        "positive_bm25_ranking_matches_locked_bm25_ru": bm25_agreement["match"],
        "original_rrf_ranking_matches_locked": original_agreement["match"],
        "historical_bm25_matches_locked": historical_agreement["match"],
        "rrf_k_60": config["k_rrf"] == 60,
        "weights_1_to_1": config["bm25_weight"] == 1 and config["dense_weight"] == 1,
        "zero_score_sparse_contribution_0": config["bm25_zero_score_rrf_contribution"] == 0,
        "retrieval_unit_sha_unchanged": unit_sha == D2A_RETRIEVAL_UNIT_SHA,
        "no_canonical_collapse": config["canonical_collapse"] is False,
        "no_candidate_cutoff": config["candidate_cutoff"] is None,
        "no_reranker": config["reranker"] is False,
        "no_query_rewrite": config["query_rewrite"] is False,
        "no_weighted_score_fusion": config["weighted_score_fusion"] is False,
        "positive_bm25_ranks_not_worsened": all(
            row["positive_bm25_rank_improved_count"] + row["positive_bm25_rank_unchanged_count"]
            == row["bm25_positive_score_count"]
            for row in impact["questions"]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise UsageError(f"RRF-zero-tie invariant failed: {failed}")
    return {"all_passed": True, "checks": checks}


def _ratio(block: dict[str, int]) -> str:
    return f"{block['numerator']}/{block['denominator']}"


def _headline_cell(metrics: dict[str, Any], k: str) -> dict[str, str]:
    block = metrics["at_k"][k]
    return {
        "claim": _ratio(block["claim_set_recall"]),
        "question": _ratio(block["question_answer_evidence_satisfied"]),
    }


def _zero_tie_summary(
    *,
    canonical_metrics: dict[str, Any],
    bm25_metrics: dict[str, Any],
    d2a_metrics: dict[str, Any],
    original_metrics: dict[str, Any],
    zero_metrics: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    systems = {
        "bm25_canonical": canonical_metrics["metrics"],
        "bm25_ru": bm25_metrics["metrics"],
        "dense_d2a": d2a_metrics["metrics"],
        "hybrid_rrf_original": original_metrics["metrics"],
        "hybrid_rrf_zero_tie": zero_metrics,
    }
    headline = {}
    for k in ("1", "3", "5", "10", "20"):
        headline[k] = {name: _headline_cell(metrics, k) for name, metrics in systems.items()}
    diversity = {}
    for name, block in comparison["diversity"].items():
        diversity[name] = {
            k: {
                "mean_unique_canonical_chunks": row["mean_unique_canonical_chunks"],
                "mean_duplicate_same_source_ru_count": row["mean_duplicate_same_source_ru_count"],
                "max_same_canonical_occupancy": row["max_same_canonical_occupancy"],
            }
            for k, row in block["positive_summary"].items()
        }
    impact_brief = [
        {
            "question_id": row["question_id"],
            "bm25_positive_score_count": row["bm25_positive_score_count"],
            "bm25_zero_score_count": row["bm25_zero_score_count"],
            "original_rrf_top20_zero_score_count": row["original_rrf_top20_zero_score_count"],
            "zero_tie_rrf_top20_zero_score_count": row["zero_tie_rrf_top20_zero_score_count"],
            "rank_changed_count": row["rank_changed_count"],
            "top20_membership_changed": row["top20_membership_changed"],
            "top20_entered_count": len(row["top20_entered"]),
            "top20_left_count": len(row["top20_left"]),
        }
        for row in comparison["zero_score_impact"]["questions"]
    ]
    return {
        "experiment": "rrf_zero_tie",
        "zero_score_policy": ZERO_SCORE_POLICY,
        "headline": headline,
        "complete_rank": comparison["per_question_complete_rank"],
        "q01": comparison["q01"],
        "q05": comparison["q05"],
        "q07": comparison["q07"],
        "original_versus_zero_tie_claims": {
            k: {
                "same_claim_set": row["same_claim_set"],
                "original_hit_count": row["original_hit_count"],
                "zero_tie_hit_count": row["zero_tie_hit_count"],
                "same_hit_count": row["same_hit_count"],
                "gained_by_zero_tie": row["gained_by_zero_tie"],
                "lost_by_zero_tie": row["lost_by_zero_tie"],
            }
            for k, row in comparison["original_versus_zero_tie_claims"]["at_k"].items()
        },
        "source_top_k_union_coverage": comparison["source_top_k_union_coverage"]["at_k"],
        "zero_score_impact": impact_brief,
        "diversity_positive": diversity,
    }


_ZERO_TIE_METRIC_KEYS = (
    "retriever_type",
    "dataset",
    "dataset_sha256",
    "reproducibility",
    "scope",
    "chunker_version",
    "chunker_config_hash",
    "chunks",
    "rrf",
    "runtime",
    "calculation_q07",
    "candidate_identity",
    "dense_reproduction",
    "bm25_ru_reproduction",
    "original_rrf_reproduction",
    "retrieval_units",
    "invariants",
    "summary",
    "metrics",
    "comparison",
    "failures",
)
