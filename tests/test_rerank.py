"""Candidate union and pure cross-encoder ordering. No model download."""

from __future__ import annotations

import pytest

from disclosure_agent.errors import UsageError
from disclosure_agent.rerank import (
    STAGE_CANDIDATE_MISS,
    STAGE_IN_TOPK,
    STAGE_NOT_IN_CORPUS,
    STAGE_RERANK_MISS,
    assert_model_inputs,
    bm25_added_impact_row,
    build_candidate_pool,
    build_dense_only_candidate_pool,
    claim_covered,
    classify_stage,
    common_candidate_text_identity,
    cross_encoder_pairs,
    expansion_interference,
    order_by_cross_encoder,
    pair_score_identity,
    summarize_truncation,
)


def _hit(chunk_id: str, rank: int, score: float) -> dict:
    return {"chunk_id": chunk_id, "rank": rank, "score": score, "ordinal": rank}


def _unit(unit_id: str, text: str, canonical: str = "canon") -> dict:
    return {
        "retrieval_unit_id": unit_id,
        "canonical_chunk_id": canonical,
        "canonical_ordinal": 1,
        "unit_index": 0,
        "text": text,
        "source_refs": [{"element_id": f"element-{unit_id}", "role": "body"}],
    }


def _paragraph(chunk_id: str, element_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "chunk_sources": [
            {
                "element_id": element_id,
                "kind": "paragraph",
                "role": "body",
                "char_start": 0,
                "char_end": 4,
            }
        ],
    }


def _claim(claim_id: str, element_id: str) -> dict:
    return {
        "claim_id": claim_id,
        "spans": [
            {
                "span_id": claim_id,
                "element_id": element_id,
                "kind": "paragraph",
                "char_start": 0,
                "char_end": 4,
            }
        ],
    }


def test_candidate_union_dedups_retrieval_unit_id() -> None:
    units = {
        "a": _unit("a", "alpha"),
        "b": _unit("b", "beta"),
        "c": _unit("c", "gamma"),
    }
    bm25 = [_hit("a", 1, 2.0), _hit("b", 2, 1.0)]
    dense = [_hit("b", 1, 0.9), _hit("c", 2, 0.2), _hit("a", 3, 0.1)]
    pool = build_candidate_pool(bm25, dense, units, top_n=2)
    assert [row["retrieval_unit_id"] for row in pool] == ["a", "b", "c"]
    assert len({row["retrieval_unit_id"] for row in pool}) == 3
    by_id = {row["retrieval_unit_id"]: row for row in pool}
    assert by_id["b"]["source_membership"] == "both"
    assert by_id["b"]["bm25_rank"] == 2
    assert by_id["b"]["dense_rank"] == 1
    assert by_id["b"]["stable_index"] == 1


def test_bm25_only_candidate_is_preserved() -> None:
    units = {"a": _unit("a", "alpha"), "b": _unit("b", "beta")}
    pool = build_candidate_pool(
        [_hit("a", 1, 3.0), _hit("b", 40, 0.4)],
        [_hit("b", 1, 0.8), _hit("a", 150, 0.01)],
        units,
        top_n=1,
    )
    by_id = {row["retrieval_unit_id"]: row for row in pool}
    assert set(by_id) == {"a", "b"}
    assert by_id["a"]["source_membership"] == "bm25_only"
    assert by_id["a"]["bm25_rank"] == 1
    assert by_id["a"]["bm25_score"] == 3.0
    assert by_id["a"]["dense_rank"] == 150
    assert by_id["a"]["in_dense_top_n"] is False
    assert by_id["b"]["source_membership"] == "dense_only"


def test_dense_only_candidate_is_preserved() -> None:
    units = {"a": _unit("a", "alpha"), "d": _unit("d", "delta")}
    pool = build_candidate_pool(
        [_hit("a", 1, 1.0)],
        [_hit("d", 1, 0.7), _hit("a", 2, 0.2)],
        units,
        top_n=1,
    )
    assert [row["retrieval_unit_id"] for row in pool] == ["a", "d"]
    dense_only = pool[1]
    assert dense_only["source_membership"] == "dense_only"
    assert dense_only["bm25_rank"] is None
    assert dense_only["dense_rank"] == 1
    assert dense_only["dense_score"] == 0.7


def test_same_canonical_chunk_is_not_collapsed() -> None:
    units = {
        "u1": _unit("u1", "left", canonical="same"),
        "u2": _unit("u2", "right", canonical="same"),
    }
    pool = build_candidate_pool(
        [_hit("u1", 1, 1.0)],
        [_hit("u2", 1, 0.5)],
        units,
        top_n=1,
    )
    assert [row["retrieval_unit_id"] for row in pool] == ["u1", "u2"]
    assert {row["canonical_chunk_id"] for row in pool} == {"same"}


def test_bm25_zero_score_is_rejected() -> None:
    units = {"a": _unit("a", "alpha")}
    with pytest.raises(UsageError):
        build_candidate_pool(
            [_hit("a", 1, 0.0)],
            [_hit("a", 1, 0.2)],
            units,
            top_n=1,
        )


def test_evaluation_metadata_is_not_in_the_model_input() -> None:
    candidate = {
        "text": "본문만 전달한다",
        "retrieval_unit_id": "u1",
        "claim_id": "Q01-C1",
        "question_id": "Q01",
        "bm25_rank": 41,
        "dense_rank": 18,
        "evidence_class": "relevant_conflicting",
        "element_key": "E186",
        "note": "SECRET_LABEL_SHOULD_NOT_LEAK",
    }
    query = "FY2025 주요 제품"
    pairs = cross_encoder_pairs(query, [candidate])
    assert_model_inputs(query, [candidate], pairs)
    assert pairs == [(query, "본문만 전달한다")]
    blob = pairs[0][0] + "\n" + pairs[0][1]
    assert "Q01-C1" not in blob
    assert "SECRET_LABEL_SHOULD_NOT_LEAK" not in blob
    assert "E186" not in blob
    poisoned = dict(candidate)
    poisoned["text"] = "본문만 전달한다 Q01-C1"
    leaked = cross_encoder_pairs(query, [poisoned])
    with pytest.raises(UsageError):
        assert_model_inputs(query, [candidate], leaked)


def test_reranker_order_is_deterministic_and_ignores_retriever_scores() -> None:
    candidates = [
        {
            "retrieval_unit_id": "b",
            "canonical_chunk_id": "cb",
            "canonical_ordinal": 2,
            "unit_index": 0,
            "stable_index": 1,
            "source_membership": "bm25_only",
            "bm25_rank": 1,
            "dense_rank": None,
            "text": "high bm25",
        },
        {
            "retrieval_unit_id": "a",
            "canonical_chunk_id": "ca",
            "canonical_ordinal": 1,
            "unit_index": 0,
            "stable_index": 0,
            "source_membership": "dense_only",
            "bm25_rank": None,
            "dense_rank": 1,
            "text": "high dense",
        },
        {
            "retrieval_unit_id": "c",
            "canonical_chunk_id": "cc",
            "canonical_ordinal": 3,
            "unit_index": 0,
            "stable_index": 2,
            "source_membership": "both",
            "bm25_rank": 2,
            "dense_rank": 2,
            "text": "tied later",
        },
    ]
    scores = [0.1, 0.1, 0.5]
    first = order_by_cross_encoder(candidates, scores)
    second = order_by_cross_encoder(candidates, scores)
    assert first == second
    assert [row["chunk_id"] for row in first] == ["c", "a", "b"]
    assert first[1]["score"] == pytest.approx(first[2]["score"])
    assert first[1]["score"] == pytest.approx(0.1)
    assert first[0]["score"] == pytest.approx(0.5)


def test_candidate_miss_and_reranking_miss_are_distinct() -> None:
    assert (
        classify_stage(covered_by_corpus=True, covered_by_pool=False, rerank_rank=None, k=20)
        == STAGE_CANDIDATE_MISS
    )
    assert (
        classify_stage(covered_by_corpus=True, covered_by_pool=True, rerank_rank=31, k=20)
        == STAGE_RERANK_MISS
    )
    assert (
        classify_stage(covered_by_corpus=True, covered_by_pool=True, rerank_rank=20, k=20)
        == STAGE_IN_TOPK
    )
    assert (
        classify_stage(covered_by_corpus=False, covered_by_pool=False, rerank_rank=None, k=20)
        == STAGE_NOT_IN_CORPUS
    )
    inside = _paragraph("in", "gold")
    outside = _paragraph("out", "gold")
    claim = _claim("Q05-C3", "gold")
    assert claim_covered([inside], claim) is True
    assert claim_covered([outside], claim) is True
    assert claim_covered([], claim) is False
    pool_has_gold = claim_covered([inside], claim)
    assert classify_stage(
        covered_by_corpus=True,
        covered_by_pool=pool_has_gold,
        rerank_rank=4,
        k=5,
    ) == STAGE_IN_TOPK
    assert classify_stage(
        covered_by_corpus=True,
        covered_by_pool=claim_covered([], claim),
        rerank_rank=None,
        k=5,
    ) == STAGE_CANDIDATE_MISS


def test_dense_only_pool_keeps_dense_order_and_drops_bm25_only() -> None:
    units = {
        "bm25-only": _unit("bm25-only", "bm25 body"),
        "both": _unit("both", "shared body"),
        "dense-only": _unit("dense-only", "dense body"),
    }
    bm25 = [_hit("bm25-only", 1, 4.0), _hit("both", 2, 3.0), _hit("dense-only", 80, 0.2)]
    dense = [_hit("both", 1, 0.9), _hit("dense-only", 2, 0.4), _hit("bm25-only", 140, 0.01)]
    pool = build_dense_only_candidate_pool(dense, units, bm25_hits=bm25, top_n=2)
    assert [row["retrieval_unit_id"] for row in pool] == ["both", "dense-only"]
    assert len(pool) == 2
    assert all(row["source_membership"] != "bm25_only" for row in pool)
    assert [row["stable_index"] for row in pool] == [0, 1]
    assert [row["dense_rank"] for row in pool] == [1, 2]
    assert pool[0]["source_membership"] == "both"
    assert pool[0]["bm25_rank"] == 2
    assert pool[1]["source_membership"] == "dense_only"
    assert pool[1]["bm25_rank"] == 80
    assert pool[1]["in_bm25_top_n"] is False
    assert "bm25-only" not in {row["retrieval_unit_id"] for row in pool}


def test_dense_only_tie_break_follows_dense_rank_and_ignores_bm25_score() -> None:
    units = {"later": _unit("later", "later"), "earlier": _unit("earlier", "earlier")}
    dense = [_hit("earlier", 1, 0.2), _hit("later", 2, 0.1)]
    bm25 = [_hit("later", 1, 9.0), _hit("earlier", 2, 1.0)]
    pool = build_dense_only_candidate_pool(dense, units, bm25_hits=bm25, top_n=2)
    ordered = order_by_cross_encoder(pool, [0.5, 0.5])
    assert [row["chunk_id"] for row in ordered] == ["earlier", "later"]
    assert ordered[0]["score"] == pytest.approx(ordered[1]["score"])
    again = order_by_cross_encoder(pool, [0.5, 0.5])
    assert again == ordered


def test_dense_only_model_input_is_query_and_retrieval_text() -> None:
    candidate = {
        "text": "본문만 전달한다",
        "retrieval_unit_id": "u1",
        "claim_id": "Q01-C1",
        "bm25_rank": 41,
        "dense_rank": 18,
        "source_membership": "both",
        "evidence_class": "answer",
        "element_key": "E187",
    }
    query = "FY2025 주요 제품"
    pairs = cross_encoder_pairs(query, [candidate])
    assert_model_inputs(query, [candidate], pairs)
    blob = pairs[0][0] + "\n" + pairs[0][1]
    assert blob == "FY2025 주요 제품\n본문만 전달한다"
    assert "E187" not in blob
    assert "Q01-C1" not in blob


def test_common_pair_scores_match_within_tolerance() -> None:
    reference = {("Q01", "gold"): -3.232, ("Q01", "bm25-added"): 1.5, ("Q05", "c3"): 0.2}
    observed = {("Q01", "gold"): -3.232, ("Q05", "c3"): 0.2}
    report = pair_score_identity(reference, observed, tolerance=1e-4)
    assert report["common_pair_count"] == 2
    assert report["reference_only_count"] == 1
    assert report["observed_only_count"] == 0
    assert report["mismatch_count"] == 0
    assert report["exact_match_count"] == 2
    assert report["identical_within_tolerance"] is True
    drifted = dict(observed)
    drifted[("Q01", "gold")] = -3.0
    broken = pair_score_identity(reference, drifted, tolerance=1e-4)
    assert broken["mismatch_count"] == 1
    assert broken["identical_within_tolerance"] is False


def test_common_candidate_text_must_match_the_union_run() -> None:
    union_rows = [
        {"question_id": "Q01", "retrieval_unit_id": "gold", "text_sha256": "aaa"},
        {"question_id": "Q01", "retrieval_unit_id": "bm25-added", "text_sha256": "bbb"},
    ]
    dense_rows = [{"question_id": "Q01", "retrieval_unit_id": "gold", "text_sha256": "aaa"}]
    assert common_candidate_text_identity(union_rows, dense_rows)["identical"] is True
    changed = [{"question_id": "Q01", "retrieval_unit_id": "gold", "text_sha256": "ccc"}]
    mismatch = common_candidate_text_identity(union_rows, changed)
    assert mismatch["identical"] is False
    assert mismatch["text_mismatch_count"] == 1
    missing = [{"question_id": "Q01", "retrieval_unit_id": "new", "text_sha256": "ddd"}]
    absent = common_candidate_text_identity(union_rows, missing)
    assert absent["missing_from_union_count"] == 1


def test_bm25_added_impact_counts_rows_above_the_gold_logit() -> None:
    added = [
        {"union_rank": 2, "ce_score": 1.0, "answer_valid_gold": False, "known_conflict": False},
        {"union_rank": 6, "ce_score": -1.0, "answer_valid_gold": False, "known_conflict": True},
        {"union_rank": 12, "ce_score": -4.0, "answer_valid_gold": True, "known_conflict": False},
    ]
    report = bm25_added_impact_row(added, gold_score=-3.232)
    assert report["bm25_added_count"] == 3
    assert report["union_top5"] == 1
    assert report["union_top10"] == 2
    assert report["union_top20"] == 3
    assert report["above_required_gold"] == 2
    assert report["answer_valid_gold"] == 1
    assert report["known_conflict"] == 1
    assert bm25_added_impact_row(added, gold_score=None)["above_required_gold"] is None


def test_expansion_interference_requires_exact_rank_accounting() -> None:
    assert expansion_interference(
        union_rank=38, dense_only_rank=18, above_gold=20, scores_identical=True
    )
    assert not expansion_interference(
        union_rank=38, dense_only_rank=18, above_gold=19, scores_identical=True
    )
    assert not expansion_interference(
        union_rank=38, dense_only_rank=18, above_gold=20, scores_identical=False
    )
    assert not expansion_interference(
        union_rank=7, dense_only_rank=7, above_gold=0, scores_identical=True
    )


def test_dense_pool_miss_stays_distinct_from_a_later_rerank_miss() -> None:
    assert (
        classify_stage(covered_by_corpus=True, covered_by_pool=False, rerank_rank=None, k=20)
        == STAGE_CANDIDATE_MISS
    )
    assert (
        classify_stage(covered_by_corpus=True, covered_by_pool=True, rerank_rank=38, k=20)
        == STAGE_RERANK_MISS
    )
    assert (
        classify_stage(covered_by_corpus=True, covered_by_pool=True, rerank_rank=7, k=20)
        == STAGE_IN_TOPK
    )


def test_truncation_summary_counts_pairs_over_the_budget() -> None:
    report = summarize_truncation(
        [
            {"pair_tokens": 100, "query_tokens": 10, "document_tokens": 87},
            {"pair_tokens": 1025, "query_tokens": 20, "document_tokens": 1001},
        ],
        1024,
    )
    assert report["total_candidates"] == 2
    assert report["truncated_candidates"] == 1
    assert report["truncation_rate"] == pytest.approx(0.5)
    assert report["max_input_tokens"] == 1025
    assert report["max_query_tokens"] == 20
    assert report["max_document_tokens"] == 1001
