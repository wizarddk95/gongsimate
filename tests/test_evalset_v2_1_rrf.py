"""RRF contract and DEV bookkeeping. These tests do not load an embedding model."""

from __future__ import annotations

import pytest

from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2.spec import repo_root
from disclosure_agent.evalset_v2_1.complementarity import (
    CELL_BM25,
    CELL_BOTH,
    CELL_DENSE,
    CELL_MISS,
)
from disclosure_agent.evalset_v2_1.rrf import (
    fuse_complete_rankings,
    implementation_audit,
    judge_hypothesis,
    movement_label,
    prefix_mismatch,
    rank_delta,
    requirement_transitions,
    summarize_retention,
    validate_source_bundle,
)
from disclosure_agent.hybrid import reciprocal_rank_fusion


def _candidate(doc_id: str, rank: int, *, ordinal: int = 0) -> dict:
    return {
        "rank": rank,
        "chunk_id": doc_id,
        "ordinal": ordinal,
        "unit_index": 0,
        "retrieval_unit_id": doc_id,
        "score": 1.0,
    }


def test_reused_rrf_rejects_a_document_missing_from_one_ranking() -> None:
    bm25 = [_candidate("a", 1), _candidate("b", 2)]
    dense = [_candidate("a", 1)]
    with pytest.raises(UsageError, match="same candidate"):
        reciprocal_rank_fusion(bm25, dense)
    audit = implementation_audit()
    assert audit["k"] == 60
    assert audit["absent_contribution_is_zero"] is False
    assert audit["weighted_score_fusion"] is False
    assert audit["zero_tie_variant_used"] is False
    assert audit["formula"] == "score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_dense(d))"


def test_fusion_score_is_the_sum_of_reciprocal_ranks_and_ids_are_unique() -> None:
    bm25 = [_candidate("a", 1, ordinal=2), _candidate("b", 2, ordinal=1)]
    dense = [_candidate("b", 1, ordinal=1), _candidate("a", 2, ordinal=2)]
    fused = fuse_complete_rankings(bm25, dense)
    assert [hit["retrieval_unit_id"] for hit in fused] == ["b", "a"]
    assert fused[0]["score"] == pytest.approx(1 / 61 + 1 / 62)
    assert len({hit["retrieval_unit_id"] for hit in fused}) == 2


def test_prefix_mismatch_names_the_first_disagreeing_rank() -> None:
    stored = [
        {"retrieval_unit_id": "a", "rank": 1},
        {"retrieval_unit_id": "b", "rank": 2},
    ]
    assert prefix_mismatch("V2-x", "dense", stored, stored) is None
    found = prefix_mismatch(
        "V2-x",
        "dense",
        stored,
        [{"retrieval_unit_id": "a", "rank": 1}, {"retrieval_unit_id": "c", "rank": 2}],
    )
    assert found is not None
    assert found["query_id"] == "V2-x"
    assert found["rank"] == 2
    assert found["fresh_retrieval_unit_id"] == "c"


def test_movement_label_and_delta_follow_the_first_relevant_rank() -> None:
    assert movement_label(8, 2) == "improvement"
    assert rank_delta(8, 2) == 6
    assert movement_label(8, 8) == "neutral"
    assert rank_delta(8, 8) == 0
    assert movement_label(8, 11) == "regression"
    assert rank_delta(8, 11) == -3
    assert movement_label(None, 4) == "recovered"
    assert rank_delta(None, 4) is None
    assert movement_label(None, 15) == "still_missing"
    assert movement_label(None, None) == "still_missing"
    assert movement_label(4, None) == "regression"


def test_retention_counts_bm25_only_losses_and_dense_only_recovery() -> None:
    rows = [
        _row("both", CELL_BOTH, True),
        _row("sparse", CELL_BM25, False),
        _row("dense", CELL_DENSE, True),
        _row("miss", CELL_MISS, False),
    ]
    summary = summarize_retention(rows)
    assert summary["both_hit"]["rrf_hit10"] == 1
    assert summary["bm25_only"]["lost_query_ids"] == ["sparse"]
    assert summary["dense_only"]["recovered"] == 1
    assert summary["both_miss"]["still_missing"] == 1


def test_requirement_transitions_keep_bm25_only_and_recover_dense_only() -> None:
    prior = [
        {"query_id": "q", "requirement_id": "R1", "cell": "bm25_only", "requirement_text": "a"},
        {"query_id": "q", "requirement_id": "R2", "cell": "dense_only", "requirement_text": "b"},
        {"query_id": "q", "requirement_id": "R3", "cell": "neither", "requirement_text": "c"},
        {"query_id": "q", "requirement_id": "R4", "cell": "both", "requirement_text": "d"},
    ]
    result = requirement_transitions(prior, {"q": {"R2", "R4"}})
    assert result["cells"]["bm25_only"]["transitions"]["lost"] == 1
    assert result["cells"]["dense_only"]["transitions"]["recovered"] == 1
    assert result["cells"]["neither"]["transitions"]["still_missing"] == 1
    assert result["cells"]["both"]["transitions"]["retained"] == 1


def test_verdict_is_a_when_coverage_holds_and_dense_only_is_recovered() -> None:
    verdict = judge_hypothesis(
        bm25_hit10=42,
        rrf_hit10=42,
        bm25_only_lost=0,
        dense_only_total=1,
        dense_only_recovered=1,
        bm25_mrr=0.7,
        rrf_mrr=0.7,
        bm25_ndcg=0.7,
        rrf_ndcg=0.7,
        bm25_recall5=0.5,
        rrf_recall5=0.5,
    )
    assert verdict["label"] == "A"


def test_verdict_is_b_when_coverage_holds_and_nothing_improves() -> None:
    verdict = judge_hypothesis(
        bm25_hit10=42,
        rrf_hit10=42,
        bm25_only_lost=0,
        dense_only_total=1,
        dense_only_recovered=0,
        bm25_mrr=0.7,
        rrf_mrr=0.7,
        bm25_ndcg=0.7,
        rrf_ndcg=0.69,
        bm25_recall5=0.5,
        rrf_recall5=0.5,
    )
    assert verdict["label"] == "B"


def test_verdict_is_c_when_two_bm25_only_hits_are_lost() -> None:
    verdict = judge_hypothesis(
        bm25_hit10=42,
        rrf_hit10=42,
        bm25_only_lost=2,
        dense_only_total=1,
        dense_only_recovered=1,
        bm25_mrr=0.7,
        rrf_mrr=0.8,
        bm25_ndcg=0.7,
        rrf_ndcg=0.8,
        bm25_recall5=0.5,
        rrf_recall5=0.6,
    )
    assert verdict["label"] == "C"


def test_committed_complementarity_bundle_is_the_rrf_baseline() -> None:
    bundle = validate_source_bundle(repo_root())
    assert len(bundle["questions"]) == 45
    assert {row["split"] for row in bundle["questions"]} == {"dev"}
    assert bundle["config"]["test_questions_searched"] == 0
    assert bundle["dense_headline"]["at_k"]["10"]["hit"]["numerator"] == 33
    assert bundle["bm25_headline"]["at_k"]["10"]["hit"]["numerator"] == 42


def _row(query_id: str, cell: str, rrf_hit: bool) -> dict:
    return {
        "query_id": query_id,
        "company": "회사",
        "question": "질문",
        "prior_cell": cell,
        "dense_first_relevant_rank": 1,
        "bm25_first_relevant_rank": 1 if cell in {CELL_BOTH, CELL_BM25} else None,
        "rrf_first_relevant_rank": 1 if rrf_hit else 12,
        "movement": "neutral",
        "rrf_hit10": rrf_hit,
        "rrf_first_relevant_preview": "",
    }
