"""H11 ranking-failure diagnosis. These tests do not load an embedding model."""

from __future__ import annotations

from pathlib import Path

import pytest

from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2.spec import repo_root
from disclosure_agent.evalset_v2_1.ranking_failure import (
    CANDIDATE_N,
    FAILURE_CANDIDATE,
    FAILURE_NONE,
    FAILURE_RANKING,
    HIT_K,
    REQUIRED_SEMANTICS_TOKENS,
    SEMANTICS_RELATIVE,
    VERDICT_PARTIAL,
    VERDICT_REJECT,
    VERDICT_STRONG,
    build_c20,
    diagnose_query,
    judge_h11,
    maximum_requirement_cover,
)


def _hits(order: list[str]) -> list[dict]:
    return [
        {
            "rank": index,
            "retrieval_unit_id": unit_id,
            "source_element_ids": [f"e-{unit_id}"],
        }
        for index, unit_id in enumerate(order, start=1)
    ]


def _question() -> dict:
    return {
        "query_id": "Q1",
        "company_name": "예시",
        "corp_code": "00000000",
        "question": "질문",
        "stratum_id": "finance_table",
        "query_type": "single_fact",
        "evidence_form": "table",
        "wording_style": "original_term",
        "period_sensitive": False,
        "answerability": "fully_answerable",
        "split": "dev",
    }


def _requirements(*requirement_ids: str) -> list[dict]:
    return [
        {
            "query_id": "Q1",
            "requirement_id": requirement_id,
            "answerability": "fully_answerable",
            "text": requirement_id,
        }
        for requirement_id in requirement_ids
    ]


def _qrel(unit_id: str, requirement_id: str, relevance: int) -> dict:
    return {
        "query_id": "Q1",
        "requirement_id": requirement_id,
        "retrieval_unit_id": unit_id,
        "relevance": relevance,
        "preferred": relevance >= 1 and requirement_id == "R1",
        "judgment_status": "judged",
        "document_name": "사업보고서",
        "filing_period": "2024.12",
    }


def test_semantics_document_locks_the_tree() -> None:
    text = (repo_root() / SEMANTICS_RELATIVE).read_text(encoding="utf-8")
    for token in REQUIRED_SEMANTICS_TOKENS:
        assert token in text
    module = Path(__file__).resolve().parents[1] / "src" / "disclosure_agent" / (
        "evalset_v2_1/ranking_failure.py"
    )
    source = module.read_text(encoding="utf-8")
    assert "test_h10_validation" not in source
    assert "multi_company_retrieval_v2_1_test" not in source


def test_c20_deduplicates_and_refuses_conflicting_source_elements() -> None:
    dense = _hits(["both", "dense-only"])
    bm25 = _hits(["both", "bm25-only"])
    pool = build_c20(dense, bm25, n=2)
    assert set(pool) == {"both", "dense-only", "bm25-only"}
    assert pool["both"]["bm25_rank"] == 1
    assert pool["both"]["dense_rank"] == 1
    dense[0]["source_element_ids"] = ["other"]
    with pytest.raises(UsageError, match="source elements"):
        build_c20(dense, bm25, n=2)


def test_requirement_cover_uses_the_budget_and_does_not_copy_grades() -> None:
    shared = maximum_requirement_cover(
        ["R1", "R2"],
        {"u1": {"R1", "R2"}},
        k=10,
    )
    assert shared["covered_count"] == 2
    assert shared["all_covered"] is True
    assert shared["min_units_for_all"] == 1
    bounded = maximum_requirement_cover(
        ["R1", "R2", "R3"],
        {"u1": {"R1"}, "u2": {"R2"}, "u3": {"R3"}},
        k=2,
    )
    assert bounded["covered_count"] == 2
    assert bounded["all_covered"] is False
    assert set(bounded["requirements_in_some_optimal_cover"]) == {"R1", "R2", "R3"}


def test_dense_rank_20_relevant_is_a_ranking_failure() -> None:
    bm25_ids = [f"b{index:02d}" for index in range(1, CANDIDATE_N + 1)]
    dense_ids = [f"d{index:02d}" for index in range(1, CANDIDATE_N + 1)]
    dense_ids[-1] = "gold"
    row = diagnose_query(
        _question(),
        dense_hits=_hits(dense_ids),
        bm25_hits=_hits(bm25_ids),
        qrels=[_qrel("gold", "R1", 2), _qrel("b01", "R1", 0)],
        requirements=_requirements("R1"),
    )
    assert row["failure_type"] == FAILURE_RANKING
    assert row["h10_hit10"] is False
    assert row["oracle_hit10"] is True
    assert row["bm25_best_relevant_rank"] is None
    assert row["dense_best_relevant_rank"] == CANDIDATE_N
    assert row["relevant_in_c20"] == 1
    assert row["relevant_in_h10_top10"] == 0
    assert row["oracle_recall10_numerator"] == 1
    assert "d02" not in [hit["retrieval_unit_id"] for hit in row["h10_top10"]]


def test_relevant_outside_both_lists_is_candidate_generation_failure() -> None:
    bm25_ids = [f"b{index:02d}" for index in range(1, CANDIDATE_N + 1)]
    dense_ids = [f"d{index:02d}" for index in range(1, CANDIDATE_N + 1)]
    row = diagnose_query(
        _question(),
        dense_hits=_hits(dense_ids),
        bm25_hits=_hits(bm25_ids),
        qrels=[_qrel("gold", "R1", 1)],
        requirements=_requirements("R1"),
    )
    assert row["failure_type"] == FAILURE_CANDIDATE
    assert row["oracle_hit10"] is False
    assert row["any_relevant_in_c20"] is False
    assert row["bm25_best_relevant_rank"] is None
    assert row["dense_best_relevant_rank"] is None
    assert row["requirements"][0]["status"] == "candidate_missing"


def test_a_unit_does_not_cover_a_sibling_requirement() -> None:
    bm25_ids = [f"b{index:02d}" for index in range(1, CANDIDATE_N + 1)]
    bm25_ids[0] = "gold-r1"
    dense_ids = [f"d{index:02d}" for index in range(1, CANDIDATE_N + 1)]
    row = diagnose_query(
        _question(),
        dense_hits=_hits(dense_ids),
        bm25_hits=_hits(bm25_ids),
        qrels=[_qrel("gold-r1", "R1", 2), _qrel("missing", "R2", 1)],
        requirements=_requirements("R1", "R2"),
    )
    by_requirement = {item["requirement_id"]: item for item in row["requirements"]}
    assert row["failure_type"] == FAILURE_NONE
    assert by_requirement["R1"]["status"] == "covered"
    assert by_requirement["R2"]["status"] == "candidate_missing"
    assert row["oracle_requirements_coverable"] == 1
    assert row["oracle_all_requirements_covered"] is False


def test_relevance_zero_and_unjudged_are_not_positive() -> None:
    bm25_ids = [f"b{index:02d}" for index in range(1, CANDIDATE_N + 1)]
    dense_ids = [f"d{index:02d}" for index in range(1, CANDIDATE_N + 1)]
    row = diagnose_query(
        _question(),
        dense_hits=_hits(dense_ids),
        bm25_hits=_hits(bm25_ids),
        qrels=[_qrel("b01", "R1", 0)],
        requirements=_requirements("R1"),
    )
    assert row["relevant_unit_count"] == 0
    assert row["h10_hit10"] is False
    assert row["oracle_hit10"] is False


def test_recall_oracle_caps_at_hit_k() -> None:
    bm25_ids = [f"g{index:02d}" for index in range(1, CANDIDATE_N + 1)]
    dense_ids = [bm25_ids[0], *[f"d{index:02d}" for index in range(2, CANDIDATE_N + 1)]]
    qrels = [_qrel(unit_id, "R1", 1) for unit_id in bm25_ids]
    qrels.append(_qrel("outside", "R1", 1))
    row = diagnose_query(
        _question(),
        dense_hits=_hits(dense_ids),
        bm25_hits=_hits(bm25_ids),
        qrels=qrels,
        requirements=_requirements("R1"),
    )
    assert row["relevant_unit_count"] == CANDIDATE_N + 1
    assert row["relevant_in_c20"] == CANDIDATE_N
    assert row["oracle_recall10_numerator"] == HIT_K
    assert row["h10_recall10_numerator"] == HIT_K
    assert row["recall_ranking_headroom_numerator"] == 0
    assert row["failure_type"] == FAILURE_NONE


def test_verdict_tree_matches_the_frozen_comparisons() -> None:
    strong = judge_h11(
        h10_hit10=40,
        oracle_hit10=43,
        h10_requirement_covered=50,
        oracle_requirement_covered=53,
        ranking_failure_queries=3,
        candidate_generation_failure_queries=1,
        ranking_failure_strata=["finance_table", "cross_period", "finance_table"],
        requirement_ranking_missed=3,
    )
    assert strong["label"] == VERDICT_STRONG
    concentrated = judge_h11(
        h10_hit10=43,
        oracle_hit10=44,
        h10_requirement_covered=52,
        oracle_requirement_covered=53,
        ranking_failure_queries=1,
        candidate_generation_failure_queries=1,
        ranking_failure_strata=["finance_table"],
        requirement_ranking_missed=1,
    )
    assert concentrated["label"] == VERDICT_PARTIAL
    assert concentrated["reason"] == "candidate_generation_comparable_or_larger"
    requirement_only = judge_h11(
        h10_hit10=43,
        oracle_hit10=43,
        h10_requirement_covered=52,
        oracle_requirement_covered=54,
        ranking_failure_queries=0,
        candidate_generation_failure_queries=2,
        ranking_failure_strata=[],
        requirement_ranking_missed=2,
    )
    assert requirement_only["label"] == VERDICT_PARTIAL
    assert requirement_only["reason"] == "requirement_only_headroom"
    rejected = judge_h11(
        h10_hit10=43,
        oracle_hit10=43,
        h10_requirement_covered=52,
        oracle_requirement_covered=52,
        ranking_failure_queries=0,
        candidate_generation_failure_queries=2,
        ranking_failure_strata=[],
        requirement_ranking_missed=0,
    )
    assert rejected["label"] == VERDICT_REJECT
    with pytest.raises(UsageError, match="oracle is below"):
        judge_h11(
            h10_hit10=44,
            oracle_hit10=43,
            h10_requirement_covered=52,
            oracle_requirement_covered=52,
            ranking_failure_queries=0,
            candidate_generation_failure_queries=0,
            ranking_failure_strata=[],
            requirement_ranking_missed=0,
        )
