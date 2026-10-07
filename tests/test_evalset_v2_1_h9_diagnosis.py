"""H9 rank-pressure diagnosis. These tests do not load an embedding model."""

from __future__ import annotations

import inspect

import pytest

from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2.spec import repo_root
from disclosure_agent.evalset_v2_1.truncated_rrf import CANDIDATE_N, candidate_truncated_rrf
from disclosure_agent.evalset_v2_1.truncated_rrf_diagnosis import (
    MECHANISM_BOTH,
    MECHANISM_DOMINANCE,
    MECHANISM_SYMMETRIC,
    aggregate_dev_diagnostics,
    attribute_rank_pressure,
    build_diagnosis,
    decompose_fused_ranking,
    dense_rank1_weight_ratio,
    intersection_dominance_weight_ratio,
    judge_h9,
    spearman,
    structural_property,
)


def _hit(doc_id: str, rank: int) -> dict:
    return {
        "rank": rank,
        "chunk_id": doc_id,
        "retrieval_unit_id": doc_id,
        "score": 1.0 / rank,
        "source_element_ids": [f"e-{doc_id}"],
    }


def _list(order: list[str]) -> list[dict]:
    return [_hit(doc_id, index) for index, doc_id in enumerate(order, start=1)]


def _fused(dense: list[str], bm25: list[str], qrels: list[dict] | None = None) -> list[dict]:
    return decompose_fused_ranking(_list_fuse(dense, bm25), qrels or [])


def _list_fuse(dense: list[str], bm25: list[str]) -> list[dict]:
    return candidate_truncated_rrf(_list(dense), _list(bm25))


def _zero_intersection() -> tuple[list[dict], str]:
    dense = [f"d{index:02d}" for index in range(CANDIDATE_N)]
    dense[7] = "z-tie"
    bm25 = [f"b{index:02d}" for index in range(7)]
    bm25.append("m-positive")
    bm25.extend(f"c{index:02d}" for index in range(12))
    qrels = [
        {
            "retrieval_unit_id": "m-positive",
            "requirement_id": "R1",
            "relevance": 2,
            "judgment_status": "judged",
        },
        {
            "retrieval_unit_id": "m-positive",
            "requirement_id": "R2",
            "relevance": 0,
            "judgment_status": "judged",
        },
    ]
    return _fused(dense, bm25, qrels), "m-positive"


def _mixed_pressure() -> tuple[list[dict], str]:
    bm25 = ["s0", "s1", "s2", "m-positive", "s3", "s4", "s5", "s6", "s7"]
    bm25.extend(f"b{index:02d}" for index in range(10, 21))
    dense = ["a-dense", "b-dense", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    dense.extend(f"d{index:02d}" for index in range(11, 21))
    return _fused(dense, bm25), "m-positive"


def _case(**overrides: object) -> dict:
    base = {
        "deep_tail_rank_used": False,
        "positive_dense_contribution": 0,
        "max_contributing_rank": 20,
        "structural_property_holds": True,
        "identity_holds": True,
        "mechanism": MECHANISM_BOTH,
    }
    base.update(overrides)
    return base


def test_zero_intersection_regression_is_dense_only_competition() -> None:
    rows, positive = _zero_intersection()
    prop = structural_property(rows)
    assert prop["holds"] is True
    assert prop["vacuous_intersection"] is True
    assert prop["intersection_count"] == 0
    attr = attribute_rank_pressure(rows, positive_unit_id=positive, bm25_rank=8)
    assert attr["truncated_rank"] == 15
    assert attr["ahead_counts"] == {"intersection": 0, "dense_only": 7, "bm25_only": 7}
    assert attr["incremental_intersection_pressure"] == 0
    assert attr["symmetric_competition_pressure"] == 7
    assert attr["identity_holds"] is True
    assert attr["mechanism"] == MECHANISM_SYMMETRIC
    assert attr["counterfactual_a_rank"] == 15
    assert attr["counterfactual_b_dense_only_ahead"] == 7
    assert attr["counterfactual_c_bm25_rank"] == 8
    detail = attr["counterfactual_d"]
    assert detail["minimum_displacement_to_exit_top10"] == 3
    assert detail["observed_displacement"] == 7
    assert detail["minimum_removals_to_restore_top10"] == 5
    assert detail["removal_restores_hit10"]["dense_only"] is True
    assert detail["removal_restores_hit10"]["intersection"] is False
    assert detail["displacement_category_can_supply_exit"]["dense_only"] is True
    assert detail["displacement_category_can_supply_exit"]["intersection"] is False
    positive_row = next(row for row in rows if row["retrieval_unit_id"] == positive)
    assert positive_row["relevant_requirement_ids"] == ["R1"]
    assert positive_row["relevance"] == 2
    assert positive_row["judgment_status"] == "judged"
    unjudged = next(row for row in rows if row["retrieval_unit_id"] == "d00")
    assert unjudged["relevance"] is None
    assert unjudged["judgment_status"] == "unjudged"
    assert unjudged["relevant_requirement_ids"] == []


def test_queue_jumps_and_dense_competition_are_both_required_to_exit() -> None:
    rows, positive = _mixed_pressure()
    prop = structural_property(rows)
    assert prop["holds"] is True
    assert prop["intersection_count"] == 8
    assert prop["worst_intersection_score"] > prop["best_single_list_score"]
    attr = attribute_rank_pressure(rows, positive_unit_id=positive, bm25_rank=4)
    assert attr["truncated_rank"] == 11
    assert attr["ahead_counts"]["intersection"] == 8
    assert attr["ahead_counts"]["dense_only"] == 2
    assert attr["ahead_counts"]["bm25_only"] == 0
    assert attr["incremental_intersection_pressure"] == 5
    assert attr["symmetric_competition_pressure"] == 2
    assert attr["counterfactual_a_rank"] == 3
    assert attr["counterfactual_a_queue_jump_only_rank"] == 6
    assert attr["counterfactual_a_removes_predecessors"] == 3
    assert attr["mechanism"] == MECHANISM_BOTH
    detail = attr["counterfactual_d"]
    assert detail["minimum_displacement_to_exit_top10"] == 7
    assert detail["observed_displacement"] == 7
    assert detail["minimum_removals_to_restore_top10"] == 1
    assert detail["removal_restores_hit10"]["queue_jumps"] is True
    assert detail["removal_restores_hit10"]["dense_only"] is True
    assert detail["displacement_category_can_supply_exit"]["intersection"] is False
    assert detail["displacement_category_can_supply_exit"]["dense_only"] is False
    assert all(unit["relevance"] is None for unit in attr["displacement_units"])


def test_weight_ratio_boundaries_are_closed_form_and_reject_a_sweep_rank() -> None:
    assert intersection_dominance_weight_ratio(1) == pytest.approx(61 / 19)
    assert intersection_dominance_weight_ratio(8) == pytest.approx(68 / 12)
    assert dense_rank1_weight_ratio(8) == pytest.approx(68 / 61)
    with pytest.raises(UsageError):
        intersection_dominance_weight_ratio(20)
    with pytest.raises(UsageError):
        dense_rank1_weight_ratio(0)


def test_h9_verdict_follows_whether_the_three_mechanisms_match() -> None:
    strong = judge_h9([_case(), _case(), _case()])
    partial = judge_h9(
        [
            _case(mechanism=MECHANISM_SYMMETRIC),
            _case(mechanism=MECHANISM_BOTH),
            _case(mechanism=MECHANISM_DOMINANCE),
        ]
    )
    rejected = judge_h9([_case(deep_tail_rank_used=True), _case(), _case()])
    assert strong["decision"] == "Strong Support"
    assert partial["decision"] == "Partial Support"
    assert partial["mechanisms_identical"] is False
    assert rejected["decision"] == "Reject"
    assert strong["asymmetric_fusion_justified"] is False
    assert strong["weight_sweep_run"] is False


def test_spearman_is_descriptive_and_undefined_on_a_constant() -> None:
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def test_aggregate_keeps_the_association_caveat() -> None:
    queries = []
    for index, intersection in enumerate((0, 0, 4, 8)):
        queries.append(
            {
                "query_id": f"q{index}",
                "movement_vs_bm25": "regression" if intersection else "neutral",
                "intersection_count": intersection,
                "dense_only_count": 20 - intersection,
                "dense_only_ahead": intersection,
                "same_unit_rank_change": intersection,
                "bm25_positive_rank": 4,
                "bm25_positive_truncated_rank": 4 + intersection,
                "bm25_hit10": True,
                "truncated_hit10": intersection == 0,
                "structural_property_holds": True,
                "identity_applies": True,
                "identity_holds": True,
                "incremental_intersection_pressure": intersection,
                "symmetric_competition_pressure": 0,
                "same_positive_unit": True,
            }
        )
    aggregate = aggregate_dev_diagnostics(queries)
    assert aggregate["association_note"]
    assert "인과" in aggregate["association_note"]
    assert aggregate["intersection_count_distribution"][0]["intersection_count"] == 0
    assert aggregate["identity_regressions"]["with_queue_jumps"] == 2
    assert all(row["reading"] for row in aggregate["associations"])


def test_diagnosis_entrypoint_has_no_weight_or_cutoff_argument() -> None:
    assert list(inspect.signature(build_diagnosis).parameters) == ["root"]


def test_stored_dev_diagnosis_matches_the_three_pinned_residuals() -> None:
    payload = build_diagnosis(repo_root())
    assert payload["question_count"] == 45
    assert payload["test_questions_searched"] == 0
    assert payload["retrieval_reexecuted"] is False
    assert payload["new_ranker_implemented"] is False
    assert payload["max_contributing_rank"] <= CANDIDATE_N
    assert payload["verdict"]["decision"] == "Partial Support"
    assert payload["verdict"]["mechanisms_identical"] is False
    assert payload["controlled_experiment"]["implemented"] is False
    assert payload["controlled_experiment"]["weight_sweep_warranted"] is False
    by_id = {case["query_id"]: case for case in payload["residual_cases"]}
    equity = by_id["V2-00121941-business_paragraph-03"]
    related = by_id["V2-00121941-finance_table-02"]
    preferred = by_id["V2-00989619-governance_paragraph-02"]
    assert equity["mechanism"] == MECHANISM_SYMMETRIC
    assert equity["incremental_intersection_pressure"] == 0
    assert equity["symmetric_competition_pressure"] == 7
    assert related["incremental_intersection_pressure"] >= 2
    assert preferred["incremental_intersection_pressure"] >= 5
    for case in payload["residual_cases"]:
        assert case["identity_holds"] is True
        assert case["positive_dense_contribution"] == 0
        assert case["structural_property_holds"] is True
        change = case["same_unit_rank_change"]
        pressure = case["incremental_intersection_pressure"]
        assert pressure + case["symmetric_competition_pressure"] == change
    assert "Partial Support" in payload["answers"]["F"]
