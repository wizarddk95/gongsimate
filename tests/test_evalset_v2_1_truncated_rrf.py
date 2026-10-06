"""Candidate-truncated RRF. These tests do not load an embedding model."""

from __future__ import annotations

import pytest

from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2.spec import repo_root
from disclosure_agent.evalset_v2_1.rrf import RUN_DIR_NAME as FULL_RUN_DIR
from disclosure_agent.evalset_v2_1.rrf import validate_source_bundle
from disclosure_agent.evalset_v2_1.truncated_rrf import (
    CANDIDATE_N,
    candidate_truncated_rrf,
    implementation_audit,
    intersection_min_score,
    judge_truncated_hypothesis,
    ranking_mismatches,
    single_list_max_score,
)
from disclosure_agent.hybrid import RRF_K, reciprocal_rank_fusion


def _hit(doc_id: str, rank: int, *, ordinal: int | None = None) -> dict:
    return {
        "rank": rank,
        "chunk_id": doc_id,
        "ordinal": rank if ordinal is None else ordinal,
        "unit_index": rank,
        "retrieval_unit_id": doc_id,
        "score": 1.0 / rank,
        "source_element_ids": [f"e-{doc_id}"],
    }


def _list(order: list[str]) -> list[dict]:
    return [_hit(doc_id, index) for index, doc_id in enumerate(order, start=1)]


def test_bm25_only_doc_gets_no_dense_deep_tail_contribution() -> None:
    lexical = "lexical"
    dense_ids = [f"d{index:02d}" for index in range(CANDIDATE_N)]
    bm25_ids = [lexical, *[f"b{index:02d}" for index in range(CANDIDATE_N - 1)]]
    dense = _list(dense_ids)
    dense.append(_hit(lexical, 705))
    fused = candidate_truncated_rrf(dense, _list(bm25_ids))
    row = next(hit for hit in fused if hit["retrieval_unit_id"] == lexical)
    assert row["dense_rank"] is None
    assert row["dense_contribution"] == 0
    assert row["bm25_rank"] == 1
    assert row["score"] == pytest.approx(1 / (RRF_K + 1))
    assert row["score"] != pytest.approx((1 / (RRF_K + 1)) + (1 / (RRF_K + 705)))
    assert lexical not in {hit["retrieval_unit_id"] for hit in fused if hit["in_dense_top_n"]}


def test_dense_only_doc_gets_no_bm25_deep_tail_contribution() -> None:
    semantic = "semantic"
    bm25_ids = [f"b{index:02d}" for index in range(CANDIDATE_N)]
    dense_ids = [semantic, *[f"d{index:02d}" for index in range(CANDIDATE_N - 1)]]
    bm25 = _list(bm25_ids)
    bm25.append(_hit(semantic, 283))
    fused = candidate_truncated_rrf(_list(dense_ids), bm25)
    row = next(hit for hit in fused if hit["retrieval_unit_id"] == semantic)
    assert row["bm25_rank"] is None
    assert row["bm25_contribution"] == 0
    assert row["dense_rank"] == 1
    assert row["score"] == pytest.approx(1 / (RRF_K + 1))


def test_units_in_both_top20_lists_add_both_contributions() -> None:
    dense = _list([f"d{index:02d}" for index in range(CANDIDATE_N)])
    bm25 = _list([f"b{index:02d}" for index in range(CANDIDATE_N)])
    dense[3] = _hit("shared", 4)
    bm25[6] = _hit("shared", 7)
    fused = candidate_truncated_rrf(dense, bm25)
    row = next(hit for hit in fused if hit["retrieval_unit_id"] == "shared")
    assert row["dense_contribution"] == pytest.approx(1 / (RRF_K + 4))
    assert row["bm25_contribution"] == pytest.approx(1 / (RRF_K + 7))
    assert row["score"] == pytest.approx(row["dense_contribution"] + row["bm25_contribution"])


def test_doc_outside_both_top20_lists_is_absent() -> None:
    dense_ids = [f"d{index:02d}" for index in range(CANDIDATE_N)]
    bm25_ids = [f"b{index:02d}" for index in range(CANDIDATE_N)]
    dense = _list(dense_ids)
    bm25 = _list(bm25_ids)
    dense.append(_hit("tail", 75))
    bm25.append(_hit("tail", 75))
    fused = candidate_truncated_rrf(dense, bm25)
    assert "tail" not in {hit["retrieval_unit_id"] for hit in fused}
    assert len(fused) == 40


def test_same_unit_appears_once_and_a_repeated_id_is_rejected() -> None:
    dense = _list([f"d{index:02d}" for index in range(CANDIDATE_N)])
    bm25 = _list([f"b{index:02d}" for index in range(CANDIDATE_N)])
    dense[0] = _hit("same", 1)
    bm25[0] = _hit("same", 1)
    fused = candidate_truncated_rrf(dense, bm25)
    assert [hit["retrieval_unit_id"] for hit in fused].count("same") == 1
    assert len({hit["retrieval_unit_id"] for hit in fused}) == len(fused)
    repeated = _list([f"d{index:02d}" for index in range(CANDIDATE_N)])
    repeated[1] = _hit(repeated[0]["retrieval_unit_id"], 2)
    with pytest.raises(UsageError, match="duplicate"):
        candidate_truncated_rrf(repeated, _list([f"b{index:02d}" for index in range(CANDIDATE_N)]))


def test_tie_break_is_deterministic_and_uses_retrieval_unit_id() -> None:
    dense = _list(["b", *[f"d{index:02d}" for index in range(CANDIDATE_N - 1)]])
    bm25 = _list(["a", *[f"s{index:02d}" for index in range(CANDIDATE_N - 1)]])
    first = candidate_truncated_rrf(dense, bm25)
    second = candidate_truncated_rrf(dense, bm25)
    assert first == second
    assert first[0]["retrieval_unit_id"] == "a"
    assert first[1]["retrieval_unit_id"] == "b"
    assert first[0]["score"] == pytest.approx(first[1]["score"])
    assert first[0]["score"] == pytest.approx(1 / (RRF_K + 1))


def test_truncated_rrf_differs_from_full_ranking_and_drops_the_deep_rank() -> None:
    rest = [f"u{index:02d}" for index in range(2, 25)]
    relevant = "u00"
    irrelevant = "u01"
    bm25_order = [relevant, *rest[:20], irrelevant, *rest[20:]]
    dense_order = [irrelevant, *rest, relevant]
    bm25 = _list(bm25_order)
    dense = _list(dense_order)
    assert {hit["chunk_id"] for hit in bm25} == {hit["chunk_id"] for hit in dense}
    full = reciprocal_rank_fusion(bm25, dense, expected_count=25)
    truncated = candidate_truncated_rrf(dense, bm25)
    full_ids = [hit["chunk_id"] for hit in full]
    trunc_ids = [hit["retrieval_unit_id"] for hit in truncated]
    assert full_ids.index(irrelevant) < full_ids.index(relevant)
    assert trunc_ids.index(relevant) < trunc_ids.index(irrelevant)
    relevant_row = next(hit for hit in truncated if hit["retrieval_unit_id"] == relevant)
    assert relevant_row["dense_rank"] is None
    assert relevant_row["dense_contribution"] == 0
    assert relevant_row["score"] == pytest.approx(1 / (RRF_K + 1))
    assert rest[20] not in trunc_ids
    audit = implementation_audit()
    assert audit["deep_tail_rank_used"] is False
    assert audit["k"] == 60
    assert audit["candidate_n"] == 20


def test_intersection_unit_outranks_a_single_list_rank_one() -> None:
    dense = _list([f"d{index:02d}" for index in range(CANDIDATE_N)])
    bm25 = _list([f"b{index:02d}" for index in range(CANDIDATE_N)])
    dense[0] = _hit("solo", 1)
    dense[-1] = _hit("both", CANDIDATE_N)
    bm25[-1] = _hit("both", CANDIDATE_N)
    fused = candidate_truncated_rrf(dense, bm25)
    both = next(hit for hit in fused if hit["retrieval_unit_id"] == "both")
    solo = next(hit for hit in fused if hit["retrieval_unit_id"] == "solo")
    assert both["bm25_rank"] == CANDIDATE_N
    assert both["dense_rank"] == CANDIDATE_N
    assert solo["dense_rank"] == 1
    assert solo["bm25_rank"] is None
    assert both["score"] == pytest.approx(intersection_min_score())
    assert solo["score"] == pytest.approx(single_list_max_score())
    assert both["rank"] < solo["rank"]
    assert intersection_min_score() > single_list_max_score()


def test_other_n_or_k_is_rejected() -> None:
    dense = _list([f"d{index:02d}" for index in range(CANDIDATE_N)])
    bm25 = _list([f"b{index:02d}" for index in range(CANDIDATE_N)])
    with pytest.raises(UsageError):
        candidate_truncated_rrf(dense, bm25, n=10)
    with pytest.raises(UsageError):
        candidate_truncated_rrf(dense, bm25, k=10)


def test_verdict_labels_follow_the_pre_registered_cutoffs() -> None:
    strong = _verdict(truncated_hit10=43, recovered=8, dense=True, both_lost=0)
    partial = _verdict(
        truncated_hit10=40,
        recovered=5,
        dense=True,
        both_lost=0,
        recall_drop=0.1,
        requirement_recall=46,
    )
    rejected = _verdict(truncated_hit10=38, recovered=2, dense=True, both_lost=0)
    assert strong["label"] == "A"
    assert partial["label"] == "B"
    assert rejected["label"] == "C"


def test_stored_dev_top20_copies_match_before_fusion() -> None:
    root = repo_root()
    bundle = validate_source_bundle(root)
    mismatches = ranking_mismatches(bundle, root / "evaluation" / "runs" / FULL_RUN_DIR)
    assert mismatches == []
    assert bundle["dense_sha256"]
    assert len(bundle["questions"]) == 45


def _verdict(**overrides: object) -> dict:
    values = {
        "truncated_hit10": 43,
        "full_hit10": 36,
        "recovered_of_eight": 8,
        "dense_only_retained": True,
        "both_hit_lost": 0,
        "prior_bm25_only_kept": 2,
        "prior_bm25_only_kept_still": 2,
        "bm25_mrr": 0.742,
        "truncated_mrr": 0.742,
        "bm25_ndcg": 0.708,
        "truncated_ndcg": 0.708,
        "bm25_recall5": 0.572,
        "truncated_recall5": 0.572,
        "truncated_requirement_recall": 51,
        "truncated_all_requirements": 41,
    }
    if "recall_drop" in overrides:
        drop = float(overrides.pop("recall_drop"))
        values["truncated_recall5"] = float(values["bm25_recall5"]) - drop
    if "requirement_recall" in overrides:
        values["truncated_requirement_recall"] = overrides.pop("requirement_recall")
    mapping = {
        "truncated_hit10": "truncated_hit10",
        "recovered": "recovered_of_eight",
        "dense": "dense_only_retained",
        "both_lost": "both_hit_lost",
    }
    for key, value in overrides.items():
        values[mapping.get(key, key)] = value
    return judge_truncated_hypothesis(**values)
