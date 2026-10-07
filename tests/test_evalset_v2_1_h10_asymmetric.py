"""H10 BM25-primary Dense-head insertion. These tests do not load an embedding model."""

from __future__ import annotations

import pytest

from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2.spec import repo_root
from disclosure_agent.evalset_v2_1.asymmetric_fusion import (
    CANDIDATE_N,
    MARGINAL_SLOT,
    PROTECTED_PREFIX,
    RANKING_DROP_LIMIT,
    REQUIRED_SEMANTICS_TOKENS,
    SEMANTICS_RELATIVE,
    STRONG_ALL_REQUIREMENTS_FLOOR,
    STRONG_BM25_ONLY_LOST_MAX,
    STRONG_HIT10_FLOOR,
    STRONG_REQUIREMENT_RECALL_FLOOR,
    TRUNCATED_BM25_ONLY_LOSSES,
    TRUNCATED_HIT10,
    bm25_primary_dense_head,
    implementation_audit,
    judge_h10,
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


def _ids(fused: dict) -> list[str]:
    return [hit["retrieval_unit_id"] for hit in fused["hits"]]


def test_semantics_document_locks_the_constants() -> None:
    text = (repo_root() / SEMANTICS_RELATIVE).read_text(encoding="utf-8")
    for token in REQUIRED_SEMANTICS_TOKENS:
        assert token in text
    expected = {
        "STRONG_HIT10_FLOOR": STRONG_HIT10_FLOOR,
        "STRONG_BM25_ONLY_LOST_MAX": STRONG_BM25_ONLY_LOST_MAX,
        "RANKING_DROP_LIMIT": RANKING_DROP_LIMIT,
        "STRONG_REQUIREMENT_RECALL_FLOOR": STRONG_REQUIREMENT_RECALL_FLOOR,
        "STRONG_ALL_REQUIREMENTS_FLOOR": STRONG_ALL_REQUIREMENTS_FLOOR,
        "TRUNCATED_HIT10": TRUNCATED_HIT10,
        "TRUNCATED_BM25_ONLY_LOSSES": TRUNCATED_BM25_ONLY_LOSSES,
        "PROTECTED_PREFIX": PROTECTED_PREFIX,
        "MARGINAL_SLOT": MARGINAL_SLOT,
        "CANDIDATE_N": CANDIDATE_N,
    }
    for name, value in expected.items():
        assert f"{name} = {value}" in text


def test_dense_head_outside_bm25_occupies_only_the_marginal_slot() -> None:
    bm25 = [f"b{index:02d}" for index in range(1, CANDIDATE_N + 1)]
    dense = ["semantic", *[f"d{index:02d}" for index in range(2, CANDIDATE_N + 1)]]
    fused = bm25_primary_dense_head(_list(dense), _list(bm25))
    ids = _ids(fused)
    assert fused["edit"] == "insertion"
    assert ids[:PROTECTED_PREFIX] == bm25[:PROTECTED_PREFIX]
    assert ids[MARGINAL_SLOT - 1] == "semantic"
    assert ids[MARGINAL_SLOT:] == bm25[PROTECTED_PREFIX:]
    assert "d02" not in ids
    assert fused["hits"][MARGINAL_SLOT - 1]["bm25_rank"] is None
    assert fused["hits"][MARGINAL_SLOT - 1]["is_dense_head"] is True


def test_dense_only_competitors_do_not_pass_a_bm25_rank() -> None:
    bm25 = [f"b{index:02d}" for index in range(1, CANDIDATE_N + 1)]
    bm25[7] = "lexical"
    dense = [f"d{index:02d}" for index in range(1, CANDIDATE_N + 1)]
    fused = bm25_primary_dense_head(_list(dense), _list(bm25))
    ids = _ids(fused)
    assert ids[7] == "lexical"
    assert ids.index("lexical") < ids.index("d01")
    assert "d02" not in ids


def test_intersection_membership_does_not_jump_ahead_of_bm25() -> None:
    bm25 = [f"b{index:02d}" for index in range(1, CANDIDATE_N + 1)]
    bm25[3] = "lexical"
    bm25[-1] = "both"
    dense = ["head", "both", *[f"d{index:02d}" for index in range(3, CANDIDATE_N + 1)]]
    fused = bm25_primary_dense_head(_list(dense), _list(bm25))
    ids = _ids(fused)
    assert ids[3] == "lexical"
    assert ids[MARGINAL_SLOT - 1] == "head"
    assert ids.index("both") == CANDIDATE_N
    assert ids.index("both") > ids.index("lexical")


def test_dense_head_already_in_the_bm25_top10_leaves_the_order_unchanged() -> None:
    bm25 = [f"b{index:02d}" for index in range(1, CANDIDATE_N + 1)]
    dense = ["b03", *[f"d{index:02d}" for index in range(2, CANDIDATE_N + 1)]]
    fused = bm25_primary_dense_head(_list(dense), _list(bm25))
    assert fused["edit"] == "none"
    assert _ids(fused) == bm25
    assert all(hit["edit"] == "none" for hit in fused["hits"])


def test_dense_head_inside_the_bm25_tail_is_promoted_only_to_rank_10() -> None:
    bm25 = [f"b{index:02d}" for index in range(1, CANDIDATE_N + 1)]
    bm25[14] = "head"
    dense = ["head", *[f"d{index:02d}" for index in range(2, CANDIDATE_N + 1)]]
    fused = bm25_primary_dense_head(_list(dense), _list(bm25))
    ids = _ids(fused)
    assert fused["edit"] == "promotion"
    assert len(ids) == CANDIDATE_N
    assert ids[:PROTECTED_PREFIX] == [f"b{index:02d}" for index in range(1, PROTECTED_PREFIX + 1)]
    assert ids[9] == "head"
    assert ids[10:15] == [f"b{index:02d}" for index in range(10, 15)]
    assert ids[15:] == [f"b{index:02d}" for index in range(16, CANDIDATE_N + 1)]


def test_disagreed_source_elements_and_other_cutoffs_are_rejected() -> None:
    bm25 = _list([f"b{index:02d}" for index in range(1, CANDIDATE_N + 1)])
    dense = _list(["b01", *[f"d{index:02d}" for index in range(2, CANDIDATE_N + 1)]])
    dense[0]["source_element_ids"] = ["other"]
    with pytest.raises(UsageError, match="source_element_ids"):
        bm25_primary_dense_head(dense, bm25)
    dense = _list([f"d{index:02d}" for index in range(1, CANDIDATE_N + 1)])
    with pytest.raises(UsageError):
        bm25_primary_dense_head(dense, bm25, n=10)
    with pytest.raises(UsageError):
        bm25_primary_dense_head(dense, bm25, hit_k=5)


def test_duplicate_dense_id_is_rejected() -> None:
    dense = _list([f"d{index:02d}" for index in range(1, CANDIDATE_N + 1)])
    dense[1] = _hit(dense[0]["retrieval_unit_id"], 2)
    bm25 = _list([f"b{index:02d}" for index in range(1, CANDIDATE_N + 1)])
    with pytest.raises(UsageError, match="duplicate"):
        bm25_primary_dense_head(dense, bm25)


def test_audit_has_no_weight_k_or_sweep() -> None:
    audit = implementation_audit()
    assert audit["rule_id"] == "bm25_primary_dense_head"
    assert audit["dense_ranks_read_by_placement"] == [1]
    assert audit["dense_ranks_2_to_20_are_placement_inputs"] is False
    assert audit["intersection_bonus"] is False
    assert audit["score_fusion"] is False
    assert audit["weights"] is None
    assert audit["k"] is None
    assert audit["parameter_sweep"] is False
    assert audit["test_search"] is False


def test_verdict_tree_follows_the_pre_registered_floors() -> None:
    strong = _verdict()
    assert strong["label"] == "Strong Support"
    assert strong["reason"] == "all_strong_floors"

    partial = _verdict(fused_hit10=41, bm25_only_lost=2)
    assert partial["label"] == "Partial Support"
    assert partial["reason"] == "lexical_regression_remains"

    quality = _verdict(recall10_drop=0.06)
    assert quality["label"] == "Partial Support"
    assert quality["reason"] == "strong_floor_missed"

    missed = _verdict(dense_only_retained=False, fused_hit10=43, bm25_only_lost=0)
    assert missed["label"] == "Reject"
    assert missed["reason"] == "dense_only_lost"

    tied = _verdict(fused_hit10=40, bm25_only_lost=0, recall10_drop=0.0)
    assert tied["label"] == "Reject"
    assert tied["reason"] == "not_clearly_better_than_truncated"

    same_losses = _verdict(fused_hit10=42, bm25_only_lost=3)
    assert same_losses["label"] == "Reject"
    assert same_losses["reason"] == "not_clearly_better_than_truncated"


def _verdict(**overrides: object) -> dict:
    payload = {
        "fused_hit10": 43,
        "bm25_only_lost": 0,
        "dense_only_retained": True,
        "recall5_drop": 0.0,
        "recall10_drop": 0.0,
        "mrr_drop": 0.0,
        "ndcg_drop": 0.0,
        "requirement_recall": 51,
        "all_requirements": 41,
    }
    payload.update(overrides)
    return judge_h10(**payload)  # type: ignore[arg-type]
