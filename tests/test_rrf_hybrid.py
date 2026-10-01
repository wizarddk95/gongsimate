"""Fixed-k RRF on the D2a retrieval-unit corpus. Oracle stays off the headline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from disclosure_agent.bm25 import Bm25Index
from disclosure_agent.errors import UsageError
from disclosure_agent.evaluation import load_dataset, q07_precision_contract, required_claims
from disclosure_agent.hybrid import (
    RRF_K,
    ZERO_SCORE_POLICY,
    claim_contribution,
    oracle_union,
    ranking_agreement,
    reciprocal_rank_fusion,
    reciprocal_rank_fusion_zero_tie,
)
from disclosure_agent.retrieval_unit import retrieval_units_sha256, scoring_chunks

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "datasets" / "isu_petasys_v0_3"
D2A = ROOT / "evaluation" / "runs" / "isu_petasys_v0_3_dense_d2a"
CANONICAL_BM25 = ROOT / "evaluation" / "runs" / "isu_petasys_v0_3"
BM25_RU = ROOT / "evaluation" / "runs" / "isu_petasys_v0_3_bm25_ru_d2a"
HYBRID = ROOT / "evaluation" / "runs" / "isu_petasys_v0_3_rrf_d2a"
D2A_SHA = "cff5e1a7d8572c6217a9fea2506581ca9b5ce6e12f60c5f9b2f32aaba4b2c9ff"
EMBEDDING_SHA = "244a5c82344cfd73c1119934cb35386f44395da9cf54eb127b15f03079efb340"
E186 = "78be6e94020c610d2411a83f2187ca58f098c8d9f071da7c4d78cb9db0ae5b62"


def _hit(
    chunk_id: str,
    rank: int,
    ordinal: int = 0,
    *,
    score: float | None = None,
    unit_index: int = 0,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "rank": rank,
        "ordinal": ordinal,
        "unit_index": unit_index,
        "score": (1.0 / rank) if score is None else score,
    }


def _paragraph_chunk(chunk_id: str, element_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "ordinal": 0,
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
        "question_id": "Q01",
        "required": True,
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


def _question_shell(success: bool, claims: list[tuple[str, str]]) -> dict:
    return {
        "question_id": "Q01",
        "in_positive_recall_denominator": True,
        "at_k": {
            "5": {
                "answer_evidence_success": success,
                "claims": [
                    {"claim_id": claim_id, "chunk_set": status} for claim_id, status in claims
                ],
            }
        },
    }


def test_rrf_uses_one_based_ranks_and_fixed_k() -> None:
    bm25 = [_hit("a", 1, 0), _hit("b", 2, 1), _hit("c", 3, 2)]
    dense = [_hit("b", 1, 1), _hit("c", 2, 2), _hit("a", 3, 0)]
    first = reciprocal_rank_fusion(bm25, dense, expected_count=3)
    second = reciprocal_rank_fusion(bm25, dense, expected_count=3)
    assert RRF_K == 60
    assert first == second
    assert [hit["rank"] for hit in first] == [1, 2, 3]
    assert [hit["chunk_id"] for hit in first] == ["b", "a", "c"]
    assert first[0]["score"] == pytest.approx((1.0 / 62) + (1.0 / 61))
    assert first[0]["bm25_rank"] == 2
    assert first[0]["dense_rank"] == 1
    tied = reciprocal_rank_fusion(
        [_hit("b", 1, 1), _hit("a", 2, 0)],
        [_hit("a", 1, 0), _hit("b", 2, 1)],
        expected_count=2,
    )
    assert tied[0]["chunk_id"] == "a"
    assert tied[0]["score"] == pytest.approx(tied[1]["score"])
    with pytest.raises(UsageError):
        reciprocal_rank_fusion(bm25, dense, k_rrf=10, expected_count=3)
    with pytest.raises(UsageError):
        reciprocal_rank_fusion(bm25[:2], dense, expected_count=3)
    with pytest.raises(UsageError):
        reciprocal_rank_fusion(
            [{"chunk_id": "a", "rank": 0, "ordinal": 0, "score": 0.0}, _hit("b", 1)],
            [_hit("a", 1), _hit("b", 2)],
        )


def test_bm25_rank_all_keeps_positive_order_and_is_deterministic() -> None:
    chunks = [
        {"chunk_id": "b", "ordinal": 1, "text": "연결 매출액"},
        {"chunk_id": "a", "ordinal": 0, "text": "연결 매출액"},
        {"chunk_id": "c", "ordinal": 2, "text": "시설투자"},
    ]
    index = Bm25Index(chunks)
    positive = index.search("매출액", index.n)
    full = index.rank_all("매출액")
    again = index.rank_all("매출액")
    assert full == again
    assert [hit["chunk_id"] for hit in full[: len(positive)]] == [
        hit["chunk_id"] for hit in positive
    ]
    assert [hit["rank"] for hit in full] == [1, 2, 3]
    assert full[-1]["chunk_id"] == "c"
    assert full[-1]["score"] == 0
    assert index.search("완전히다른질의xyz", 3) == []


def test_claim_contribution_keeps_sparse_and_dense_only_hits() -> None:
    bm25 = [_question_shell(True, [("Q01-C1", "full"), ("Q01-C2", "none")])]
    dense = [_question_shell(False, [("Q01-C1", "none"), ("Q01-C2", "full")])]
    hybrid = [_question_shell(True, [("Q01-C1", "full"), ("Q01-C2", "none")])]
    report = claim_contribution(bm25, dense, hybrid, ks=(5,))
    block = report["at_k"]["5"]
    assert report["not_a_headline_metric"] is True
    assert block["bm25_only_claims"] == [{"question_id": "Q01", "claim_id": "Q01-C1"}]
    assert block["dense_only_claims"] == [{"question_id": "Q01", "claim_id": "Q01-C2"}]
    assert block["hybrid_preserved_sparse_claims"] == [
        {"question_id": "Q01", "claim_id": "Q01-C1"}
    ]
    assert block["hybrid_regression_claims"] == [{"question_id": "Q01", "claim_id": "Q01-C2"}]
    assert block["question_hybrid_regression"] == []


def test_oracle_union_is_not_a_headline_retriever() -> None:
    claim = _claim("Q01-C1", "gold")
    dataset = {
        "questions": [
            {
                "question_id": "Q01",
                "text": "질문",
                "in_positive_recall_denominator": True,
                "answerability": "supported",
                "claim_ids": ["Q01-C1"],
            }
        ],
        "claims": {"Q01-C1": claim},
    }
    chunks = [_paragraph_chunk("bm25-only", "gold"), _paragraph_chunk("dense-only", "other")]
    bm25_hits = {"질문": [{"chunk_id": "bm25-only", "rank": 1}]}
    dense_hits = {"질문": [{"chunk_id": "dense-only", "rank": 1}]}
    report = oracle_union(dataset, chunks, bm25_hits, dense_hits, ks=(1,))
    assert report["not_a_headline_metric"] is True
    assert report["not_a_ranking_system"] is True
    assert report["at_k"]["1"]["claim_recall"] == {"numerator": 1, "denominator": 1}
    assert report["at_k"]["1"]["question_answer_evidence_satisfied"] == {
        "numerator": 1,
        "denominator": 1,
    }


def test_required_denominator_excludes_e186_and_keeps_q07_precision() -> None:
    dataset = load_dataset(DATASET)
    positive = [
        question
        for question in dataset["questions"]
        if question.get("in_positive_recall_denominator")
    ]
    claims = []
    for question in positive:
        owned = [
            claim
            for claim in dataset["claims"].values()
            if claim["question_id"] == question["question_id"]
        ]
        claims.extend(required_claims(owned))
    assert len(positive) == 8
    assert len(claims) == 27
    for claim in dataset["claims"].values():
        for span in claim["spans"]:
            assert span["element_id"] != E186
    q07 = next(item for item in dataset["questions"] if item["question_id"] == "Q07")
    contract = q07_precision_contract(q07)
    assert contract["reported_metric_precision_policy"]["delta_pp"] == "6.65"
    assert contract["raw_operand_precision_policy"]["delta_pp"] == "6.64"
    assert contract["reported_metric_precision_policy"]["fy2025"] == "18.82"
    assert contract["reported_metric_precision_policy"]["fy2024"] == "12.17"


@pytest.mark.skipif(not (CANONICAL_BM25 / "metrics.json").exists(), reason="locked BM25")
def test_historical_bm25_locked_headline_stays_canonical() -> None:
    metrics = json.loads((CANONICAL_BM25 / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["retriever_type"] == "bm25"
    assert metrics["tokenizer_version"] == "ko-mixed-v1"
    assert metrics["bm25"]["k1"] == 1.2
    assert metrics["bm25"]["b"] == 0.75
    assert metrics["chunks"]["chunk_count"] == 2803
    claim_5 = metrics["metrics"]["at_k"]["5"]["claim_set_recall"]
    question_5 = metrics["metrics"]["at_k"]["5"]["question_answer_evidence_satisfied"]
    claim_20 = metrics["metrics"]["at_k"]["20"]["claim_set_recall"]
    question_20 = metrics["metrics"]["at_k"]["20"]["question_answer_evidence_satisfied"]
    assert (claim_5["numerator"], claim_5["denominator"]) == (5, 27)
    assert (question_5["numerator"], question_5["denominator"]) == (1, 8)
    assert (claim_20["numerator"], claim_20["denominator"]) == (19, 27)
    assert (question_20["numerator"], question_20["denominator"]) == (3, 8)


@pytest.mark.skipif(not (D2A / "metrics.json").exists(), reason="locked D2a")
def test_dense_d2a_locked_headline_and_candidate_hash() -> None:
    metrics = json.loads((D2A / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["dense"]["embedding_model_revision"] == (
        "5617a9f61b028005a4858fdac845db406aefb181"
    )
    assert metrics["retrieval_units"]["artifact_sha256"] == D2A_SHA
    assert metrics["dense"]["embedding_artifact_sha256"] == EMBEDDING_SHA
    expected = {
        "1": (6, 0),
        "3": (9, 0),
        "5": (15, 2),
        "10": (20, 3),
        "20": (25, 6),
    }
    for k, (claim_numerator, question_numerator) in expected.items():
        block = metrics["metrics"]["at_k"][k]
        claim = block["claim_set_recall"]
        question = block["question_answer_evidence_satisfied"]
        assert (claim["numerator"], claim["denominator"]) == (claim_numerator, 27)
        assert (question["numerator"], question["denominator"]) == (question_numerator, 8)
    units = [
        json.loads(line)
        for line in (D2A / "retrieval_units.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(units) == 2869
    assert retrieval_units_sha256(units) == D2A_SHA
    raw = (D2A / "embeddings.f32").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EMBEDDING_SHA
    assert len(raw) == 2869 * 1024 * 4
    view = scoring_chunks(units)
    index = Bm25Index(view)
    assert [chunk["chunk_id"] for chunk in index.chunks] == [
        unit["retrieval_unit_id"] for unit in units
    ]
    assert all(unit["text"] == chunk["text"] for unit, chunk in zip(units, view, strict=True))
    first = index.search(units[0]["text"][:80], 5)
    second = index.search(units[0]["text"][:80], 5)
    assert first == second
    snapshot = [
        (
            unit["retrieval_unit_id"],
            unit["canonical_chunk_id"],
            unit["unit_index"],
            unit["representation_version"],
            unit["source_refs"],
        )
        for unit in units
    ]
    assert snapshot[0][0] == units[0]["retrieval_unit_id"]


def test_ranking_agreement_detects_a_top20_change() -> None:
    left = [
        {
            "question_id": "Q01",
            "at_k": {"1": {"claims": [], "answer_evidence_success": False},
                     "3": {"claims": [], "answer_evidence_success": False},
                     "5": {"claims": [], "answer_evidence_success": False},
                     "10": {"claims": [], "answer_evidence_success": False},
                     "20": {"claims": [], "answer_evidence_success": False}},
            "evidence_classification": {"first_complete_evidence_path_rank": None},
            "diagnosis": {"claims": []},
            "top20": [{"chunk_id": "a"}],
        }
    ]
    right = json.loads(json.dumps(left))
    right[0]["top20"] = [{"chunk_id": "b"}]
    assert ranking_agreement(left, left)["match"] is True
    assert ranking_agreement(left, right)["match"] is False


@pytest.mark.skipif(not (HYBRID / "metrics.json").exists(), reason="hybrid run")
def test_hybrid_run_keeps_identity_oracle_and_e186_contract() -> None:
    hybrid = json.loads((HYBRID / "metrics.json").read_text(encoding="utf-8"))
    bm25 = json.loads((BM25_RU / "metrics.json").read_text(encoding="utf-8"))
    identity = hybrid["candidate_identity"]
    assert identity["bm25_ids_equal_d2a"] is True
    assert identity["dense_ids_equal_d2a"] is True
    assert identity["topology_mutated"] is False
    assert identity["same_candidate_universe"] is True
    assert identity["retrieval_unit_count"] == 2869
    assert identity["retrieval_unit_artifact_sha256"] == D2A_SHA
    assert hybrid["rrf"]["k_rrf"] == 60
    assert hybrid["rrf"]["rank_base"] == 1
    assert hybrid["rrf"]["full_ranking_fusion"] is True
    assert hybrid["rrf"]["candidate_cutoff"] is None
    assert hybrid["rrf"]["weighted_score_fusion"] is False
    assert hybrid["dense_reproduction"]["document_reembedded"] is False
    assert hybrid["dense_reproduction"]["match"] is True
    assert hybrid["metrics"]["headline_metrics"] == [
        "claim_set_recall",
        "question_answer_evidence_satisfied",
    ]
    oracle = hybrid["comparison"]["oracle_union"]
    assert oracle["not_a_headline_metric"] is True
    assert oracle["not_a_ranking_system"] is True
    assert "oracle_union" not in hybrid["metrics"]
    assert hybrid["metrics"]["at_k"]["5"]["claim_set_recall"]["denominator"] == 27
    assert hybrid["metrics"]["positive_question_count"] == 8
    q07 = hybrid["comparison"]["q07"]
    for system in q07.values():
        assert system["E186_counted_as_positive_evidence"] is False
        assert system["E186_in_positive_claim_recall"] is False
        assert system["E186_evidence_class"] == "relevant_conflicting"
    contract = hybrid["calculation_q07"]["precision_contract"]
    assert contract["reported_metric_precision_policy"]["delta_pp"] == "6.65"
    assert contract["raw_operand_precision_policy"]["delta_pp"] == "6.64"
    diversity = hybrid["comparison"]["diversity"]["hybrid_rrf"]["positive_summary"]["5"]
    assert diversity["not_a_headline_metric"] is True
    assert "max_same_canonical_occupancy" in diversity
    assert bm25["retriever_type"] == "bm25_ru"
    assert bm25["bm25_canonical_locked"]["not_this_run"] is True
    assert bm25["bm25_canonical_locked"]["headline"]["5"]["claim_recall"]["numerator"] == 5
    assert bm25["tokenizer_version"] == "ko-mixed-v1"
    assert bm25["bm25"]["k1"] == 1.2
    assert bm25["bm25"]["b"] == 0.75
    assert bm25["retrieval_units"]["artifact_sha256"] == D2A_SHA


def _by_id(hits: list[dict]) -> dict[str, dict]:
    return {hit["chunk_id"]: hit for hit in hits}


def test_rrf_zero_tie_gives_zero_score_candidates_no_sparse_contribution() -> None:
    bm25 = [
        _hit("a", 1, 0, score=3.0),
        _hit("b", 2, 1, score=0.0),
        _hit("c", 3, 2, score=0.0),
    ]
    dense = [_hit("b", 1, 1), _hit("c", 2, 2), _hit("a", 3, 0)]
    swapped = [
        _hit("a", 1, 0, score=3.0),
        _hit("c", 2, 5, score=0.0),
        _hit("b", 3, 9, score=0.0),
    ]
    original = reciprocal_rank_fusion(bm25, dense, expected_count=3)
    original_swapped = reciprocal_rank_fusion(swapped, dense, expected_count=3)
    zero = reciprocal_rank_fusion_zero_tie(bm25, dense, expected_count=3)
    zero_swapped = reciprocal_rank_fusion_zero_tie(swapped, dense, expected_count=3)
    original_b = _by_id(original)["b"]["score"]
    swapped_b = _by_id(original_swapped)["b"]["score"]
    assert original_b != pytest.approx(swapped_b)
    scored = _by_id(zero)
    rescored = _by_id(zero_swapped)
    assert scored["b"]["bm25_rrf_contribution"] == 0
    assert scored["c"]["bm25_rrf_contribution"] == 0
    assert scored["b"]["score"] == pytest.approx(1.0 / 61)
    assert scored["c"]["score"] == pytest.approx(1.0 / 62)
    assert scored["b"]["score"] - scored["c"]["score"] == pytest.approx((1.0 / 61) - (1.0 / 62))
    assert scored["b"]["score"] == pytest.approx(rescored["b"]["score"])
    assert scored["c"]["score"] == pytest.approx(rescored["c"]["score"])
    assert scored["a"]["score"] == pytest.approx(_by_id(original)["a"]["score"])
    assert scored["a"]["bm25_rank"] == 1
    assert scored["a"]["bm25_rrf_contribution"] == pytest.approx(1.0 / 61)
    assert ZERO_SCORE_POLICY == "bm25_zero_score_rrf_contribution = 0"
    assert RRF_K == 60
    with pytest.raises(UsageError):
        reciprocal_rank_fusion_zero_tie(bm25, dense, k_rrf=10, expected_count=3)


def test_rrf_zero_tie_matches_original_when_every_bm25_score_is_positive() -> None:
    bm25 = [
        _hit("a", 1, 0, score=2.0),
        _hit("b", 2, 1, score=1.0),
        _hit("c", 3, 2, score=0.5),
    ]
    dense = [_hit("c", 1, 2), _hit("a", 2, 0), _hit("b", 3, 1)]
    original = reciprocal_rank_fusion(bm25, dense, expected_count=3)
    zero = reciprocal_rank_fusion_zero_tie(bm25, dense, expected_count=3)
    assert [hit["chunk_id"] for hit in zero] == [hit["chunk_id"] for hit in original]
    assert [hit["score"] for hit in zero] == [hit["score"] for hit in original]
    assert [hit["bm25_rank"] for hit in zero] == [hit["bm25_rank"] for hit in original]


def test_rrf_zero_tie_keeps_the_full_candidate_universe() -> None:
    count = 2869
    bm25 = []
    dense = []
    for index in range(count):
        doc_id = f"u{index:04d}"
        bm25.append(_hit(doc_id, index + 1, index, score=1.0 if index == 0 else 0.0))
        dense.append(_hit(doc_id, count - index, index))
    fused = reciprocal_rank_fusion_zero_tie(bm25, dense, expected_count=count)
    assert RRF_K == 60
    assert len(fused) == count
    assert {hit["chunk_id"] for hit in fused} == {hit["chunk_id"] for hit in bm25}
    zeros = [hit for hit in fused if hit["bm25_score"] == 0]
    positive = [hit for hit in fused if hit["bm25_score"] > 0]
    assert len(zeros) == count - 1
    assert len(positive) == 1
    assert positive[0]["bm25_rank"] == 1
    assert all(hit["bm25_rrf_contribution"] == 0 for hit in zeros)
    with pytest.raises(UsageError):
        reciprocal_rank_fusion_zero_tie(bm25, dense, expected_count=count - 1)
