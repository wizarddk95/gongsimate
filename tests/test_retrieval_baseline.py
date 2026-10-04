"""Graded retrieval metrics for the multi-company Dense-D2a baseline.

These tests do not load the embedding model or the product index.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from disclosure_agent.errors import UsageError
from disclosure_agent.retrieval_baseline import (
    DATASET_ID,
    aggregate,
    load_baseline_dataset,
    score_question,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "datasets" / DATASET_ID


def _question() -> dict:
    return {
        "question_id": "Q",
        "company_name": "예시",
        "corp_code": "00000000",
        "query_type": "business",
        "wording": "exact",
        "evidence_form": "paragraph",
        "period_sensitive": False,
        "subset": "core",
    }


def _hit(rank: int, element_ids: list[str]) -> dict:
    return {
        "rank": rank,
        "score": 1.0 / rank,
        "retrieval_unit_id": f"u{rank}",
        "receipt_no": "20260101000001",
        "document_name": "사업보고서",
        "filing_period": "2025.12",
        "section_path": ["II. 사업의 내용"],
        "source_element_ids": element_ids,
        "text": "본문",
    }


def test_supporting_evidence_is_a_hit_without_the_preferred_filing() -> None:
    evidence = [
        {"evidence_id": "E2", "element_id": "direct", "relevance": 2, "preferred": True},
        {"evidence_id": "E1", "element_id": "support", "relevance": 1, "preferred": False},
    ]
    row = score_question(_question(), evidence, [_hit(1, ["support"])])
    assert row["at_k"]["1"]["hit"] is True
    assert row["at_k"]["1"]["preferred_hit"] is False
    assert row["mrr"] == 1.0
    assert row["first_direct_rank"] is None
    assert row["at_k"]["1"]["recall"]["value"] == 0.5


def test_duplicate_element_does_not_increase_recall_or_ideal_gain() -> None:
    evidence = [
        {"evidence_id": "E2", "element_id": "direct", "relevance": 2, "preferred": True},
        {"evidence_id": "E1", "element_id": "support", "relevance": 1, "preferred": False},
    ]
    hits = [
        _hit(1, ["other"]),
        _hit(2, ["support"]),
        _hit(3, ["direct"]),
        _hit(4, ["direct"]),
    ]
    row = score_question(_question(), evidence, hits)
    assert row["at_k"]["1"]["hit"] is False
    assert row["at_k"]["5"]["hit"] is True
    assert row["at_k"]["5"]["recall"]["value"] == 1.0
    assert row["mrr"] == 0.5
    assert row["first_preferred_rank"] == 3
    assert row["duplicate_relevant_hits_at_10"] == 1
    assert row["at_k"]["10"]["ndcg"] == pytest.approx(0.58688267143572)


def test_relevance_zero_does_not_count() -> None:
    evidence = [
        {"evidence_id": "E0", "element_id": "noise", "relevance": 0, "preferred": False},
        {"evidence_id": "E2", "element_id": "direct", "relevance": 2, "preferred": True},
    ]
    row = score_question(_question(), evidence, [_hit(1, ["noise"])])
    assert row["at_k"]["10"]["hit"] is False
    assert row["mrr"] == 0.0


def test_dataset_loads_and_covers_five_companies() -> None:
    dataset = load_baseline_dataset(DATASET)
    companies = {item["corp_code"] for item in dataset["questions"]}
    assert companies == {"00107613", "00126380", "00126955", "00989619", "00121941"}
    core = [item for item in dataset["questions"] if item["subset"] == "core"]
    assert 30 <= len(core) <= 50
    assert any(item["subset"] == "filing_policy" for item in dataset["questions"])
    assert any(item.get("reused_from") for item in dataset["questions"])


def test_dataset_rejects_a_question_without_positive_evidence(tmp_path: Path) -> None:
    snapshot = {
        "dataset_id": DATASET_ID,
        "schema_version": 1,
    }
    questions = {
        "questions": [
            {
                "question_id": "Q",
                "subset": "core",
                "corp_code": "00107613",
            }
        ]
    }
    evidence = {
        "evidence": [
            {
                "evidence_id": "Q-E0",
                "question_id": "Q",
                "relevance": 0,
                "element_id": "e",
                "receipt_no": "20260101000001",
                "quote": "없음",
            }
        ]
    }
    (tmp_path / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (tmp_path / "questions.json").write_text(json.dumps(questions), encoding="utf-8")
    (tmp_path / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(UsageError, match="relevance >= 1"):
        load_baseline_dataset(tmp_path)


def test_aggregate_keeps_counts_and_marks_small_groups() -> None:
    evidence = [
        {"evidence_id": "E2", "element_id": "direct", "relevance": 2, "preferred": True}
    ]
    hit = score_question(_question(), evidence, [_hit(1, ["direct"])])
    miss = score_question(_question(), evidence, [_hit(1, ["other"])])
    block = aggregate([hit, miss], population="example")
    assert block["question_count"] == 2
    assert block["small_n"] is True
    assert block["at_k"]["1"]["hit"]["numerator"] == 1
    assert block["at_k"]["1"]["hit"]["denominator"] == 2
    assert block["mrr"]["value"] == 0.5
