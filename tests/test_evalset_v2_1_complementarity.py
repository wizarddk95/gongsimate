"""DEV complementarity helpers. These tests do not load an embedding model."""

from __future__ import annotations

import json

from disclosure_agent.bm25 import Bm25Index
from disclosure_agent.evalset_v2.spec import repo_root
from disclosure_agent.evalset_v2_1.complementarity import (
    CELL_BM25,
    CELL_BOTH,
    CELL_DENSE,
    CELL_MISS,
    assert_dev_population,
    bm25_chunks,
    crossed_query_ids,
    decide_hybrid,
    dev_fully_answerable,
    metric_hits,
    query_cell,
    ranking_duplicates,
    requirement_complementarity,
)
from disclosure_agent.evalset_v2_1.metrics import score_question
from disclosure_agent.retrieval_unit import D2A_VERSION


def test_dev_population_drops_test_partial_and_unanswerable() -> None:
    questions = [
        {"query_id": "dev", "answerability": "fully_answerable", "split": "dev"},
        {"query_id": "test", "answerability": "fully_answerable", "split": "test"},
        {"query_id": "partial", "answerability": "partially_answerable", "split": "dev"},
        {"query_id": "none", "answerability": "unanswerable", "split": "dev"},
    ]
    assert [item["query_id"] for item in dev_fully_answerable(questions)] == ["dev"]
    assert [item["query_id"] for item in assert_dev_population(questions, expected_count=1)] == [
        "dev"
    ]


def test_frozen_v21_dev_is_45_fully_answerable() -> None:
    root = repo_root()
    directory = root / "evaluation" / "datasets" / "multi_company_retrieval_v2_1"
    frozen = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
    questions = json.loads((directory / "questions.json").read_text(encoding="utf-8"))["questions"]
    kept = assert_dev_population(questions, expected_count=int(frozen["dev_size"]))
    assert len(kept) == 45
    assert {item["split"] for item in kept} == {"dev"}
    assert {item["answerability"] for item in kept} == {"fully_answerable"}


def test_query_cells_cover_the_four_patterns() -> None:
    assert query_cell(True, True) == CELL_BOTH
    assert query_cell(True, False) == CELL_DENSE
    assert query_cell(False, True) == CELL_BM25
    assert query_cell(False, False) == CELL_MISS


def test_requirement_cross_is_not_visible_from_query_hit_alone() -> None:
    requirements = [
        {"query_id": "q", "requirement_id": "R1", "answerability": "fully_answerable", "text": "a"},
        {"query_id": "q", "requirement_id": "R2", "answerability": "fully_answerable", "text": "b"},
    ]
    qrels = [
        {"query_id": "q", "requirement_id": "R1", "retrieval_unit_id": "ua", "relevance": 2},
        {"query_id": "q", "requirement_id": "R2", "retrieval_unit_id": "ub", "relevance": 2},
    ]
    dense_hits = [{"rank": 1, "retrieval_unit_id": "ua", "source_element_ids": []}]
    bm25_hits = [{"rank": 1, "retrieval_unit_id": "ub", "source_element_ids": []}]
    rows = requirement_complementarity("q", requirements, qrels, dense_hits, bm25_hits, k=10)
    assert [row["cell"] for row in rows] == ["dense_only", "bm25_only"]
    assert crossed_query_ids(rows) == ["q"]
    assert query_cell(True, True) == CELL_BOTH


def test_hybrid_rule_is_b_when_bm25_adds_nothing_and_c_when_misses_dominate() -> None:
    nested = decide_hybrid(
        {CELL_BOTH: 30, CELL_DENSE: 10, CELL_BM25: 0, CELL_MISS: 5},
        {"bm25_only": 0, "dense_only": 4},
        crossed_queries=0,
    )
    assert nested["primary"] == "B"
    shared = decide_hybrid(
        {CELL_BOTH: 10, CELL_DENSE: 2, CELL_BM25: 2, CELL_MISS: 31},
        {"bm25_only": 2, "dense_only": 2},
        crossed_queries=0,
    )
    assert shared["primary"] == "C"
    complement = decide_hybrid(
        {CELL_BOTH: 20, CELL_DENSE: 8, CELL_BM25: 8, CELL_MISS: 9},
        {"bm25_only": 3, "dense_only": 3},
        crossed_queries=1,
    )
    assert complement["primary"] == "A"
    assert complement["test_used"] is False


def test_duplicate_ranks_stay_visible() -> None:
    hits = [
        {"rank": 1, "retrieval_unit_id": "u", "source_element_ids": ["e"]},
        {"rank": 2, "retrieval_unit_id": "u", "source_element_ids": ["e"]},
    ]
    report = ranking_duplicates([{"query_id": "q", "hits": hits}])
    assert report["questions_with_duplicates"] == 1
    assert report["duplicate_slots"] == 1
    question = {
        "query_id": "q",
        "answerability": "fully_answerable",
        "split": "dev",
        "company_name": "c",
        "query_type": "single_fact",
        "evidence_form": "paragraph",
        "wording_style": "original_term",
        "period_sensitive": False,
        "stratum_id": "s",
    }
    requirements = [
        {"query_id": "q", "requirement_id": "R1", "answerability": "fully_answerable"}
    ]
    qrels = [
        {
            "query_id": "q",
            "requirement_id": "R1",
            "retrieval_unit_id": "u",
            "relevance": 1,
            "judgment_status": "judged",
        }
    ]
    scored = score_question(question, qrels, requirements, hits)
    assert [hit["retrieval_unit_id"] for hit in scored["hits"]] == ["u", "u"]
    assert scored["at_k"]["1"]["hit"] is True


def test_bm25_hit_is_graded_by_retrieval_unit_not_source_element() -> None:
    chunks = [
        {
            "chunk_id": "unit-relevant",
            "ordinal": 0,
            "text": "매출액 100",
            "retrieval_unit_id": "unit-relevant",
            "unit_index": 0,
            "source_element_ids": ["element-aaa"],
        },
        {
            "chunk_id": "unit-other",
            "ordinal": 1,
            "text": "기타 설명",
            "retrieval_unit_id": "unit-other",
            "unit_index": 0,
            "source_element_ids": ["element-bbb"],
        },
    ]
    raw = Bm25Index(chunks).search("매출액", 5)
    hits = metric_hits(raw, {chunk["retrieval_unit_id"]: chunk for chunk in chunks})
    assert hits[0]["retrieval_unit_id"] == "unit-relevant"
    assert hits[0]["retrieval_unit_id"] != "element-aaa"
    assert hits[0]["source_element_ids"] == ["element-aaa"]
    question = {
        "query_id": "q",
        "answerability": "fully_answerable",
        "split": "dev",
        "company_name": "c",
        "query_type": "single_fact",
        "evidence_form": "paragraph",
        "wording_style": "original_term",
        "period_sensitive": False,
        "stratum_id": "s",
    }
    requirements = [
        {"query_id": "q", "requirement_id": "R1", "answerability": "fully_answerable"}
    ]
    unit_qrel = [
        {
            "query_id": "q",
            "requirement_id": "R1",
            "retrieval_unit_id": "unit-relevant",
            "relevance": 2,
            "judgment_status": "judged",
        }
    ]
    element_qrel = [
        {
            "query_id": "q",
            "requirement_id": "R1",
            "retrieval_unit_id": "element-aaa",
            "relevance": 2,
            "judgment_status": "judged",
        }
    ]
    assert score_question(question, unit_qrel, requirements, hits)["at_k"]["1"]["hit"] is True
    assert score_question(question, element_qrel, requirements, hits)["at_k"]["1"]["hit"] is False


def test_bm25_chunks_follow_the_eligible_unit_ids() -> None:
    units = [
        {
            "retrieval_unit_id": "u1",
            "text": "본문",
            "representation_version": D2A_VERSION,
        }
    ]
    scoring = [
        {
            "chunk_id": "u1",
            "ordinal": 3,
            "text": "본문",
            "retrieval_unit_id": "u1",
            "unit_index": 0,
        }
    ]
    chunks = bm25_chunks(units, scoring)
    assert chunks[0]["chunk_id"] == "u1"
    assert chunks[0]["retrieval_unit_id"] == "u1"
    assert chunks[0]["text"] == "본문"
