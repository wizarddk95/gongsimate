"""Retrieval eval set v2: quotas, labels, pooling, and incomplete-qrel metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from disclosure_agent.chunking import canonical_dumps, sha256_text
from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2.corpus_sample import (
    build_sampling_manifest,
    manifest_is_retrieval_free,
)
from disclosure_agent.evalset_v2.generate import (
    _normalize_decomposition,
    copied_source_sentence,
    logged_complete,
)
from disclosure_agent.evalset_v2.judge import judge_candidate
from disclosure_agent.evalset_v2.llm import CallLog, coerce_judgment, judge_payload, judgments_agree
from disclosure_agent.evalset_v2.metrics import (
    build_qrels,
    filter_headline,
    qrel_state,
    score_question,
)
from disclosure_agent.evalset_v2.pool import select_candidates
from disclosure_agent.evalset_v2.schemas import Decomposition, PassageJudgment
from disclosure_agent.evalset_v2.spec import load_spec, validate_spec

ROOT = Path(__file__).resolve().parents[1]
POOLING = {
    "dense_depth": 2,
    "bm25_depth": 2,
    "unanswerable_dense_depth": 1,
    "unanswerable_bm25_depth": 1,
    "unanswerable_tail": False,
    "tail_random_seed": 7,
    "tail_buckets": [
        {"name": "21-50", "start": 21, "end": 50, "n": 1},
        {"name": "51-200", "start": 51, "end": 200, "n": 1},
        {"name": "201+", "start": 201, "end": None, "n": 1},
    ],
}


def test_population_spec_is_not_a_cartesian_product() -> None:
    spec = load_spec(ROOT)
    validate_spec(spec)
    broken = json.loads(json.dumps(spec))
    broken["human_review"] = True
    with pytest.raises(UsageError):
        validate_spec(broken)


def test_sampling_is_deterministic_and_retrieval_free() -> None:
    spec = load_spec(ROOT)
    units = {
        company["corp_code"]: _fake_units(company["corp_code"])
        for company in spec["companies"]
    }
    first = build_sampling_manifest(units, spec)
    second = build_sampling_manifest(units, spec)
    assert [slot["query_id"] for slot in first["slots"]] == [
        slot["query_id"] for slot in second["slots"]]
    first_seed = first["slots"][0]["seeds"][0]["retrieval_unit_id"]
    second_seed = second["slots"][0]["seeds"][0]["retrieval_unit_id"]
    assert first_seed == second_seed
    assert manifest_is_retrieval_free(first)
    assert len(first["slots"]) == 120
    assert sum(slot["answerability"] == "unanswerable" for slot in first["slots"]) == 10


def test_structured_judgment_and_decomposition_parse() -> None:
    parsed = PassageJudgment.model_validate(
        {"requirement_judgments": [
            {"requirement_id": "R1", "relevance": 2}], "passage_relevance": 0}
    )
    coerced = coerce_judgment(parsed.model_dump(), ["R1"])
    assert coerced is not None
    assert coerced["passage_relevance"] == 2
    assert coerced["passage_relevance_repaired"] is True
    assert coerce_judgment(
        {"requirement_judgments": [{"requirement_id": "R9", "relevance": 1}]}, ["R1"]) is None
    decomposition = Decomposition.model_validate(
        {"requirements": [{"requirement_id": "R1", "text": "매출", "evidence_labels": ["E1"]}]}
    )
    rows = _normalize_decomposition(
        {"query_id": "Q"},
        {"seeds": [{"element_ids": ["e1"]}]},
        decomposition.model_dump(),
        ["E1"],
    )
    assert rows[0]["seed_evidence_ids"] == ["e1"]
    assert rows[0]["answerability"] == "fully_answerable"
    assert copied_source_sentence(
        "이 회사의 주력 제품은 고다층 인쇄회로기판이다",
        "이 회사의 주력 제품은 고다층 인쇄회로기판이다\n다른 문장",
    )


def test_qrel_keeps_irrelevant_distinct_from_unjudged() -> None:
    assert qrel_state(None) == "unjudged"
    assert qrel_state(0) == "irrelevant"
    assert qrel_state(2) == "relevant"
    questions = [
        {
            "query_id": "Q",
            "seed_evidence_ids": ["seed"],
            "answerability": "fully_answerable",
            "split": "test",
        }
    ]
    judgments = [
        _judgment("Q", "u1", ["seed"], 2, document_name="사업보고서", filing_period="2025.12"),
        _judgment("Q", "u2", ["other"], 0, document_name="반기보고서", filing_period="2026.06"),
        {
            "query_id": "Q",
            "retrieval_unit_id": "u3",
            "element_ids": ["missing"],
            "final": None,
            "final_source": "unresolved",
            "adjudicated": False,
        },
    ]
    qrels = build_qrels(judgments, questions)
    states = {(row["element_id"], row["relevance"]) for row in qrels}
    assert ("seed", 2) in states
    assert ("other", 0) in states
    assert "missing" not in {row["element_id"] for row in qrels}
    preferred = {row["element_id"] for row in qrels if row["preferred"]}
    assert preferred == {"seed"}


def test_pool_dedup_and_stable_tail() -> None:
    units = {
        f"u{index}": _pool_unit(f"u{index}")
        for index in range(1, 31)
    }
    dense = [{"retrieval_unit_id": f"u{index}", "rank": index, "score": 1 / index}
             for index in range(1, 31)]
    bm25 = [
        {"retrieval_unit_id": "u2", "rank": 1, "score": 3.0},
        {"retrieval_unit_id": "u9", "rank": 2, "score": 2.0},
    ]
    pooling = {
        **POOLING,
        "tail_buckets": [{"name": "21-50", "start": 21, "end": 50, "n": 1}],
    }
    first = select_candidates(
        query_id="Q",
        answerability="fully_answerable",
        seed_unit_ids=["u1"],
        dense_ranked=dense,
        bm25_ranked=bm25,
        units_by_id=units,
        pooling=pooling,
    )
    second = select_candidates(
        query_id="Q",
        answerability="fully_answerable",
        seed_unit_ids=["u1"],
        dense_ranked=dense,
        bm25_ranked=bm25,
        units_by_id=units,
        pooling=pooling,
    )
    ids = [item["retrieval_unit_id"] for item in first["candidates"]]
    assert len(ids) == len(set(ids))
    shared = next(item for item in first["candidates"] if item["retrieval_unit_id"] == "u2")
    assert {source["retriever"]
            for source in shared["sources"]} >= {"dense_d2a", "bm25_ko_mixed_v1"}
    assert [item["retrieval_unit_id"] for item in first["candidates"]] == [
        item["retrieval_unit_id"] for item in second["candidates"]
    ]


def test_blind_payload_hides_retriever_fields() -> None:
    payload = judge_payload(
        "질문",
        [{"requirement_id": "R1", "text": "매출", "answerability": "fully_answerable"}],
        {
            "text": "본문",
            "document_name": "사업보고서",
            "filing_period": "2025.12",
            "section_path": ["II. 사업의 내용"],
            "receipt_date": "2026-03-01",
            "rank": 1,
            "score": 0.9,
            "retriever": "dense_d2a",
            "sources": [{"retriever": "bm25_ko_mixed_v1", "rank": 4}],
        },
    )
    dumped = json.dumps(payload, ensure_ascii=False)
    assert "dense_d2a" not in dumped
    assert "bm25" not in dumped
    assert "rank" not in dumped
    assert payload["evidence"]["text"] == "본문"


def test_disagreement_calls_adjudicator(tmp_path: Path) -> None:
    left = _client_with(2)
    right = _client_with(0)
    third = _client_with(1)
    row = judge_candidate(
        query_id="Q",
        question="매출은?",
        requirements=[
            {
                "requirement_id": "R1",
                "text": "매출",
                "answerability": "fully_answerable",
            }
        ],
        candidate=_pool_unit("u1") | {"candidate_id": "C01"},
        judge_a=left,
        judge_b=right,
        adjudicator=third,
        log_a=CallLog(tmp_path / "a.jsonl"),
        log_b=CallLog(tmp_path / "b.jsonl"),
        log_c=CallLog(tmp_path / "c.jsonl"),
        max_retries=1,
        reasoning_setting="provider_default",
    )
    assert row["adjudicated"] is True
    assert row["final_source"] == "adjudicator"
    assert row["final"]["passage_relevance"] == 1
    assert third.calls == 1
    assert not judgments_agree(row["judge_a"], row["judge_b"])


def test_freeze_hash_changes_when_a_qrel_changes() -> None:
    body = {"questions": ["매출"], "qrels": [{"element_id": "e", "relevance": 2}]}
    again = {"qrels": [{"relevance": 2, "element_id": "e"}], "questions": ["매출"]}
    assert sha256_text(canonical_dumps(body)) == sha256_text(canonical_dumps(again))
    changed = {"questions": ["매출"], "qrels": [{"element_id": "e", "relevance": 0}]}
    assert sha256_text(canonical_dumps(body)) != sha256_text(canonical_dumps(changed))


def test_unanswerable_is_outside_the_headline_and_requirements_are_partial() -> None:
    questions = [
        {"query_id": "A", "answerability": "fully_answerable", "split": "test"},
        {"query_id": "U", "answerability": "unanswerable", "split": "test"},
    ]
    requirements = [
        {"query_id": "A", "requirement_id": "R1", "text": "2025 매출",
            "answerability": "fully_answerable"},
        {"query_id": "A", "requirement_id": "R2", "text": "증가 이유",
            "answerability": "fully_answerable"},
    ]
    qrels = [
        {
            "query_id": "A",
            "requirement_id": "R1",
            "element_id": "e1",
            "retrieval_unit_id": "u1",
            "relevance": 2,
            "preferred": True,
            "document_name": "사업보고서",
            "filing_period": "2025.12",
        }
    ]
    hits = [
        {"rank": 1, "source_element_ids": ["unknown"]},
        {"rank": 2, "source_element_ids": ["e1"]},
    ]
    scored = score_question(questions[0], qrels, requirements, hits)
    assert scored["at_k"]["1"]["hit"] is False
    assert scored["at_k"]["1"]["unjudged_count"] == 1
    assert scored["hits"][0]["relevance"] is None
    assert scored["mrr"] == 0.5
    assert scored["requirement_covered@10"] == 1
    assert scored["requirement_total"] == 2
    assert scored["all_requirements_covered@10"] is False
    hidden = score_question(questions[1], [], [], hits)
    assert filter_headline(questions, [scored, hidden]) == [scored]


def test_logged_call_is_not_repeated(tmp_path: Path) -> None:
    client = _client_with(2)
    log = CallLog(tmp_path / "calls.jsonl")
    payload = {"question": "매출"}
    first = logged_complete(
        log,
        client,
        stage="generate",
        instructions="rules",
        payload=payload,
        schema=PassageJudgment,
        max_retries=1,
        reasoning_setting="provider_default",
    )
    second = logged_complete(
        log,
        client,
        stage="generate",
        instructions="rules",
        payload=payload,
        schema=PassageJudgment,
        max_retries=1,
        reasoning_setting="provider_default",
    )
    assert first["output"] == second["output"]
    assert client.calls == 1


class _FakeClient:
    def __init__(self, relevance: int) -> None:
        self.model = f"fake-{relevance}"
        self.provider = "openai"
        self.calls = 0
        self._relevance = relevance

    def complete(self, **_kwargs: object) -> tuple[dict, dict, float]:
        self.calls += 1
        output = {
            "requirement_judgments": [{"requirement_id": "R1", "relevance": self._relevance}],
            "passage_relevance": self._relevance,
        }
        return output, {"input_tokens": 3, "output_tokens": 2}, 1.0


def _client_with(relevance: int) -> _FakeClient:
    return _FakeClient(relevance)


def _judgment(
    query_id: str,
    unit_id: str,
    element_ids: list[str],
    relevance: int,
    *,
    document_name: str,
    filing_period: str,
) -> dict:
    final = {
        "passage_relevance": relevance,
        "requirement_judgments": [{"requirement_id": "R1", "relevance": relevance}],
    }
    return {
        "query_id": query_id,
        "retrieval_unit_id": unit_id,
        "element_ids": element_ids,
        "document_name": document_name,
        "filing_period": filing_period,
        "final": final,
        "final_source": "agreement",
        "adjudicated": False,
        "judge_a": final,
        "judge_b": final,
    }


def _pool_unit(unit_id: str) -> dict:
    return {
        "retrieval_unit_id": unit_id,
        "element_ids": [unit_id],
        "text": f"본문 {unit_id}",
        "document_name": "사업보고서",
        "filing_period": "2025.12",
        "section_path": ["II. 사업의 내용"],
        "receipt_no": "20260101000001",
        "receipt_date": "2026-03-01",
    }


def _fake_units(corp: str) -> list[dict]:
    rows = []
    topics = {
        "business": "paragraph",
        "overview": "paragraph",
        "mdna": "paragraph",
        "finance": "paragraph",
        "governance": "paragraph",
    }
    index = 0
    for topic, form in topics.items():
        for copy in range(8):
            rows.append(_fake_unit(corp, topic, form, "2024.12", "사업보고서", 1, index))
            index += 1
            rows.append(_fake_unit(corp, topic, "table", "2025.12", "반기보고서", 1, index))
            index += 1
            rows.append(_fake_unit(corp, topic, "table", "2026.06", "분기보고서", 1, index))
            index += 1
    for copy in range(6):
        rows.append(_fake_unit(corp, "business", "paragraph", "2025.12", "사업보고서", 2, index))
        index += 1
    return rows


def _fake_unit(
    corp: str,
    topic: str,
    form: str,
    period: str,
    document: str,
    n_elements: int,
    index: int,
) -> dict:
    unit_id = f"{corp}-{topic}-{form}-{period}-{index}"
    marker = "수치 | 금액" if form == "table" else "문장"
    body = f"{topic} {form} {period} 설명 문단 {index} " + marker
    return {
        "corp_code": corp,
        "retrieval_unit_id": unit_id,
        "element_ids": [f"{unit_id}-e{item}" for item in range(n_elements)],
        "text": body,
        "body": body,
        "document_name": document,
        "report_name": document,
        "filing_period": period,
        "receipt_no": f"{index:014d}",
        "receipt_date": "2026-03-01",
        "section_path": [topic],
        "topic_bucket": topic,
        "surface_form": form,
        "n_elements": n_elements,
        "prefix_hash": unit_id,
        "has_cause_language": False,
    }
