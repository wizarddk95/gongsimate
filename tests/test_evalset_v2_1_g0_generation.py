"""G0 grounded generation. These tests do not call a model or search."""

from __future__ import annotations

from pathlib import Path

import pytest

from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2.spec import repo_root
from disclosure_agent.evalset_v2_1.grounded_generation import (
    GENERATOR_INSTRUCTIONS,
    RequirementSignals,
    assemble_query_result,
    assert_no_gold,
    build_model_evidence,
    check_stored_h10_ranking,
    citation_integrity,
    classify_requirement,
    load_g0_packets,
    resolve_g0_models,
    select_representative_cases,
    summarize_results,
)


def test_generator_instructions_keep_the_grounding_rules() -> None:
    required = (
        "입력 evidence의 text만",
        "직접 확인할 수 없는",
        "requirements를 모두",
        "연결/별도",
        "원인을 만들지 않습니다",
        "citation_id만",
        "일부만 지지하면",
        "입력 evidence에 있는 값만",
    )
    for phrase in required:
        assert phrase in GENERATOR_INSTRUCTIONS


def test_model_names_come_from_the_environment() -> None:
    resolved = resolve_g0_models({"OPENAI_MODEL": "model-a", "OPENAI_API_KEY": "present"})
    assert resolved["generator_model"] == "model-a"
    assert resolved["judge_model"] == "model-a"
    custom = resolve_g0_models(
        {
            "OPENAI_MODEL": "model-a",
            "G0_JUDGE_MODEL": "model-b",
            "OPENAI_API_KEY": "present",
        }
    )
    assert custom["judge_model"] == "model-b"
    with pytest.raises(UsageError):
        resolve_g0_models({"OPENAI_API_KEY": "present"})


def test_model_evidence_hides_grades_and_locators() -> None:
    evidence = build_model_evidence(
        1,
        {
            "retrieval_unit_id": "unit-1",
            "text": "매출은 10이다.",
            "document_name": "사업보고서",
            "report_name": "사업보고서",
            "report_type": "annual",
            "filing_period": "2024",
            "period_end_year": 2024,
            "period_end_month": 12,
            "receipt_date": "20250301",
            "receipt_no": "20250301000001",
            "section_path": ["본문"],
            "relevance": 2,
            "preferred": True,
            "source_refs": [{"element_id": "element-1", "char_start": 1, "char_end": 8}],
            "source_element_ids": ["element-1"],
        },
        "이수페타시스",
    )
    assert evidence["citation_id"] == "C1"
    assert evidence["text"] == "매출은 10이다."
    assert "relevance" not in evidence
    assert "source_refs" not in evidence
    payload = {"question": "매출은?", "requirements": [], "evidence": [evidence]}
    assert_no_gold(payload)
    with pytest.raises(UsageError):
        assert_no_gold({**payload, "qrels": []})


def test_invalid_citations_stay_out_of_the_correctness_denominator() -> None:
    integrity = citation_integrity(
        "매출은 10이다. [C1] [nope]",
        [{"claim_id": "K1", "text": "매출은 10이다.", "citation_ids": ["C9", "C1"]}],
        {"C1"},
    )
    assert {item["token"] for item in integrity["unknown"]} == {"C9"}
    assert {item["token"] for item in integrity["malformed"]} == {"nope"}
    assert integrity["integrity_error_count"] == 2
    result = assemble_query_result(
        _packet({"R1": True}),
        _generation(
            [
                {
                    "claim_id": "K1",
                    "text": "매출은 10이다.",
                    "citation_ids": ["C9"],
                    "factual": True,
                }
            ],
            "매출은 10이다. [C9]",
        ),
        _faithfulness([("K1", "supported", "none")]),
        _requirements([("R1", "covered", ["K1"])]),
        _citations([]),
    )
    assert result["citation"]["score"] is None
    assert result["citation"]["does_not_support"] == 0
    assert result["citation"]["completeness"] == 0
    assert result["integrity"]["integrity_error_count"] == 2
    assert "citation_hallucination" in result["query_flags"]


def test_unjudged_requirement_is_not_counted_as_missed() -> None:
    result = assemble_query_result(
        _packet({"R1": True, "R2": True}),
        _generation([_claim("K1", "C1")], "매출은 10이다. [C1]"),
        _faithfulness([("K1", "supported", "none")]),
        _requirements([("R1", "covered", ["K1"])]),
        _citations([("K1", "C1", "supports")]),
    )
    coverage = result["requirement_coverage"]
    assert coverage["covered"] == 1
    assert coverage["omitted"] == 0
    assert coverage["unjudged"] == 1
    assert coverage["score"] == 1
    summary = summarize_results([result], _verification())
    assert summary["requirement_coverage"]["denominator"] == 1
    assert summary["requirement_coverage"]["unjudged"] == 1


def test_missing_evidence_is_retrieval_limited_not_omission() -> None:
    result = assemble_query_result(
        _packet({"R1": False}),
        _generation([_claim("K1", "C1")], "확인할 수 없다."),
        _faithfulness([("K1", "not_factual", "none")]),
        _requirements([("R1", "omitted", [])]),
        _citations([]),
    )
    row = result["requirements"][0]
    assert row["primary"] == "retrieval_limited"
    assert "retrieval_failure" in row["flags"]
    assert "generation_omission" not in row["flags"]


def test_present_evidence_and_omission_is_generation_limited() -> None:
    result = assemble_query_result(
        _packet({"R1": True}),
        _generation([], "다른 이야기."),
        _faithfulness([]),
        _requirements([("R1", "omitted", [])]),
        _citations([]),
    )
    row = result["requirements"][0]
    assert row["primary"] == "generation_limited"
    assert row["flags"] == ["generation_omission"]


def test_supported_answer_with_evidence_is_success() -> None:
    result = _success_result()
    assert result["requirements"][0]["primary"] == "success"
    assert result["faithfulness"]["score"] == 1
    assert result["citation"]["score"] == 1
    cases = select_representative_cases([result])
    assert cases[0]["role"] == "clean_success"


def test_wrong_claim_with_evidence_is_unsupported() -> None:
    decision = classify_requirement(
        RequirementSignals(
            evidence_in_top10=True,
            coverage="covered",
            support="unsupported",
            scope_issue="none",
            citation_missing=False,
            citation_mismatch=False,
            citation_hallucination=False,
        )
    )
    assert decision["primary"] == "generation_limited"
    assert decision["flags"] == ["unsupported_claim"]


def test_scope_confusion_demotes_an_otherwise_supported_claim() -> None:
    decision = classify_requirement(
        RequirementSignals(
            evidence_in_top10=True,
            coverage="covered",
            support="supported",
            scope_issue="period",
            citation_missing=False,
            citation_mismatch=False,
            citation_hallucination=False,
        )
    )
    assert decision["primary"] == "generation_limited"
    assert "period_scope_confusion" in decision["flags"]


def test_stored_h10_ranking_is_unchanged() -> None:
    verification = check_stored_h10_ranking(repo_root())
    assert verification["retrieval_reexecuted"] is False
    assert verification["test_questions_generated"] == 0
    assert verification["question_count"] == 45
    assert verification["hit10"] == {"numerator": 43, "denominator": 45}
    assert verification["requirement_recall"] == {"numerator": 52, "denominator": 58}
    assert verification["retrieval_policy"] == "bm25_primary_dense_head"


def test_stored_h10_text_join_does_not_search() -> None:
    index = (
        repo_root()
        / "data"
        / "product"
        / "indexes"
        / "isu_petasys_periodic_d2a"
        / "retrieval_units.jsonl"
    )
    if not index.is_file():
        pytest.skip("local retrieval unit index is not in git")
    population = load_g0_packets(repo_root())
    assert population["verification"]["retrieval_reexecuted"] is False
    assert len(population["packets"]) == 45
    assert population["verification"]["requirement_count"] == 58
    for packet in population["packets"]:
        assert len(packet["model_payload"]["evidence"]) == 10
        assert_no_gold(packet["model_payload"])
        assert [row["citation_id"] for row in packet["model_payload"]["evidence"]] == [
            f"C{rank}" for rank in range(1, 11)
        ]


def _success_result() -> dict:
    return assemble_query_result(
        _packet({"R1": True}),
        _generation([_claim("K1", "C1")], "매출은 10이다. [C1]"),
        _faithfulness([("K1", "supported", "none")]),
        _requirements([("R1", "covered", ["K1"])], "direct"),
        _citations([("K1", "C1", "supports")]),
    )


def _packet(evidence_flags: dict[str, bool]) -> dict:
    evidence = [
        {
            "citation_id": "C1",
            "retrieval_unit_id": "unit-1",
            "filing_company": "대상",
            "document_name": "사업보고서",
            "report_name": "사업보고서",
            "report_type": "annual",
            "filing_period": "2024",
            "period_end_year": "2024",
            "period_end_month": "12",
            "receipt_date": "20250301",
            "receipt_no": "20250301000001",
            "section_path": ["본문"],
            "text": "매출은 10이다.",
        }
    ]
    requirements = [
        {"requirement_id": requirement_id, "text": requirement_id}
        for requirement_id in evidence_flags
    ]
    return {
        "query_id": "Q1",
        "question": "매출은 얼마인가?",
        "company_name": "대상",
        "query_type": "single_numeric",
        "evidence_form": "paragraph",
        "stratum_id": "finance_paragraph",
        "period_sensitive": True,
        "difficulty_tags": ["period_sensitive"],
        "requirements": requirements,
        "model_payload": {
            "question": "매출은 얼마인가?",
            "requirements": requirements,
            "evidence": evidence,
        },
        "resolution": [],
        "retrieval_unit_ids": ["unit-1"],
        "evidence_in_top10": evidence_flags,
    }


def _claim(claim_id: str, citation_id: str) -> dict:
    return {
        "claim_id": claim_id,
        "text": "매출은 10이다.",
        "citation_ids": [citation_id],
        "factual": True,
    }


def _generation(claims: list[dict], answer: str) -> dict:
    output = {"answer": answer, "claims": claims, "limitations": []}
    return _call("ok", output)


def _faithfulness(rows: list[tuple[str, str, str]]) -> dict:
    return _call(
        "ok",
        {
            "items": [
                {
                    "claim_id": claim_id,
                    "verdict": verdict,
                    "scope_issue": scope,
                    "reason": "근거",
                }
                for claim_id, verdict, scope in rows
            ],
            "reason": "faithfulness",
        },
    )


def _requirements(rows: list[tuple[str, str, list[str]]], relevance: str = "direct") -> dict:
    return _call(
        "ok",
        {
            "items": [
                {
                    "requirement_id": requirement_id,
                    "verdict": verdict,
                    "claim_ids": claim_ids,
                    "reason": "coverage",
                }
                for requirement_id, verdict, claim_ids in rows
            ],
            "answer_relevance": relevance,
            "reason": "requirements",
        },
    )


def _citations(rows: list[tuple[str, str, str]]) -> dict:
    return _call(
        "ok",
        {
            "items": [
                {
                    "claim_id": claim_id,
                    "citation_id": citation_id,
                    "verdict": verdict,
                    "reason": "citation",
                }
                for claim_id, citation_id, verdict in rows
            ],
            "reason": "citations",
        },
    )


def _call(status: str, output: dict | None) -> dict:
    return {
        "parse_status": status,
        "output": output,
        "raw": output,
        "provider": "openai",
        "model": "model-a",
        "prompt_version": "g0-test",
        "prompt_sha256": "abc",
        "evaluator_version": "g0-answer-evaluation-v1",
        "decoding": {},
        "latency_ms": 10,
        "usage": {"input_tokens": 5, "output_tokens": 2},
        "estimated_cost_usd": 0.001,
        "error": None,
        "input_sha256": "hash",
        "retry_count": 0,
        "reason": "" if output is None else output.get("reason", ""),
        "answer_relevance": None if output is None else output.get("answer_relevance"),
    }


def _verification() -> dict:
    return {
        "dataset_id": "multi_company_retrieval_v2_1",
        "dataset_sha256": "abc",
        "question_count": 1,
        "requirement_count": 1,
        "h10_run": "multi_company_retrieval_v2_1_dev_h10_asymmetric",
        "retrieval_policy": "bm25_primary_dense_head",
        "retrieval_reexecuted": False,
        "test_questions_generated": 0,
        "hit10": {"numerator": 43, "denominator": 45},
        "requirement_recall": {"numerator": 52, "denominator": 58},
        "all_requirements": {"numerator": 42, "denominator": 45},
        "requirement_texts_containing_digits": 0,
    }


def test_report_names_retrieval_and_generation_separately(tmp_path: Path) -> None:
    del tmp_path
    summary = summarize_results([_success_result()], _verification())
    assert summary["generation_limited_requirements"] == 0
    assert summary["retrieval_limited_requirements"] == 0
    assert summary["faithfulness"]["denominator_definition"] == "judged_only"
